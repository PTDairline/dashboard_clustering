from flask import render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename
import os
import logging
from utils.data_processing import read_data, handle_null_values, allowed_file

def index_dashkit():
    # Khởi tạo dictionary `data` để lưu trạng thái cho giao diện.
    data = {
        'file_uploaded': False,
        'preview_data': None,
        'features': [],
        'num_features': 0,
        'num_rows': 0
    }
    
    if request.method == 'POST':
        # Xử lý yêu cầu POST khi người dùng tải file lên.
        if 'file' in request.files:
            file = request.files['file']
            # Kiểm tra xem file có hợp lệ không (được phép tải lên).
            if file and allowed_file(file.filename):
                # Xóa các file trung gian trước khi xử lý file mới.
                # Đảm bảo pipeline bắt đầu lại từ đầu, tránh xung đột.
                files_to_remove = [
                    'data.pkl', 'selected_features.txt', 'processed_data.pkl',
                    'use_pca.txt', 'explained_variance.txt', 'pca_result.csv',
                    'pca_results.json', 'clustering_results.json'
                ]
                for file_name in files_to_remove:
                    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.debug(f"Removed intermediate file: {file_path}")
                
                # Lưu file tải lên với tên an toàn.
                filename = secure_filename(file.filename)
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Đọc dữ liệu từ file và kiểm tra lỗi.
                df, error = read_data(file_path)
                if df is None:
                    flash(f"Lỗi đọc file: {error}")
                    return render_template('index_dashkit.html', data=data)
                
                # Hiển thị thông tin cơ bản về dữ liệu.
                data['file_uploaded'] = True
                data['num_features'] = df.shape[1]
                data['num_rows'] = df.shape[0]
                data['filename'] = filename
                data['features'] = df.columns.tolist()
                data['feature_types'] = {col: str(df[col].dtype) for col in df.columns}
                
                # Lưu DataFrame để sử dụng ở các trang khác.
                df_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')
                df.to_pickle(df_path)
                
                # Xử lý các yêu cầu POST cho phần chọn feature.
                # Nếu `select_features` được gửi lên, xử lý việc chọn các feature.
                if 'select_features' in request.form:
                    # Xác định feature được chọn dựa trên tùy chọn 'feature_option'.
                    feature_option = request.form.get('feature_option', 'default')
                    
                    selected_features = []
                    if feature_option == 'default':
                        # Nếu là 'default', sử dụng tất cả feature.
                        selected_features = data['features']
                    else:
                        # Nếu là 'custom', sử dụng các feature được chọn trong form.
                        # Form trả về danh sách feature được chọn qua `request.form.getlist('features')`.
                        selected_features = request.form.getlist('features')
                        if not selected_features or len(selected_features) < 2:
                            # Kiểm tra xem đã chọn ít nhất 2 feature chưa.
                            flash("Vui lòng chọn ít nhất 2 feature.")
                            return render_template('index_dashkit.html', data=data)
                    
                    # Lưu danh sách feature đã chọn vào `data`.
                    data['selected_features'] = selected_features
                    
                    # Lưu danh sách feature đã chọn vào file để sử dụng ở các trang khác.
                    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'), 'w') as f:
                        f.write(','.join(selected_features))
                
                # Xử lý yêu cầu phân tích dữ liệu.
                if 'run_analysis' in request.form:
                    # Lấy phương pháp xử lý dữ liệu (PCA hoặc không PCA).
                    process_method = request.form.get('process_method', 'pca')
                    # Lưu phương pháp vào file để sử dụng ở các trang khác.
                    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt'), 'w') as f:
                        f.write('true' if process_method == 'pca' else 'false')
                    
                    # Lấy tỷ lệ phương sai giải thích nếu sử dụng PCA.
                    if process_method == 'pca':
                        explained_variance = request.form.get('explained_variance', '90')
                        # Lưu tỷ lệ phương sai vào file để sử dụng ở các trang khác.
                        with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt'), 'w') as f:
                            f.write(explained_variance)
                    
                    # Đánh dấu là có thể tiến hành xử lý dữ liệu.
                    data['proceed_to_model'] = True
                    return redirect(url_for('process_data'))
    
    # Check if data.pkl exists    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
        # Data file exists, load it and show the UI
        try:
            import pandas as pd
            df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
            data['file_uploaded'] = True
            data['num_features'] = df.shape[1]
            data['num_rows'] = df.shape[0]
            data['features'] = df.columns.tolist()
            data['feature_types'] = {col: str(df[col].dtype) for col in df.columns}
            data['filename'] = os.path.basename(df.name) if hasattr(df, 'name') else "uploaded_file"
            
            # Thêm dữ liệu preview cho trang chủ
            data['preview_data'] = df.head(5).to_dict('records')
            
            # Check if selected_features.txt exists
            if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')):
                with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'), 'r') as f:
                    content = f.read().strip()
                    if content:
                        data['selected_features'] = content.split(',')
            
            # Check if processed_data.pkl exists
            if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')):
                data['proceed_to_model'] = True
        except Exception as e:
            flash(f"Error loading data: {str(e)}")
            
    # Render với template Dashkit mới
    return render_template('index_dashkit.html', data=data)
