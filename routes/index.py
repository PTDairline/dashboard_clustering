from flask import render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename
import os
import logging
from utils.data_processing import read_data, handle_null_values, allowed_file

# Comment: Import các thư viện và module cần thiết.
# - Flask: Dùng để xử lý request, render giao diện, và quản lý ứng dụng web.
# - werkzeug.utils.secure_filename: Dùng để bảo mật tên file khi lưu.
# - os, logging: Dùng cho xử lý file và ghi log.
# - read_data, handle_null_values, allowed_file: Các hàm từ module utils.data_processing để đọc và xử lý dữ liệu.

def index():
    # Comment: Khởi tạo dictionary `data` để lưu trạng thái cho giao diện.
    # - `file_uploaded`: Trạng thái file đã được tải hay chưa (mặc định là False).
    data = {
        'file_uploaded': False
    }
    
    if request.method == 'POST':
        # Comment: Xử lý yêu cầu POST khi người dùng tải file lên.
        if 'file' in request.files:
            file = request.files['file']
            # Comment: Kiểm tra xem file có hợp lệ không (được phép tải lên).
            if file and allowed_file(file.filename):
                # Comment: Xóa các file trung gian trước khi xử lý file mới.
                # - Đảm bảo pipeline bắt đầu lại từ đầu, tránh xung đột.
                files_to_remove = [
                    'data.pkl', 'selected_features.txt', 'processed_data.pkl',
                    'use_pca.txt', 'explained_variance.txt', 'pca_result.csv',
                    'pca_results.json', 'clustering_results.json'
                ]
                for file_name in files_to_remove:
                    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], file_name)  # Sửa request.app thành current_app
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.debug(f"Removed intermediate file: {file_path}")
                
                # Comment: Lưu file tải lên với tên an toàn.
                # - Sử dụng `secure_filename` để bảo mật tên file.
                # - Lưu file vào thư mục `UPLOAD_FOLDER`.
                filename = secure_filename(file.filename)
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)  # Sửa request.app thành current_app
                file.save(file_path)
                
                # Comment: Đọc dữ liệu từ file và kiểm tra lỗi.
                # - Sử dụng hàm `read_data` từ module utils để đọc file.
                # - Nếu đọc thất bại, thông báo lỗi và trả về giao diện.
                df, error = read_data(file_path)
                if df is None:
                    flash(f"Lỗi đọc file: {error}")
                    return render_template('index.html', data=data)
                
                # Comment: Lưu DataFrame vào file `data.pkl` để sử dụng ở các bước sau.
                df.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))  # Sửa request.app thành current_app
                
                # Comment: Xử lý giá trị null trong dữ liệu.
                # - Sử dụng hàm `handle_null_values` để loại bỏ các cột có quá nhiều giá trị null.
                # - Thông báo nếu có cột bị xóa.
                df, dropped_columns = handle_null_values(df)
                if dropped_columns:
                    flash(f"Xóa {len(dropped_columns)} cột nhiều null: {', '.join(dropped_columns)}")
                
                # Comment: Chuyển hướng đến trang preview dữ liệu sau khi xử lý thành công.
                return redirect(url_for('data_preview'))
    
    # Comment: Trả về giao diện mặc định (`index.html`) nếu không có yêu cầu POST.
    return render_template('index.html', data=data)