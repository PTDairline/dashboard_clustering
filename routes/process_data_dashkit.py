from flask import render_template, request, redirect, url_for, flash, send_file, current_app
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.pca import perform_pca
from utils.data_processing import handle_null_values

# Danh sách feature mặc định
DEFAULT_FEATURES = [
    "Age", "Overall", "Potential", "Value", "Wage", "Height", "Weight",
    "Crossing", "Finishing", "HeadingAccuracy", "ShortPassing", "Volleys",
    "Dribbling", "Curve", "FKAccuracy", "LongPassing", "BallControl",
    "Acceleration", "SprintSpeed", "Agility", "Reactions", "Balance",
    "ShotPower", "Jumping", "Stamina", "Strength", "LongShots", "Aggression",
    "Interceptions", "Positioning", "Vision", "Penalties", "Composure",
    "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling",
    "GKKicking", "GKPositioning", "GKReflexes", "Release Clause"
]

def process_data_dashkit():
    # Khởi tạo dictionary `data` để lưu trạng thái và dữ liệu cho giao diện.
    data = {
        'features': [],
        'numerical_features': [],
        'categorical_features': [],
        'selected_features': [],
        'num_features': 0,
        'feature_types': {},
        'pca_results': None,
        'pca_fig_html': None,
        'proceed_to_model': False
    }
    
    # Kiểm tra nếu file data.pkl tồn tại
    if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
        flash("Không tìm thấy dữ liệu. Vui lòng tải file lên trước.")
        return redirect(url_for('dashkit_index'))
    
    # Đọc dữ liệu
    df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
    data['features'] = df.columns.tolist()
    data['num_features'] = len(data['features'])
    data['feature_types'] = {col: str(df[col].dtype) for col in df.columns}
    data['num_rows'] = df.shape[0]
    
    # Phân loại features
    data['numerical_features'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    data['categorical_features'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Xử lý POST request
    if request.method == 'POST':
        # Xử lý chọn feature
        if 'select_features' in request.form:
            feature_option = request.form.get('feature_option', 'default')
            
            if feature_option == 'default':
                # Sử dụng danh sách feature mặc định
                selected_features = [f for f in DEFAULT_FEATURES if f in df.columns]
                if not selected_features:
                    # Nếu không có feature nào trong danh sách mặc định phù hợp, dùng tất cả feature số
                    selected_features = data['numerical_features']
            else:
                # Sử dụng feature được chọn
                selected_features = request.form.getlist('features')
            
            if not selected_features:
                flash("Vui lòng chọn ít nhất một feature.")
                return render_template('process_data_dashkit.html', data=data)
            
            # Lưu danh sách feature đã chọn
            with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'), 'w') as f:
                f.write(','.join(selected_features))
            
            data['selected_features'] = selected_features
            flash("Đã chọn feature thành công.")
        
        # Xử lý phân tích PCA
        elif 'run_analysis' in request.form:
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')):
                flash("Vui lòng chọn feature trước khi thực hiện phân tích.")
                return render_template('process_data_dashkit.html', data=data)
            
            # Đọc danh sách feature đã chọn
            with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'), 'r') as f:
                selected_features = f.read().strip().split(',')
            
            # Đọc tùy chọn xử lý
            process_method = request.form.get('process_method', 'pca')
            
            if process_method == 'pca':
                # Thực hiện PCA
                explained_variance = float(request.form.get('explained_variance', 90))
                
                try:
                    # Xử lý dữ liệu null
                    df_processed = handle_null_values(df[selected_features])
                    
                    # Thực hiện PCA
                    pca_results, n_components, pca, explained_variance_ratio = perform_pca(df_processed, explained_variance / 100)
                    
                    # Lưu kết quả PCA
                    pca_df = pd.DataFrame(pca_results, columns=[f'PC{i+1}' for i in range(n_components)])
                    pca_df.to_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_result.csv'), index=False)
                    
                    # Lưu các tham số PCA
                    pca_params = {
                        'n_components': int(n_components),
                        'explained_variance_ratio': explained_variance_ratio.tolist()
                    }
                    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json'), 'w') as f:
                        json.dump(pca_params, f)
                    
                    # Lưu các tham số khác
                    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt'), 'w') as f:
                        f.write('1')
                    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt'), 'w') as f:
                        f.write(str(explained_variance))
                    
                    # Lưu dữ liệu đã xử lý
                    df_processed.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl'))
                    
                    # Cập nhật thông tin PCA
                    data['pca_results'] = {
                        'n_components': n_components,
                        'explained_variance': explained_variance,
                        'explained_variance_ratio': [round(x * 100, 2) for x in explained_variance_ratio],
                        'cumulative_variance': round(sum(explained_variance_ratio) * 100, 2)
                    }
                    
                    data['proceed_to_model'] = True
                    flash(f"Đã thực hiện PCA thành công với {n_components} thành phần chính.")
                    
                except Exception as e:
                    logging.error(f"PCA error: {str(e)}")
                    flash(f"Lỗi khi thực hiện PCA: {str(e)}")
            else:
                # Sử dụng feature đã chọn mà không PCA
                try:
                    # Xử lý dữ liệu null
                    df_processed = handle_null_values(df[selected_features])
                    
                    # Lưu dữ liệu đã xử lý
                    df_processed.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl'))
                    
                    # Lưu các tham số
                    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt'), 'w') as f:
                        f.write('0')
                    
                    data['proceed_to_model'] = True
                    flash("Đã xử lý dữ liệu thành công với các feature đã chọn.")
                    
                except Exception as e:
                    logging.error(f"Data processing error: {str(e)}")
                    flash(f"Lỗi khi xử lý dữ liệu: {str(e)}")
    
    # Kiểm tra các file liên quan
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')):
        with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'), 'r') as f:
            content = f.read().strip()
            if content:
                data['selected_features'] = content.split(',')
    
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')):
        with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json'), 'r') as f:
            pca_params = json.load(f)
            data['pca_results'] = {
                'n_components': pca_params['n_components'],
                'explained_variance_ratio': [round(x * 100, 2) for x in pca_params['explained_variance_ratio']],
                'cumulative_variance': round(sum(pca_params['explained_variance_ratio']) * 100, 2)
            }
        
        if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt')):
            with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt'), 'r') as f:
                data['pca_results']['explained_variance'] = float(f.read().strip())
    
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')):
        data['proceed_to_model'] = True
    
    return render_template('process_data_dashkit.html', data=data)
