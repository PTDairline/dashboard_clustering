from flask import render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename
import os
import logging
from utils.data_processing import read_data, handle_null_values, allowed_file

def index():
    data = {
        'file_uploaded': False
    }
    
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Xóa các file trung gian trước khi xử lý file mới
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
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)  # Sửa request.app thành current_app
                file.save(file_path)
                df, error = read_data(file_path)
                if df is None:
                    flash(f"Lỗi đọc file: {error}")
                    return render_template('index.html', data=data)
                
                df.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))  # Sửa request.app thành current_app
                df, dropped_columns = handle_null_values(df)
                if dropped_columns:
                    flash(f"Xóa {len(dropped_columns)} cột nhiều null: {', '.join(dropped_columns)}")
                
                return redirect(url_for('data_preview'))
    
    return render_template('index.html', data=data)