from flask import render_template, request, redirect, url_for, flash, current_app
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering import generate_clustering_plots
from utils.metrics import suggest_optimal_k

def select_model():
    data = {
        'k_range': list(range(2, 11)),
        'models': [],
        'selected_k': 2,
        'plots': {},
        'optimal_k_suggestions': {},
        'use_pca': False,
        'explained_variance_ratio': None,
        'selected_features': []
    }
    
    processed_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')
    if not os.path.exists(processed_data_file):
        flash("Vui lòng xử lý dữ liệu trước.")
        return redirect(url_for('process_data'))
    
    X = pd.read_pickle(processed_data_file)
    use_pca_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt')
    if os.path.exists(use_pca_file):
        with open(use_pca_file, 'r') as f:
            data['use_pca'] = f.read().strip() == 'True'
    
    if data['use_pca']:
        explained_variance_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt')
        if os.path.exists(explained_variance_file):
            with open(explained_variance_file, 'r') as f:
                data['explained_variance_ratio'] = float(f.read().strip())
    
    selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
    if os.path.exists(selected_features_file):
        try:
            with open(selected_features_file, 'r') as f:
                data['selected_features'] = f.read().split(',')
        except Exception as e:
            logging.error(f"Error reading selected_features.txt: {str(e)}")
            flash("Lỗi khi đọc danh sách feature đã chọn. Vui lòng chọn lại feature.")
    
    clustering_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'clustering_results.json')
    if os.path.exists(clustering_results_file):
        try:
            with open(clustering_results_file, 'r') as f:
                clustering_results = json.load(f)
            data['models'] = clustering_results.get('models', [])
            data['selected_k'] = clustering_results.get('selected_k', 2)
            data['plots'] = clustering_results.get('plots', {})
            data['optimal_k_suggestions'] = clustering_results.get('optimal_k_suggestions', {})
        except Exception as e:
            logging.error(f"Error reading clustering_results.json: {str(e)}")
            flash("Lỗi khi đọc kết quả phân cụm trước đó. Vui lòng chạy lại mô hình.")
    
    if request.method == 'POST':
        models = request.form.getlist('models')
        selected_k = int(request.form.get('k'))
        
        # Kiểm tra selected_k
        if selected_k < 4:
            flash("Số cụm tối đa (k) phải lớn hơn hoặc bằng 4 để áp dụng phương pháp khuỷu tay.")
            return render_template('select_model.html', data=data)
        
        if not models:
            flash("Vui lòng chọn ít nhất một mô hình.")
            return render_template('select_model.html', data=data)
        
        # Kiểm tra dữ liệu trước khi chạy phân cụm
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty or len(X_numeric.columns) < 2:
            flash("Dữ liệu không chứa đủ cột số (cần ít nhất 2 cột số). Vui lòng kiểm tra và xử lý lại dữ liệu.")
            return render_template('select_model.html', data=data)
        
        if X_numeric.isna().any().any():
            flash("Dữ liệu chứa giá trị NaN. Vui lòng xử lý dữ liệu trước khi chạy phân cụm.")
            return render_template('select_model.html', data=data)
        
        if np.isinf(X_numeric.values).any():
            flash("Dữ liệu chứa giá trị vô cực (inf). Vui lòng xử lý dữ liệu trước khi chạy phân cụm.")
            return render_template('select_model.html', data=data)
        
        # Reset models để chỉ lưu các mô hình thành công
        data['models'] = []
        data['selected_k'] = selected_k
        
        for model in models:
            plots = generate_clustering_plots(X, model, range(2, selected_k + 1), selected_k, data['use_pca'], data['selected_features'], data['explained_variance_ratio'])
            if 'error' in plots:
                flash(plots['error'])
                continue  # Bỏ qua mô hình này và tiếp tục với mô hình tiếp theo
            data['plots'][model] = plots
            
            optimal_k, reasoning = suggest_optimal_k(plots, list(range(2, selected_k + 1)))
            data['optimal_k_suggestions'][model] = {'k': optimal_k, 'reasoning': reasoning}
            
            # Chỉ thêm mô hình vào data['models'] nếu thành công
            if 'cvi' in plots:
                data['models'].append(model)
            else:
                flash(f"Mô hình {model} không thể chạy thành công. Vui lòng kiểm tra dữ liệu hoặc thử lại.")
        
        # Lưu kết quả nếu có mô hình thành công
        if data['models']:
            clustering_results = {
                'models': data['models'],
                'selected_k': data['selected_k'],
                'plots': data['plots'],
                'optimal_k_suggestions': data['optimal_k_suggestions']
            }
            with open(clustering_results_file, 'w') as f:
                json.dump(clustering_results, f)
        
        return render_template('select_model.html', data=data)
    
    return render_template('select_model.html', data=data)