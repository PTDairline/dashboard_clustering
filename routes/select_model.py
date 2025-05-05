from flask import render_template, request, redirect, url_for, flash, current_app
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering import generate_clustering_plots
from utils.metrics import suggest_optimal_k
from joblib import Parallel, delayed

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_clustering(model, X, k_range, selected_k, use_pca, selected_features, explained_variance):
    """Hàm hỗ trợ chạy phân cụm cho một mô hình, dùng trong joblib."""
    logging.debug(f"Chạy phân cụm cho mô hình {model} với selected_k={selected_k}")
    plots = generate_clustering_plots(X, model, k_range, selected_k, use_pca, selected_features, explained_variance)
    return model, plots

def select_model():
    logging.debug("Bắt đầu hàm select_model")
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
    
    # Kiểm tra file dữ liệu đã xử lý
    processed_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')
    if not os.path.exists(processed_data_file):
        logging.error("Không tìm thấy processed_data.pkl")
        flash("Vui lòng xử lý dữ liệu trước.")
        return redirect(url_for('process_data'))
    
    # Đọc dữ liệu
    logging.debug(f"Đọc dữ liệu từ {processed_data_file}")
    X = pd.read_pickle(processed_data_file)
    logging.debug(f"Kích thước dữ liệu: {X.shape}")
    
    # Kiểm tra sử dụng PCA
    use_pca_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt')
    if os.path.exists(use_pca_file):
        with open(use_pca_file, 'r') as f:
            data['use_pca'] = f.read().strip() == 'True'
        logging.debug(f"Use PCA: {data['use_pca']}")
    
    # Đọc explained_variance_ratio nếu dùng PCA
    if data['use_pca']:
        explained_variance_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt')
        if os.path.exists(explained_variance_file):
            with open(explained_variance_file, 'r') as f:
                data['explained_variance_ratio'] = float(f.read().strip())
            logging.debug(f"Explained variance ratio: {data['explained_variance_ratio']}")
    
    # Đọc danh sách feature đã chọn
    selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
    if os.path.exists(selected_features_file):
        try:
            with open(selected_features_file, 'r') as f:
                data['selected_features'] = f.read().split(',')
            logging.debug(f"Selected features: {data['selected_features']}")
        except Exception as e:
            logging.error(f"Lỗi đọc selected_features.txt: {str(e)}")
            flash("Lỗi khi đọc danh sách feature đã chọn. Vui lòng chọn lại feature.")
    
    # Đọc kết quả phân cụm đã lưu
    clustering_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'clustering_results.json')
    if os.path.exists(clustering_results_file):
        try:
            with open(clustering_results_file, 'r') as f:
                clustering_results = json.load(f)
            data['models'] = clustering_results.get('models', [])
            data['selected_k'] = clustering_results.get('selected_k', 2)
            data['plots'] = clustering_results.get('plots', {})
            data['optimal_k_suggestions'] = clustering_results.get('optimal_k_suggestions', {})
            logging.debug(f"Đã đọc clustering_results.json: {data['models']}, selected_k={data['selected_k']}")
        except Exception as e:
            logging.error(f"Lỗi đọc clustering_results.json: {str(e)}")
            flash("Lỗi khi đọc kết quả phân cụm trước đó. Vui lòng chạy lại mô hình.")
    
    if request.method == 'POST':
        logging.debug("Nhận yêu cầu POST")
        models = request.form.getlist('models')
        selected_k = int(request.form.get('k'))
        
        # Kiểm tra selected_k
        if selected_k < 2:
            logging.error(f"selected_k={selected_k} nhỏ hơn 2")
            flash("Số cụm tối đa (k) phải lớn hơn hoặc bằng 2 để áp dụng phương pháp khuỷu tay.")
            return render_template('select_model.html', data=data)
        logging.debug(f"Selected_k: {selected_k}, Models: {models}")
        
        if not models:
            logging.error("Không có mô hình nào được chọn")
            flash("Vui lòng chọn ít nhất một mô hình.")
            return render_template('select_model.html', data=data)
        
        # Kiểm tra dữ liệu trước khi chạy phân cụm
        X_numeric = X.select_dtypes(include=[np.number])
        logging.debug(f"Kích thước X_numeric: {X_numeric.shape}")
        if X_numeric.empty or len(X_numeric.columns) < 2:
            logging.error("Dữ liệu không đủ cột số")
            flash("Dữ liệu không chứa đủ cột số (cần ít nhất 2 cột số). Vui lòng kiểm tra và xử lý lại dữ liệu.")
            return render_template('select_model.html', data=data)
        
        if X_numeric.isna().any().any():
            logging.error("Dữ liệu chứa giá trị NaN")
            flash("Dữ liệu chứa giá trị NaN. Vui lòng xử lý dữ liệu trước khi chạy phân cụm.")
            return render_template('select_model.html', data=data)
        
        if np.isinf(X_numeric.values).any():
            logging.error("Dữ liệu chứa giá trị vô cực")
            flash("Dữ liệu chứa giá trị vô cực (inf). Vui lòng xử lý dữ liệu trước khi chạy phân cụm.")
            return render_template('select_model.html', data=data)
        
        # Kiểm tra số chiều
        if X_numeric.shape[1] > 20:
            logging.warning(f"Dữ liệu có {X_numeric.shape[1]} chiều, có thể làm chậm phân cụm")
            flash(f"Dữ liệu có {X_numeric.shape[1]} chiều, có thể làm chậm phân cụm. Hãy giảm số chiều bằng PCA hoặc chọn ít feature hơn.")
        
        # Reset models để chỉ lưu các mô hình thành công
        data['models'] = []
        data['selected_k'] = selected_k
        
        # Kiểm tra kết quả đã lưu trước khi chạy phân cụm mới
        for model in models:
            cache_key = f"{model}_{selected_k}"
            if cache_key in data['plots']:
                logging.debug(f"Sử dụng kết quả đã lưu cho {cache_key}")
                plots = data['plots'][cache_key]
                if 'error' in plots:
                    flash(plots['error'])
                    continue
                data['plots'][model] = plots
                optimal_k, reasoning = suggest_optimal_k(plots, list(range(2, selected_k + 1)))
                data['optimal_k_suggestions'][model] = {'k': optimal_k, 'reasoning': reasoning}
                if 'cvi' in plots:
                    data['models'].append(model)
                continue
        
        # Chạy phân cụm song song cho các mô hình chưa có kết quả
        logging.debug(f"Chạy phân cụm song song cho các mô hình: {[m for m in models if f'{m}_{selected_k}' not in data['plots']]}")
        results = Parallel(n_jobs=-1)(
            delayed(run_clustering)(
                model, X, range(2, selected_k + 1), selected_k, data['use_pca'],
                data['selected_features'], data['explained_variance_ratio']
            )
            for model in models if f"{model}_{selected_k}" not in data['plots']
        )
        
        # Xử lý kết quả
        for model, plots in results:
            logging.debug(f"Kết quả phân cụm cho {model}: {'error' in plots}")
            if 'error' in plots:
                flash(plots['error'])
                continue
            data['plots'][model] = plots
            optimal_k, reasoning = suggest_optimal_k(plots, list(range(2, selected_k + 1)))
            data['optimal_k_suggestions'][model] = {'k': optimal_k, 'reasoning': reasoning}
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
            logging.debug("Lưu kết quả phân cụm vào clustering_results.json")
            with open(clustering_results_file, 'w') as f:
                json.dump(clustering_results, f)
        else:
            logging.warning("Không có mô hình nào chạy thành công")
        
        logging.debug("Hoàn thành xử lý POST, trả về giao diện")
        return render_template('select_model.html', data=data)
    
    logging.debug("Trả về giao diện mặc định (GET)")
    return render_template('select_model.html', data=data)