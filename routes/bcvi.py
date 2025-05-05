from flask import render_template, request, redirect, url_for, flash, send_file, current_app
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering import compute_bcvi

def bcvi():
    data = {
        'k_range': list(range(2, 11)),
        'selected_k': 2,
        'models': [],
        'plots': {},
        'bcvi_results': {},
        'optimal_k': {},
        'alpha': []
    }
    
    clustering_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'clustering_results.json')
    if not os.path.exists(clustering_results_file):
        flash("Vui lòng chạy phân cụm trước.")
        return redirect(url_for('select_model'))
    
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
        return redirect(url_for('select_model'))
    
    if not data['models']:
        flash("Không có mô hình nào được chạy thành công. Vui lòng chạy lại phân cụm.")
        return redirect(url_for('select_model'))
    
    # Cập nhật k_range dựa trên selected_k
    data['k_range'] = list(range(2, data['selected_k'] + 1))
    
    if request.method == 'POST':
        # Lấy tham số alpha từ form
        alpha = []
        for k in data['k_range']:
            alpha_k = request.form.get(f'alpha_{k}')
            try:
                alpha_k = float(alpha_k)
                if alpha_k < 0:
                    raise ValueError
                alpha.append(alpha_k)
            except (ValueError, TypeError):
                flash(f"Tham số alpha_{k} không hợp lệ. Vui lòng nhập số không âm.")
                return render_template('bcvi.html', data=data)
        
        # Số lượng mẫu n (cố định)
        n = 100
        
        # Tính toán BCVI cho các chỉ số CVI
        try:
            bcvi_results = {}
            optimal_k = {}
            
            for model in data['models']:
                cvi_indices = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Starczewski', 'Wiroonsri']
                model_results = []
                
                # Kiểm tra và làm sạch dữ liệu CVI
                cleaned_cvi_data = []
                for cvi in data['plots'][model]['cvi']:
                    cleaned_cvi = {}
                    for key, value in cvi.items():
                        if key == 'k':
                            cleaned_cvi[key] = value
                        else:
                            try:
                                cleaned_cvi[key] = float(value) if value is not None else 0
                            except (ValueError, TypeError):
                                logging.warning(f"Invalid CVI value for {key} in model {model}: {value}. Setting to 0.")
                                cleaned_cvi[key] = 0
                    cleaned_cvi_data.append(cleaned_cvi)
                
                # Lấy danh sách các giá trị k thực sự có trong cleaned_cvi_data
                actual_k_values = [cvi['k'] for cvi in cleaned_cvi_data]
                logging.debug(f"Model: {model}, Actual k values in CVI data: {actual_k_values}")
                
                # Chỉ giữ các alpha tương ứng với các giá trị k có trong cleaned_cvi_data
                filtered_alpha = []
                filtered_k_range = []
                alpha_index = 0
                for k in data['k_range']:
                    if k in actual_k_values:
                        filtered_k_range.append(k)
                        filtered_alpha.append(alpha[alpha_index])
                    alpha_index += 1
                logging.debug(f"Model: {model}, Filtered k range: {filtered_k_range}, Filtered alpha: {filtered_alpha}")
                
                if not filtered_k_range:
                    flash(f"Không có dữ liệu CVI cho mô hình {model} trong khoảng k đã chọn.")
                    continue
                
                for k in filtered_k_range:
                    cvi_entry = next((cvi for cvi in cleaned_cvi_data if cvi['k'] == k), None)
                    if not cvi_entry:
                        continue
                    
                    bcvi_entry = {'k': k, 'bcvi': {}}
                    for cvi_index in cvi_indices:
                        cvi_values = [cvi.get(cvi_index, 0) for cvi in cleaned_cvi_data if cvi['k'] in filtered_k_range]
                        logging.debug(f"Model: {model}, CVI Index: {cvi_index}, CVI Values: {cvi_values}")
                        
                        # Kiểm tra xem cvi_values có hợp lệ để tính BCVI không
                        if all(v == 0 for v in cvi_values):
                            logging.warning(f"All CVI values for {cvi_index} in model {model} are 0. Skipping BCVI calculation for this index.")
                            bcvi_entry['bcvi'][cvi_index] = 0.0
                            continue
                        
                        # Xác định opt_type dựa trên cvi_index
                        if cvi_index in ['Silhouette', 'Calinski-Harabasz', 'Starczewski', 'Wiroonsri']:
                            opt_type = 'max'
                        else:
                            opt_type = 'min'
                        
                        bcvi_values = compute_bcvi(cvi_values, filtered_k_range, filtered_alpha, n, opt_type=opt_type)
                        k_index = filtered_k_range.index(k)
                        bcvi_entry['bcvi'][cvi_index] = bcvi_values[k_index]
                    
                    model_results.append(bcvi_entry)
                
                if not model_results:
                    flash(f"Không có kết quả BCVI cho mô hình {model}.")
                    continue
                
                # Tìm số k tối ưu dựa trên Silhouette BCVI
                silhouette_bcvi = [(result['k'], result['bcvi']['Silhouette']) for result in model_results if result['bcvi']['Silhouette'] != 0.0]
                optimal_k[model] = max(silhouette_bcvi, key=lambda x: x[1])[0] if silhouette_bcvi else filtered_k_range[0]
                
                bcvi_results[model] = model_results
            
            if not bcvi_results:
                flash("Không có kết quả BCVI nào được tạo. Vui lòng kiểm tra dữ liệu phân cụm.")
                return render_template('bcvi.html', data=data)
            
            data['bcvi_results'] = bcvi_results
            data['optimal_k'] = optimal_k
            data['alpha'] = alpha
            
            # Lưu kết quả BCVI vào file CSV
            result_data = []
            for model in data['models']:
                if model not in bcvi_results:
                    continue
                for result in bcvi_results[model]:
                    row = {'Model': model, 'k': result['k']}
                    for cvi_index, value in result['bcvi'].items():
                        row[f'{cvi_index} BCVI'] = value
                    result_data.append(row)
            result_df = pd.DataFrame(result_data)
            result_df.to_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_result.csv'), index=False)
            
        except Exception as e:
            flash(f"Lỗi khi tính toán BCVI: {str(e)}")
            return render_template('bcvi.html', data=data)
        
        return render_template('bcvi.html', data=data)
    
    return render_template('bcvi.html', data=data)

def download_bcvi():
    bcvi_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_result.csv')
    if os.path.exists(bcvi_file):
        return send_file(bcvi_file, as_attachment=True)
    else:
        flash("Chưa có kết quả BCVI.")
        return redirect(url_for('bcvi'))