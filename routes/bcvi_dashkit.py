from flask import render_template, request, redirect, url_for, flash, send_file, current_app, session, jsonify
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering import compute_bcvi
from utils.metrics import suggest_optimal_k
import time

def bcvi_dashkit():
    """Hiển thị trang BCVI - phiên bản Dashkit"""
    
    # Kiểm tra cache trước
    bcvi_cache_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_cache.pkl')
    bcvi_flag_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_calculated.flag')
    
    if os.path.exists(bcvi_flag_file) and request.method != 'POST':
        try:
            logging.debug("Loading BCVI results from cache")
            data = pd.read_pickle(bcvi_cache_file)
            
            if 'bcvi_results' in data and data['bcvi_results']:
                return render_template('bcvi_dashkit.html', data=data)
            else:
                # Cache không hợp lệ, xóa
                if os.path.exists(bcvi_flag_file):
                    os.remove(bcvi_flag_file)
                if os.path.exists(bcvi_cache_file):
                    os.remove(bcvi_cache_file)
        except Exception as e:
            logging.error(f"Error loading BCVI cache: {str(e)}")
            if os.path.exists(bcvi_flag_file):
                os.remove(bcvi_flag_file)
            if os.path.exists(bcvi_cache_file):
                os.remove(bcvi_cache_file)
    
    # Khởi tạo data
    data = {
        'k_range': list(range(2, 11)),
        'selected_k': 2,
        'models': [],
        'plots': {},
        'bcvi_results': {},
        'optimal_k': {},
        'alpha': [],
        'cluster_stats': {},
        'cluster_sizes': {}
    }
    
    # Kiểm tra kết quả phân cụm
    model_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'model_results.json')
    if not os.path.exists(model_results_file):
        flash("Vui lòng chạy phân cụm trước.")
        return redirect(url_for('select_model'))
    
    try:
        with open(model_results_file, 'r', encoding='utf-8') as f:
            clustering_results = json.load(f)
        
        data['models'] = clustering_results.get('models', [])
        data['selected_k'] = clustering_results.get('selected_k', 2)
        data['plots'] = clustering_results.get('plots', {})
        data['optimal_k_suggestions'] = clustering_results.get('optimal_k_suggestions', {})
        
    except Exception as e:
        logging.error(f"Error reading model results: {str(e)}")
        flash("Lỗi khi đọc kết quả phân cụm.")
        return redirect(url_for('select_model'))
    
    if not data['models']:
        flash("Không có mô hình nào được chạy thành công.")
        return redirect(url_for('select_model'))
    
    # Xử lý POST request (tính BCVI)
    if request.method == 'POST':
        try:
            # Lấy alpha values
            alpha_str = request.form.get('alpha', '0.1,0.5,1.0')
            alpha = [float(x.strip()) for x in alpha_str.split(',') if x.strip()]
            
            if not alpha:
                alpha = [0.1, 0.5, 1.0]
            
            data['alpha'] = alpha
            
            # Load dữ liệu PCA
            pca_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_data.pkl')
            if not os.path.exists(pca_data_file):
                flash("Không tìm thấy dữ liệu PCA.")
                return redirect(url_for('process_data_dashkit'))
            
            X = pd.read_pickle(pca_data_file).values
              # Tính BCVI sử dụng CVI scores thực tế
            bcvi_results = {}
            optimal_k = {}
            
            start_time = time.time()
            for model in data['models']:
                try:
                    logging.debug(f"Computing BCVI for model: {model}")
                    
                    # Lấy CVI scores thực tế từ clustering results
                    if model in clustering_results.get('cvi_scores', {}):
                        model_cvi = clustering_results['cvi_scores'][model]
                        
                        # Sử dụng CVI scores thực tế để tính BCVI
                        cvi_types = ['silhouette', 'calinski_harabasz', 'starczewski', 'wiroonsri']
                        model_bcvi_results = {}
                        
                        # Sử dụng suggest_optimal_k để có gợi ý ban đầu
                        try:
                            if model in data.get('plots', {}):
                                # Gợi ý K tối ưu từ traditional methods trước khi tính BCVI
                                traditional_k, reasoning = suggest_optimal_k(
                                    plots=data['plots'][model],
                                    k_range=data['k_range'],
                                    use_wiroonsri_starczewski=False
                                )
                                logging.debug(f"Traditional optimal k for {model}: {traditional_k}")
                        except Exception as e:
                            logging.error(f"Error in suggest_optimal_k for {model}: {str(e)}")
                            traditional_k = 3  # Default fallback
                        
                        for cvi_type in cvi_types:
                            try:
                                # Lấy giá trị CVI cho k_range
                                cvi_values = []
                                k_range_for_bcvi = []
                                
                                for k_str in sorted(model_cvi.keys(), key=int):
                                    k = int(k_str)
                                    if k >= 2:  # Chỉ lấy k >= 2
                                        if cvi_type == 'silhouette' and model_cvi[k_str].get('silhouette', 0) != 0:
                                            cvi_values.append(model_cvi[k_str]['silhouette'])
                                            k_range_for_bcvi.append(k)
                                        elif cvi_type == 'calinski_harabasz' and model_cvi[k_str].get('calinski_harabasz', 0) != 0:
                                            cvi_values.append(model_cvi[k_str]['calinski_harabasz'])
                                            k_range_for_bcvi.append(k)
                                        elif cvi_type == 'starczewski' and model_cvi[k_str].get('starczewski', 0) != 0:
                                            cvi_values.append(model_cvi[k_str]['starczewski'])
                                            k_range_for_bcvi.append(k)
                                        elif cvi_type == 'wiroonsri' and model_cvi[k_str].get('wiroonsri', 0) != 0:
                                            cvi_values.append(model_cvi[k_str]['wiroonsri'])
                                            k_range_for_bcvi.append(k)
                                
                                if len(cvi_values) >= 2:  # Cần ít nhất 2 giá trị để tính BCVI
                                    # Tạo alpha vector với cùng độ dài
                                    alpha_vector = alpha * len(k_range_for_bcvi)
                                    if len(alpha_vector) != len(k_range_for_bcvi):
                                        alpha_vector = [1.0] * len(k_range_for_bcvi)
                                    
                                    # Xác định loại tối ưu (max hoặc min)
                                    opt_type = 'max' if cvi_type in ['silhouette', 'calinski_harabasz', 'starczewski', 'wiroonsri'] else 'min'
                                    
                                    # Tính BCVI
                                    bcvi_values = compute_bcvi(
                                        cvi_values=cvi_values,
                                        k_range=k_range_for_bcvi,
                                        alpha=alpha_vector,
                                        n=10,  # Tham số n cố định
                                        opt_type=opt_type
                                    )
                                    
                                    # Lưu kết quả
                                    model_bcvi_results[cvi_type] = [
                                        {
                                            'k': k,
                                            'cvi': cvi_val,
                                            'bcvi': bcvi_val,
                                            'alpha': alpha_val
                                        }
                                        for k, cvi_val, bcvi_val, alpha_val in zip(k_range_for_bcvi, cvi_values, bcvi_values, alpha_vector)
                                    ]
                                    
                            except Exception as e:
                                logging.error(f"Error computing BCVI for {model} - {cvi_type}: {str(e)}")
                        if model_bcvi_results:
                            bcvi_results[model] = model_bcvi_results
                            
                            # Tìm k tối ưu dựa trên BCVI - Cải thiện thuật toán
                            optimal_k[model] = {}
                            
                            # 1. Tìm K tối ưu cho từng loại CVI
                            for cvi_type, results_list in model_bcvi_results.items():
                                if results_list:
                                    # Tìm k có BCVI cao nhất cho CVI type này
                                    best_result = max(results_list, key=lambda x: x['bcvi'])
                                    optimal_k[model][f'{cvi_type}_best_k'] = best_result['k']
                                    optimal_k[model][f'{cvi_type}_best_bcvi'] = best_result['bcvi']
                            
                            # 2. Tìm K tối ưu nhất TỔNG THỂ dựa trên average BCVI
                            k_averages = {}
                            for cvi_type, results_list in model_bcvi_results.items():
                                for result in results_list:
                                    k = result['k']
                                    if k not in k_averages:
                                        k_averages[k] = []
                                    k_averages[k].append(result['bcvi'])
                            
                            # Tính average BCVI cho mỗi k
                            k_avg_bcvi = {}
                            for k, bcvi_list in k_averages.items():
                                k_avg_bcvi[k] = np.mean(bcvi_list)
                            
                            # Tìm K có average BCVI cao nhất
                            if k_avg_bcvi:
                                best_overall_k = max(k_avg_bcvi.keys(), key=lambda k: k_avg_bcvi[k])
                                optimal_k[model]['overall_best_k'] = best_overall_k
                                optimal_k[model]['overall_best_bcvi'] = k_avg_bcvi[best_overall_k]
                                
                                logging.debug(f"Model {model}: Overall best K = {best_overall_k} with avg BCVI = {k_avg_bcvi[best_overall_k]:.4f}")
                            
                            # 3. So sánh với traditional optimal k
                            optimal_k[model]['traditional_k'] = traditional_k
                            
                except Exception as e:
                    logging.error(f"Error computing BCVI for {model}: {str(e)}")
                    flash(f"Lỗi khi tính BCVI cho mô hình {model}: {str(e)}")
            
            processing_time = time.time() - start_time
            logging.debug(f"BCVI computation time: {processing_time:.2f} seconds")
            
            data['bcvi_results'] = bcvi_results
            data['optimal_k'] = optimal_k
            
            # Lưu cache
            try:
                cache_data = data.copy()
                essential_keys = ['k_range', 'selected_k', 'models', 'plots', 'bcvi_results', 
                                'optimal_k', 'alpha', 'optimal_k_suggestions']
                cache_data = {key: data[key] for key in essential_keys if key in data}
                
                pd.to_pickle(cache_data, bcvi_cache_file, compression='gzip')
                
                with open(bcvi_flag_file, 'w') as f:
                    f.write('1')
                
                logging.debug("BCVI cache saved successfully")
                
            except Exception as e:
                logging.error(f"Error saving BCVI cache: {str(e)}")
              # Lưu kết quả CSV - Tối ưu hóa format
            try:
                result_data = []
                for model in data['models']:
                    if model in bcvi_results:
                        for cvi_type, results_list in bcvi_results[model].items():
                            for result in results_list:
                                result_data.append({
                                    'Model': model,
                                    'CVI_Type': cvi_type.upper(),
                                    'K': result['k'],
                                    'CVI_Value': result['cvi'],
                                    'BCVI': result['bcvi'],
                                    'Alpha': result['alpha']
                                })
                
                if result_data:
                    result_df = pd.DataFrame(result_data)
                    bcvi_csv_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_result.csv')
                    result_df.to_csv(bcvi_csv_file, index=False)
                    logging.debug(f"BCVI results saved to {bcvi_csv_file}")
                
            except Exception as e:
                logging.error(f"Error saving BCVI CSV: {str(e)}")
            
            flash(f"BCVI đã được tính toán thành công cho {len(bcvi_results)} mô hình.")
            
        except Exception as e:
            logging.error(f"Error in BCVI computation: {str(e)}")
            flash(f"Lỗi khi tính BCVI: {str(e)}")
    
    return render_template('bcvi_dashkit.html', data=data)

def download_bcvi_dashkit():
    """Download kết quả BCVI - phiên bản Dashkit"""
    bcvi_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_result.csv')
    if os.path.exists(bcvi_file):
        return send_file(bcvi_file, as_attachment=True)
    else:
        flash("Chưa có kết quả BCVI.")
        return redirect(url_for('bcvi'))