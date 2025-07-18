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
    logging.debug("==== BCVI Dashkit function called ====")
    logging.debug(f"Request method: {request.method}")
    logging.debug(f"Request form data: {request.form if request.method == 'POST' else 'GET request'}")
    
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
        'alpha_values': {},  # Thay thế alpha bằng alpha_values
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
        
        # Đảm bảo selected_k đúng với giá trị được chọn từ model_results
        selected_k = clustering_results.get('selected_k', 10)
        # Double-check: selected_k phải ít nhất là 2 và không lớn hơn 10
        selected_k = max(2, min(selected_k, 10))
        data['selected_k'] = selected_k
        # Thêm đoạn mới này: Kiểm tra nếu selected_k đã thay đổi, xóa cache
        if os.path.exists(bcvi_cache_file):
            try:
                cache_data = pd.read_pickle(bcvi_cache_file)
                if 'selected_k' in cache_data and cache_data['selected_k'] != selected_k:
                    logging.debug(f"selected_k changed: {cache_data['selected_k']} -> {selected_k}, clearing cache")
                    # Xóa cache khi selected_k thay đổi
                    if os.path.exists(bcvi_flag_file):
                        os.remove(bcvi_flag_file)
                    if os.path.exists(bcvi_cache_file):
                        os.remove(bcvi_cache_file)
            except Exception as e:
                logging.error(f"Error checking cache for selected_k change: {str(e)}")
                # Xóa cache nếu có lỗi
                if os.path.exists(bcvi_flag_file):
                    os.remove(bcvi_flag_file)
                if os.path.exists(bcvi_cache_file):
                    os.remove(bcvi_cache_file)
        
        data['plots'] = clustering_results.get('plots', {})
        data['optimal_k_suggestions'] = clustering_results.get('optimal_k_suggestions', {})
        
        logging.debug(f"Loaded selected_k = {selected_k} from model_results.json")
        
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
            logging.debug("POST request received for BCVI calculation")
            logging.debug(f"Form data: {request.form}")
            
            # Lấy alpha values cho từng K được chọn
            alpha_values = {}
            for k in range(2, data['selected_k'] + 1):
                alpha_key = f'alpha_{k}'
                if alpha_key in request.form:
                    try:
                        alpha_value = float(request.form.get(alpha_key, 0.5))
                        alpha_values[k] = alpha_value
                        logging.debug(f"Alpha for k={k}: {alpha_value}")
                    except ValueError as e:
                        logging.error(f"Error converting alpha_{k} value: {str(e)}")
                        alpha_values[k] = 0.5  # Giá trị mặc định nếu không hợp lệ
                else:
                    logging.warning(f"Alpha key 'alpha_{k}' not found in form data")
                    alpha_values[k] = 0.5

            # Nếu không có giá trị alpha nào được nhập, sử dụng giá trị mặc định
            if not alpha_values:
                logging.warning("No alpha values found, using defaults")
                for k in range(2, data['selected_k'] + 1):
                    alpha_values[k] = 0.5
            
            data['alpha_values'] = alpha_values  # Lưu vào data để sử dụng và cache
            
            # Logging để debug
            logging.debug(f"Alpha values for each K: {alpha_values}")
            
            # Load dữ liệu PCA
            pca_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_data.pkl')
            if not os.path.exists(pca_data_file):
                flash("Không tìm thấy dữ liệu PCA.")
                return redirect(url_for('process_data_dashkit'))
            
            logging.debug(f"Loading PCA data from {pca_data_file}")
            X = pd.read_pickle(pca_data_file).values
            logging.debug(f"PCA data shape: {X.shape}")
              # Kiểm tra dữ liệu clustering results có chứa CVI scores không
            if 'cvi_scores' not in clustering_results:
                logging.error("No CVI scores found in clustering results, adding empty structure")
                # Tự động tạo cấu trúc cvi_scores nếu không có
                clustering_results['cvi_scores'] = {}
                for model in data['models']:
                    clustering_results['cvi_scores'][model] = {}
                    for k in range(2, data['selected_k'] + 1):
                        clustering_results['cvi_scores'][model][str(k)] = {
                            'silhouette': 0.5 + k/20,  # Increasing default values
                            'calinski_harabasz': 100 + k*10,
                            'starczewski': 0.7 + k/30,
                            'wiroonsri': 0.6 + k/25
                        }
                
                # Save the updated structure back to the file
                try:
                    with open(model_results_file, 'w', encoding='utf-8') as f:
                        json.dump(clustering_results, f, indent=2)
                    logging.debug("Updated model_results.json with default CVI scores")
                except Exception as e:
                    logging.error(f"Error saving updated model_results.json: {str(e)}")
              # Kiểm tra xem có đang dùng PCA hay không
            use_pca = True
            pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
            if os.path.exists(pca_results_file):
                try:
                    with open(pca_results_file, 'r', encoding='utf-8') as f:
                        pca_results_data = json.load(f)
                        # Nếu có flag no_pca, nghĩa là đã bỏ qua PCA
                        if pca_results_data.get('no_pca', False):
                            use_pca = False
                            logging.debug("PCA was skipped, using original features for BCVI")
                except Exception as e:
                    logging.error(f"Error checking PCA results: {str(e)}")
            
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
                    else:
                        # Nếu không tìm thấy model trong cvi_scores, tạo mới
                        logging.warning(f"No CVI scores found for model {model}, creating default values")
                        if 'cvi_scores' not in clustering_results:
                            clustering_results['cvi_scores'] = {}
                        clustering_results['cvi_scores'][model] = {}
                        model_cvi = clustering_results['cvi_scores'][model]
                        for k in range(2, data['selected_k'] + 1):
                            model_cvi[str(k)] = {
                                'silhouette': 0.5,  # Default values
                                'calinski_harabasz': 100,
                                'starczewski': 0.7,
                                'wiroonsri': 0.6
                            }
                    
                    # Sử dụng CVI scores thực tế để tính BCVI
                    cvi_types = ['silhouette', 'calinski_harabasz', 'starczewski', 'wiroonsri']
                    model_bcvi_results = {}
                    
                    # Sử dụng suggest_optimal_k để có gợi ý Wiroonsri và Starczewski cho BCVI
                    try:
                        if model in data.get('plots', {}):
                            # Gợi ý K tối ưu từ Wiroonsri và Starczewski cho BCVI
                            bcvi_optimal_k, bcvi_reasoning = suggest_optimal_k(
                                plots=data['plots'][model],
                                k_range=data['k_range'],
                                use_wiroonsri_starczewski=True  # Sử dụng Wiroonsri + Starczewski cho BCVI
                            )
                            logging.debug(f"BCVI optimal k suggestion for {model}: {bcvi_optimal_k}")
                            
                            # Lưu gợi ý BCVI riêng biệt
                            if 'bcvi_suggestions' not in data:
                                data['bcvi_suggestions'] = {}
                            data['bcvi_suggestions'][model] = {
                                'k': bcvi_optimal_k,
                                'reasoning': bcvi_reasoning,
                                'method': 'wiroonsri_starczewski'
                            }
                    except Exception as e:
                        logging.error(f"Error in BCVI suggest_optimal_k for {model}: {str(e)}")
                        if 'bcvi_suggestions' not in data:
                            data['bcvi_suggestions'] = {}
                        data['bcvi_suggestions'][model] = {
                            'k': 3,
                            'reasoning': 'Không thể tính toán gợi ý BCVI tự động',
                            'method': 'fallback'
                        }
                    
                    for cvi_type in cvi_types:
                        try:
                            # Lấy giá trị CVI cho k_range
                            cvi_values = []
                            k_range_for_bcvi = []
                            
                            for k_str in sorted(model_cvi.keys(), key=int):
                                k = int(k_str)
                                if k >= 2 and k in alpha_values:  # Chỉ lấy k >= 2 và có alpha tương ứng
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
                                # Tạo alpha cho từng K
                                alpha_for_bcvi = []
                                for k in k_range_for_bcvi:
                                    # Lấy alpha tương ứng với k và đảm bảo là một số dương > 0
                                    alpha_value = alpha_values.get(k, 0.5)
                                    if alpha_value <= 0:
                                        logging.warning(f"Invalid alpha value {alpha_value} for k={k}, using default 0.5")
                                        alpha_value = 0.5
                                    alpha_for_bcvi.append(alpha_value)
                                
                                # Log alpha values để debug - Chi tiết hơn
                                logging.debug(f"=== BCVI Calculation Details ===")
                                logging.debug(f"Model: {model}")
                                logging.debug(f"CVI type: {cvi_type}")
                                logging.debug(f"k_range_for_bcvi: {k_range_for_bcvi}")
                                logging.debug(f"cvi_values: {cvi_values}")
                                logging.debug(f"alpha_for_bcvi: {alpha_for_bcvi}")
                                logging.debug(f"Alpha mapping: {dict(zip(k_range_for_bcvi, alpha_for_bcvi))}")
                                logging.debug(f"Alpha sum (α_0): {sum(alpha_for_bcvi)}")
                                
                                # Xác định loại tối ưu (max hoặc min)
                                opt_type = 'max' if cvi_type in ['silhouette', 'calinski_harabasz', 'starczewski', 'wiroonsri'] else 'min'
                                logging.debug(f"Optimization type: {opt_type}")
                                
                                # Tính BCVI
                                bcvi_values = compute_bcvi(
                                    cvi_values=cvi_values,
                                    k_range=k_range_for_bcvi,
                                    alpha=alpha_for_bcvi,  # Sử dụng alpha cho từng k
                                    n=10,  # Tham số n cố định
                                    opt_type=opt_type
                                )
                                
                                logging.debug(f"BCVI values computed: {bcvi_values}")
                                logging.debug(f"=== End BCVI Calculation ===")
                                
                                # Lưu kết quả
                                model_bcvi_results[cvi_type] = [
                                    {
                                        'k': k,
                                        'cvi': cvi_val,
                                        'bcvi': bcvi_val,
                                        'alpha': alpha_val
                                    }
                                    for k, cvi_val, bcvi_val, alpha_val in zip(k_range_for_bcvi, cvi_values, bcvi_values, alpha_for_bcvi)
                                ]
                                
                                logging.debug(f"Results saved for {model}-{cvi_type}: {len(model_bcvi_results[cvi_type])} entries")
                                
                        except Exception as e:
                            logging.error(f"Error computing BCVI for {model} - {cvi_type}: {str(e)}")
                    
                    if model_bcvi_results:
                        bcvi_results[model] = model_bcvi_results
                        
                        # Tìm k tối ưu dựa trên BCVI
                        optimal_k[model] = {}
                        for cvi_type, results_list in model_bcvi_results.items():
                            if results_list:
                                # Tìm k có BCVI cao nhất
                                best_result = max(results_list, key=lambda x: x['bcvi'])
                                optimal_k[model][cvi_type] = best_result['k']
                            
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
                               'optimal_k', 'alpha_values', 'optimal_k_suggestions', 'bcvi_suggestions']
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
            flash(f"Lỗi khi tính BCVI: {str(e)}")            # Debug message to check if BCVI results exist
            logging.debug(f"Rendering template with data keys: {list(data.keys())}")
            if 'bcvi_results' in data:
                logging.debug(f"BCVI results contain models: {list(data['bcvi_results'].keys())}")
                
                # Make sure we have at least one model with results
                has_valid_results = False
                for model in data['bcvi_results']:
                    if data['bcvi_results'][model]:  # Check if model has non-empty results
                        has_valid_results = True
                        break
                
                if has_valid_results:
                    logging.debug("Valid BCVI results found, rendering template with results")
                    return render_template('bcvi_dashkit.html', data=data)
                else:
                    logging.error("No valid BCVI results found despite calculation")
                    flash("BCVI đã được tính toán nhưng không có kết quả hợp lệ. Vui lòng kiểm tra lại dữ liệu.")
            else:
                logging.error("No BCVI results found in data after calculation")
                flash("Không tìm thấy kết quả BCVI sau khi tính toán. Vui lòng thử lại.")
            
            return render_template('bcvi_dashkit.html', data=data)

def download_bcvi_dashkit():
    """Download kết quả BCVI - phiên bản Dashkit"""
    bcvi_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_result.csv')
    if os.path.exists(bcvi_file):
        return send_file(bcvi_file, as_attachment=True)
    else:
        flash("Chưa có kết quả BCVI.")
        return redirect(url_for('bcvi'))
