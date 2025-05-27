from flask import render_template, request, redirect, url_for, flash, current_app, session
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering import generate_clustering_plots
from utils.metrics import suggest_optimal_k
import time

def select_model_dashkit():
    """Chọn mô hình phân cụm - phiên bản Dashkit"""
    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            selected_models = request.form.getlist('models')
            max_k = int(request.form.get('k', 10))
            
            if not selected_models:
                flash('Vui lòng chọn ít nhất một mô hình.')
                return redirect(url_for('select_model'))
            
            # Kiểm tra dữ liệu PCA
            pca_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_data.pkl')
            if not os.path.exists(pca_data_file):
                flash('Vui lòng thực hiện PCA trước khi chọn mô hình.')
                return redirect(url_for('process_data_dashkit'))
            
            # Load dữ liệu PCA
            df_pca = pd.read_pickle(pca_data_file)
            
            logging.debug(f"Loaded PCA data shape: {df_pca.shape}")
            logging.debug(f"Selected models: {selected_models}, max_k: {max_k}")
            
            # Khởi tạo kết quả
            results = {
                'models': [],
                'k_range': list(range(2, max_k + 1)),
                'selected_k': max_k,
                'cvi_scores': {},
                'plots': {},
                'optimal_k_suggestions': {}
            }
            
            start_time = time.time()
              # Tối ưu hóa: Preprocessing dữ liệu một lần cho tất cả models
            X_array = df_pca.values.astype(float)
            
            # Chạy các mô hình được chọn
            for model_name in selected_models:
                try:
                    logging.debug(f"Processing model: {model_name}")
                    
                    # Tối ưu hóa: Sử dụng tham số được tối ưu cho tốc độ
                    plots = generate_clustering_plots(
                        X=df_pca,
                        model_name=model_name,
                        k_range=results['k_range'],
                        selected_k=max_k,
                        use_pca=True,
                        selected_features=list(df_pca.columns),
                        explained_variance_ratio=0.95
                    )
                    
                    if 'error' in plots:
                        flash(f"Lỗi khi chạy mô hình {model_name}: {plots['error']}")
                        continue
                    
                    if plots and 'cvi' in plots:
                        results['models'].append(model_name)
                        
                        # Chuyển đổi cvi_scores về format mong muốn
                        cvi_dict = {}
                        for cvi_entry in plots['cvi']:
                            k = str(cvi_entry['k'])
                            cvi_dict[k] = {
                                'silhouette': cvi_entry['Silhouette'],
                                'calinski_harabasz': cvi_entry['Calinski-Harabasz'],
                                'davies_bouldin': cvi_entry['Davies-Bouldin'],
                                'starczewski': cvi_entry['Starczewski'],
                                'wiroonsri': cvi_entry['Wiroonsri']
                            }
                        
                        results['cvi_scores'][model_name] = cvi_dict
                        results['plots'][model_name] = plots
                        
                        # Gợi ý k tối ưu
                        try:
                            optimal_k, reasoning = suggest_optimal_k(
                                plots=plots,
                                k_range=results['k_range'],
                                use_wiroonsri_starczewski=False
                            )
                            results['optimal_k_suggestions'][model_name] = {
                                'k': optimal_k,
                                'reasoning': reasoning
                            }
                        except Exception as e:
                            logging.error(f"Error suggesting optimal k for {model_name}: {str(e)}")
                            results['optimal_k_suggestions'][model_name] = {
                                'k': 3,
                                'reasoning': 'Không thể tính toán gợi ý tự động'
                            }
                    else:
                        flash(f"Không thể chạy mô hình {model_name}. Vui lòng thử lại.")
                        
                except Exception as e:
                    logging.error(f"Error processing model {model_name}: {str(e)}")
                    flash(f"Lỗi khi chạy mô hình {model_name}: {str(e)}")
            
            processing_time = time.time() - start_time
            logging.debug(f"Total processing time: {processing_time:.2f} seconds")
            
            if results['models']:
                # Lưu kết quả
                results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'model_results.json')
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                flash(f'Đã chạy thành công {len(results["models"])} mô hình với k từ 2 đến {max_k}.')
            else:
                flash('Không có mô hình nào chạy thành công. Vui lòng kiểm tra dữ liệu.')
            
            return redirect(url_for('select_model'))
            
        except Exception as e:
            logging.error(f"Error in select_model_dashkit POST: {str(e)}")
            flash(f'Lỗi khi chạy mô hình: {str(e)}')
            return redirect(url_for('select_model'))
    
    # GET request
    try:
        # Load kết quả nếu có
        results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'model_results.json')
        data = {
            'k_range': list(range(2, 11)),
            'selected_k': 10,
            'models': [],
            'cvi_scores': {},
            'plots': {},
            'optimal_k_suggestions': {}
        }
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    data.update(results)
                logging.debug(f"Loaded previous results: {len(data.get('models', []))} models")
            except Exception as e:
                logging.error(f"Error loading results file: {str(e)}")
        
        return render_template('select_model_dashkit.html', data=data)
        
    except Exception as e:
        logging.error(f"Error in select_model_dashkit GET: {str(e)}")
        flash(f'Lỗi khi tải trang: {str(e)}')
        return redirect(url_for('dashkit_index'))