from flask import render_template, request, redirect, url_for, flash, current_app, session
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering_optimized import generate_clustering_plots, run_multiple_models  # Sử dụng phiên bản tối ưu hóa
from utils.metrics import suggest_optimal_k, suggest_optimal_k_parallel
import time
import multiprocessing  # Thêm multiprocessing để xác định số lõi CPU

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
              # Kiểm tra dữ liệu PCA hoặc dữ liệu đã chọn
            pca_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_data.pkl')
            if not os.path.exists(pca_data_file):
                flash('Vui lòng xử lý dữ liệu trước khi chọn mô hình.')
                return redirect(url_for('process_data_dashkit'))
            
            # Load dữ liệu PCA
            df_pca = pd.read_pickle(pca_data_file)
            
            logging.debug(f"Loaded PCA data shape: {df_pca.shape}")
            logging.debug(f"Selected models: {selected_models}, max_k: {max_k}")
            
            # Kiểm tra xem có đang dùng PCA hay không
            use_pca = True
            pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
            if os.path.exists(pca_results_file):
                try:
                    with open(pca_results_file, 'r', encoding='utf-8') as f:
                        pca_results = json.load(f)
                        # Nếu có flag no_pca, nghĩa là đã bỏ qua PCA
                        if pca_results.get('no_pca', False):
                            use_pca = False
                            logging.debug("PCA was skipped, using original features")
                except Exception as e:
                    logging.error(f"Error checking PCA results: {str(e)}")
            
            # Khởi tạo kết quả
            results = {
                'models': [],
                'k_range': list(range(2, max_k + 1)),
                'selected_k': max_k,
                'cvi_scores': {},
                'plots': {},
                'optimal_k_suggestions': {}
            }
            
            start_time = time.time()            # Tối ưu hóa: Preprocessing dữ liệu một lần cho tất cả models
            X_array = df_pca.values.astype(float)
              # Xác định số lõi CPU và cách phân bổ tài nguyên
            cpu_count = multiprocessing.cpu_count()
            logging.debug(f"Số lõi CPU khả dụng: {cpu_count}")
              # CHIẾN LƯỢC TỐI ƯU: 
            # 1. Chạy song song 2 mô hình
            # 2. Mỗi mô hình dùng 5 luồng (tổng 10/12 luồng có sẵn)
            # 3. Để lại 2 luồng cho hệ thống

            # Cấu hình cố định cho máy 6 nhân 12 luồng
            TOTAL_THREADS = 8  # Tổng số luồng muốn sử dụng
            PARALLEL_MODELS = 2  # Số mô hình chạy song song
            THREADS_PER_MODEL = 4  # Số luồng cho mỗi mô hình

            # Số mô hình chạy song song (luôn là 2 hoặc ít hơn nếu chọn ít mô hình)
            parallel_models = min(len(selected_models), PARALLEL_MODELS)

            # Số luồng cho mỗi mô hình (cố định 5)
            threads_per_model = THREADS_PER_MODEL

            logging.debug(f"Cấu hình tài nguyên:")
            logging.debug(f"- Tổng số luồng sử dụng: {TOTAL_THREADS}/12 luồng")
            logging.debug(f"- Số mô hình chạy song song: {parallel_models}")
            logging.debug(f"- Số luồng mỗi mô hình: {threads_per_model}")
            
            # Thời gian bắt đầu xử lý mô hình
            models_start_time = time.time()
            
            # Chạy song song tất cả các mô hình (cải tiến lớn so với phiên bản trước)
            results_parallel = run_multiple_models(
                X=df_pca,
                models=selected_models,
                k_range=results['k_range'],
                selected_k=max_k,
                use_pca=use_pca,
                selected_features=list(df_pca.columns),
                explained_variance_ratio=0.95,
                max_workers=parallel_models,  # Số mô hình chạy song song
                threads_per_model=threads_per_model  # Số luồng cho mỗi mô hình
            )
            
            # Thời gian hoàn thành xử lý mô hình
            models_time = time.time() - models_start_time
            logging.debug(f"Tất cả mô hình hoàn thành sau {models_time:.2f} giây")
              # Cập nhật kết quả từ xử lý song song
            results['models'] = results_parallel['models']
            results['cvi_scores'] = results_parallel['cvi_scores']
            results['plots'] = results_parallel['plots']
            model_times = results_parallel['processing_times']
              # Xử lý lỗi nếu không có mô hình nào chạy thành công
            if not results['models']:
                flash('Không có mô hình nào chạy thành công. Vui lòng kiểm tra dữ liệu.')
                return redirect(url_for('select_model'))
            processing_time = time.time() - start_time
            logging.debug(f"Total processing time: {processing_time:.2f} seconds")
            
            # Gợi ý k tối ưu song song cho tất cả mô hình sau khi tất cả đã chạy xong
            if results['models']:
                try:
                    logging.debug("Đang gợi ý k tối ưu song song cho tất cả mô hình")
                      # Giới hạn số luồng cho quá trình gợi ý (tối đa 2)
                    optimal_suggestion_threads = min(2, max(1, cpu_count // 4))
                    
                    # Gợi ý song song sử dụng Wiroonsri và Starczewski
                    start_suggest_time = time.time()
                    optimal_k_suggestions = suggest_optimal_k_parallel(
                        model_results=results,
                        use_wiroonsri_starczewski=True,
                        max_workers=optimal_suggestion_threads
                    )
                    suggest_time = time.time() - start_suggest_time
                    
                    # Cập nhật kết quả với gợi ý mới
                    results['optimal_k_suggestions'] = optimal_k_suggestions
                    
                    # Cập nhật thông tin thời gian xử lý
                    results['processing_info'] = {
                        'total_time': processing_time,
                        'models_time': models_time,
                        'suggestion_time': suggest_time,
                        'model_times': model_times,
                        'parallel_info': {
                            'models_run_parallel': parallel_models,
                            'threads_per_model': threads_per_model,
                            'suggestion_threads': optimal_suggestion_threads
                        }
                    }
                    
                    logging.debug(f"Hoàn thành gợi ý k tối ưu trong {suggest_time:.2f} giây")
                except Exception as e:
                    logging.error(f"Lỗi khi gợi ý k tối ưu song song: {str(e)}")
                    # Fallback: sử dụng k=3 mặc định nếu có lỗi
                    for model_name in results['models']:
                        results['optimal_k_suggestions'][model_name] = {
                            'k': 3,
                            'reasoning': f'Lỗi gợi ý tự động: {str(e)}',
                            'method': 'fallback'
                        }
                
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
            'optimal_k_suggestions': {}        }
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    # Giữ lại k_range và selected_k mặc định, chỉ load models và kết quả khác
                    original_k_range = data['k_range']
                    original_selected_k = data['selected_k']
                    data.update(results)
                    data['k_range'] = original_k_range  # Khôi phục k_range mặc định
                    data['selected_k'] = original_selected_k  # Khôi phục selected_k mặc định
                logging.debug(f"Loaded previous results: {len(data.get('models', []))} models")
            except Exception as e:
                logging.error(f"Error loading results file: {str(e)}")
        
        return render_template('select_model_dashkit.html', data=data)
        
    except Exception as e:
        logging.error(f"Error in select_model_dashkit GET: {str(e)}")
        flash(f'Lỗi khi tải trang: {str(e)}')
        return redirect(url_for('dashkit_index'))