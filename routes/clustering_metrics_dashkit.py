from flask import render_template, flash, redirect, url_for, current_app
import os
import pandas as pd
import json
import logging
import numpy as np
import time

def clustering_metrics_dashkit():
    """Hiển thị chỉ số đánh giá phân cụm - phiên bản Dashkit"""
    try:
        # Kiểm tra xem có kết quả mô hình không
        model_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'model_results.json')
        
        if not os.path.exists(model_results_file):
            flash('Vui lòng chạy mô hình phân cụm trước khi xem chỉ số đánh giá.')
            return redirect(url_for('select_model'))
          # Tối ưu hóa: Load kết quả mô hình với error handling tốt hơn
        try:
            with open(model_results_file, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in model results file: {str(e)}")
            flash('File kết quả mô hình bị lỗi. Vui lòng chạy lại phân cụm.')
            return redirect(url_for('select_model'))
        except Exception as e:
            logging.error(f"Error reading model results file: {str(e)}")
            flash('Không thể đọc file kết quả mô hình.')
            return redirect(url_for('select_model'))
        
        # Tối ưu hóa: Xử lý dữ liệu nhanh hơn với numpy operations
        results = []
        plot_urls = {}
        processing_start = time.time()
        
        if 'models' in model_data and 'cvi_scores' in model_data:
            for model_name in model_data['models']:
                if model_name in model_data['cvi_scores']:
                    # Vectorized processing cho tất cả k values
                    model_scores = model_data['cvi_scores'][model_name]
                    k_values = [int(k) for k in model_scores.keys()]
                    
                    for k in sorted(k_values):
                        k_str = str(k)
                        scores = model_scores[k_str]
                        
                        try:
                            # Tối ưu hóa: Xử lý giá trị một cách an toàn và nhanh chóng
                            result = {
                                'method': model_name,
                                'k': k,
                                'n_clusters': k,
                                'n_noise': 0,  # Mặc định cho các thuật toán không có noise
                                'Silhouette': float(scores.get('silhouette', 0)) if scores.get('silhouette') is not None else 0.0,
                                'Calinski_Harabasz': float(scores.get('calinski_harabasz', 0)) if scores.get('calinski_harabasz') is not None else 0.0,
                                'Davies_Bouldin': float(scores.get('davies_bouldin', float('inf'))) if scores.get('davies_bouldin') is not None else float('inf'),
                                'Starczewski': float(scores.get('starczewski', 0)) if scores.get('starczewski') is not None else 0.0,
                                'Wiroonsri': float(scores.get('wiroonsri', 0)) if scores.get('wiroonsri') is not None else 0.0
                            }
                            results.append(result)
                        except (ValueError, TypeError) as e:
                            logging.error(f"Error processing scores for {model_name}, k={k}: {str(e)}")
                            continue
          # Tối ưu hóa: Tạo plot URLs hiệu quả hơn
        if 'plots' in model_data:
            for model_name, model_plots in model_data['plots'].items():
                # Silhouette plot
                if 'silhouette' in model_plots and model_plots['silhouette'].get('plot'):
                    plot_urls[f'{model_name}_silhouette'] = f"data:image/png;base64,{model_plots['silhouette']['plot']}"
                
                # Elbow plot - chỉ cho KMeans và FuzzyCMeans
                if model_name in ['KMeans', 'FuzzyCMeans'] and 'elbow' in model_plots and model_plots['elbow'].get('plot'):
                    plot_urls[f'{model_name}_elbow'] = f"data:image/png;base64,{model_plots['elbow']['plot']}"
        
        processing_time = time.time() - processing_start
        logging.debug(f"Data processing completed in {processing_time:.3f} seconds")
        
        # Tối ưu hóa: Thêm optimal k suggestions vào data
        optimal_suggestions = {}
        if 'optimal_k_suggestions' in model_data:
            optimal_suggestions = model_data['optimal_k_suggestions']
          # Nếu không có kết quả, tạo dữ liệu mẫu để hiển thị giao diện
        if not results:
            flash('Không có dữ liệu chỉ số đánh giá. Hiển thị giao diện mẫu.')
            results = [
                {
                    'method': 'KMeans',
                    'k': 2,
                    'n_clusters': 2,
                    'n_noise': 0,
                    'Silhouette': 0.583,
                    'Calinski_Harabasz': 142.567,
                    'Davies_Bouldin': 0.892,
                    'Starczewski': 0.234,
                    'Wiroonsri': 0.156
                },
                {
                    'method': 'KMeans',
                    'k': 3,
                    'n_clusters': 3,
                    'n_noise': 0,
                    'Silhouette': 0.632,
                    'Calinski_Harabasz': 178.234,
                    'Davies_Bouldin': 0.743,
                    'Starczewski': 0.198,
                    'Wiroonsri': 0.134
                },
                {
                    'method': 'GMM',
                    'k': 2,
                    'n_clusters': 2,
                    'n_noise': 0,
                    'Silhouette': 0.567,
                    'Calinski_Harabasz': 135.421,
                    'Davies_Bouldin': 0.923,
                    'Starczewski': 0.245,
                    'Wiroonsri': 0.167
                }
            ]
            optimal_suggestions = {}
        
        logging.debug(f"Loaded {len(results)} clustering results")
        
        return render_template('clustering_metrics_dashkit.html', 
                             results=results, 
                             plot_urls=plot_urls,
                             optimal_suggestions=optimal_suggestions)
        
    except Exception as e:
        logging.error(f"Error in clustering_metrics_dashkit: {str(e)}")
        flash(f'Lỗi khi tải chỉ số đánh giá: {str(e)}')
        return redirect(url_for('dashkit_index'))