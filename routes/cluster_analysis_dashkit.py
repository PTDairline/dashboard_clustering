from flask import render_template, request, redirect, url_for, flash, current_app, jsonify
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.cluster_analysis import analyze_cluster_characteristics, get_cluster_predictions, create_cluster_comparison_table

def cluster_analysis_dashkit():
    """Trang phân tích đặc trưng cụm - phiên bản Dashkit"""
    try:
        # Kiểm tra có kết quả model không (từ BCVI)
        model_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'model_results.json')
        if not os.path.exists(model_results_file):
            flash("Vui lòng thực hiện BCVI trước khi phân tích cụm.", "warning")
            return redirect(url_for('bcvi'))
        
        # Load model results
        try:
            with open(model_results_file, 'r', encoding='utf-8') as f:
                bcvi_data = json.load(f)
            logging.info("Successfully loaded model results data")
        except Exception as e:
            logging.error(f"Error loading model results: {str(e)}")
            flash("File kết quả mô hình bị hỏng. Vui lòng thực hiện BCVI lại.", "error")
            return redirect(url_for('bcvi'))
        
        # Load dữ liệu gốc và PCA
        original_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')
        pca_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_data.pkl')
        
        if not os.path.exists(original_data_file):
            flash("Không tìm thấy dữ liệu gốc. Vui lòng tải dữ liệu lại.", "error")
            return redirect(url_for('dashkit_index'))
            
        if not os.path.exists(pca_data_file):
            flash("Không tìm thấy dữ liệu đã xử lý. Vui lòng xử lý dữ liệu lại.", "error")
            return redirect(url_for('process_data_dashkit'))
        
        try:
            original_data = pd.read_pickle(original_data_file)
            pca_data = pd.read_pickle(pca_data_file)
            logging.info("Successfully loaded data files")
        except Exception as e:
            logging.error(f"Error loading data files: {str(e)}")
            flash(f"Lỗi khi đọc dữ liệu: {str(e)}", "error")
            return redirect(url_for('dashkit_index'))
        
        # Kiểm tra có đang dùng PCA hay không
        use_pca = True
        pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
        feature_names = list(pca_data.columns)
        clustering_data = pca_data.values
        original_feature_names = None
        
        if os.path.exists(pca_results_file):
            try:
                with open(pca_results_file, 'r', encoding='utf-8') as f:
                    pca_results = json.load(f)
                    if pca_results.get('no_pca', False):
                        use_pca = False
                        selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
                        if os.path.exists(selected_features_file):
                            with open(selected_features_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    selected_features = [f.strip() for f in content.split(',')]
                                    numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                    available_features = [f for f in selected_features if f in numeric_cols]
                                    if available_features:
                                        clustering_data = original_data[available_features].copy()
                                        clustering_data = clustering_data.fillna(clustering_data.median(numeric_only=True))
                                        feature_names = available_features
                                        original_feature_names = available_features
                                    else:
                                        logging.warning("No valid numeric features found, using PCA data")
                                        use_pca = True
                                        feature_names = list(pca_data.columns)
                                        clustering_data = pca_data.values
                    else:
                        original_feature_names = pca_results.get('original_features', [])
            except Exception as e:
                logging.error(f"Error reading PCA results: {str(e)}")
                use_pca = True
                feature_names = list(pca_data.columns)
                clustering_data = pca_data.values
        
        # Khởi tạo data với giá trị mặc định
        data = {
            'models': bcvi_data.get('models', []),
            'cvi_scores': bcvi_data.get('cvi_scores', {}),
            'optimal_k': bcvi_data.get('optimal_k', {}),
            'use_pca': use_pca,
            'feature_names': feature_names,
            'original_feature_names': original_feature_names,
            'cluster_analysis': {}
        }
        logging.info(f"Data initialized - Models: {data['models']}, Use PCA: {use_pca}, Features: {len(feature_names)}")
        
        # Xử lý clear cache request
        if request.method == 'POST' and request.form.get('action') == 'clear_cache':
            analysis_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'cluster_analysis.pkl')
            try:
                if os.path.exists(analysis_file):
                    os.remove(analysis_file)
                    flash("Đã xóa cache phân tích cụm thành công.", "success")
                    logging.info("Manual cache clear successful")
                else:
                    flash("Không có cache để xóa.", "info")
            except Exception as e:
                flash(f"Lỗi khi xóa cache: {str(e)}", "error")
                logging.error(f"Error clearing cache: {str(e)}")
            return redirect(url_for('cluster_analysis_dashkit'))
        
        # Xử lý POST request - chọn mô hình và k để phân tích
        if request.method == 'POST' and request.form.get('action') != 'clear_cache':
            selected_model = request.form.get('model')
            selected_k = int(request.form.get('k', 3))
            
            logging.info(f"POST request - Model: {selected_model}, K: {selected_k}")
            
            if selected_model and selected_model in data['models']:
                try:
                    logging.info(f"Starting cluster analysis for {selected_model} with k={selected_k}")
                    
                    # Kiểm tra dữ liệu clustering
                    if clustering_data is None or len(clustering_data) == 0:
                        raise ValueError("Clustering data is empty")
                    
                    # Thực hiện phân cụm
                    labels = get_cluster_predictions(clustering_data, selected_model, selected_k)
                    logging.info(f"Got cluster labels: {len(labels)} samples, {len(set(labels))} unique clusters")
                    
                    # Chuẩn bị dữ liệu gốc cho phân tích
                    if use_pca:
                        selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
                        try:
                            if os.path.exists(selected_features_file):
                                with open(selected_features_file, 'r') as f:
                                    content = f.read().strip()
                                    if content:
                                        original_features = [f.strip() for f in content.split(',')]
                                        numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                        available_original_features = [f for f in original_features if f in numeric_cols]
                                        logging.info(f"Found {len(available_original_features)}/{len(original_features)} numeric features in original data")
                                        
                                        if available_original_features:
                                            analysis_data = original_data[available_original_features].copy()
                                            if analysis_data.isna().any().any():
                                                nan_counts = analysis_data.isna().sum()
                                                for col, count in nan_counts.items():
                                                    if count > 0:
                                                        logging.warning(f"Column '{col}' has {count} NaN values")
                                                analysis_data = analysis_data.fillna(analysis_data.median(numeric_only=True))
                                                if analysis_data.isna().any().any():
                                                    nan_count = analysis_data.isna().sum().sum()
                                                    logging.warning(f"Found {nan_count} NaN values in analysis_data after fillna, dropping rows")
                                                    analysis_data = analysis_data.dropna()
                                                    if len(analysis_data) == 0:
                                                        logging.error("All rows dropped due to NaN values, cannot proceed with clustering")
                                                        flash("Lỗi: Dữ liệu chứa quá nhiều giá trị NaN, không thể thực hiện phân cụm.", "error")
                                                        return redirect(url_for('process_data_dashkit'))
                                        else:
                                            numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                            analysis_data = original_data[numeric_cols].copy()
                                            analysis_data = analysis_data.fillna(analysis_data.median(numeric_only=True))
                                            logging.info(f"Fallback: using {len(numeric_cols)} numeric columns from original data")
                                    else:
                                        numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                        analysis_data = original_data[numeric_cols].copy()
                                        analysis_data = analysis_data.fillna(analysis_data.median(numeric_only=True))
                                        logging.info(f"No selected features found, using {len(numeric_cols)} numeric columns from original data")
                            else:
                                numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                analysis_data = original_data[numeric_cols].copy()
                                analysis_data = analysis_data.fillna(analysis_data.median(numeric_only=True))
                                logging.info(f"No selected features file found, using {len(numeric_cols)} numeric columns from original data")
                        except Exception as e:
                            logging.error(f"Error preparing analysis data: {str(e)}")
                            numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                            analysis_data = original_data[numeric_cols].copy()
                            analysis_data = analysis_data.fillna(analysis_data.median(numeric_only=True))
                    else:
                        analysis_data = pd.DataFrame(clustering_data, columns=feature_names)
                        original_feature_names = feature_names
                        logging.info(f"Not using PCA, analysis_data = clustering_data with shape {analysis_data.shape}")
                    
                    # Phân tích đặc trưng cụm
                    cluster_analysis = analyze_cluster_characteristics(
                        X=clustering_data,
                        labels=labels,
                        feature_names=feature_names,
                        original_data=analysis_data,
                        original_feature_names=original_feature_names,
                        top_features=len(analysis_data.columns)
                    )
                    logging.info(f"Analyzed clusters with {'PCA' if use_pca else 'original features'} using {len(original_feature_names or feature_names)} feature columns")
                    
                    if not cluster_analysis:
                        logging.error("Error in cluster analysis: Cluster analysis returned empty results")
                        flash("Lỗi trong phân tích cụm: Kết quả phân tích cụm rỗng.", "error")
                        return redirect(url_for('cluster_analysis_dashkit'))
                    
                    # Tạo bảng so sánh
                    comparison_table = create_cluster_comparison_table(
                        cluster_analysis, 
                        original_feature_names if use_pca and original_feature_names else feature_names
                    )
                    logging.info(f"Created comparison table with {len(original_feature_names if use_pca and original_feature_names else feature_names)} features")
                    
                    # Khởi tạo các biến cho dữ liệu PCA và đặc trưng gốc
                    pca_component_info = {}
                    pca_full_info = {}
                    original_feature_means = {}
                    reconstructed_clusters = {'comparison_table': []}
                    
                    if use_pca and original_feature_names:
                        try:
                            with open(pca_results_file, 'r', encoding='utf-8') as f:
                                pca_results = json.load(f)
                                
                                if 'top_features_by_component' in pca_results:
                                    pca_component_info = pca_results['top_features_by_component']
                                    logging.info(f"Loaded PCA component info with {len(pca_component_info)} components")
                                
                                if 'full_pca_loadings' in pca_results:
                                    pca_full_info = pca_results['full_pca_loadings']
                                
                                components_matrix = None
                                if 'components_matrix' in pca_results:
                                    components_matrix = np.array(pca_results['components_matrix'])
                                else:
                                    components_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_components.npy')
                                    if os.path.exists(components_file):
                                        with open(components_file, 'rb') as f:
                                            components_matrix = np.load(f)
                                
                                if components_matrix is not None and original_feature_names:
                                    orig_features_data = original_data[original_feature_names].copy()
                                    orig_features_data = orig_features_data.fillna(orig_features_data.median(numeric_only=True))
                                    orig_features_data['cluster'] = labels
                                    
                                    for cluster_id in np.unique(labels):
                                        cluster_data = orig_features_data[orig_features_data['cluster'] == cluster_id]
                                        original_feature_means[f'Cụm {cluster_id}'] = {}
                                        for feature in original_feature_names:
                                            try:
                                                mean_val = cluster_data[feature].mean()
                                                original_feature_means[f'Cụm {cluster_id}'][feature] = float(mean_val) if not pd.isna(mean_val) else 0.0
                                            except Exception as e:
                                                logging.error(f"Error calculating mean for {feature}: {str(e)}")
                                    
                                    reconstructed_table = []
                                    features_with_differences = []
                                    
                                    for feature in original_feature_names:
                                        row = {'Feature': feature}
                                        feature_values = {}
                                        for cluster_id in np.unique(labels):
                                            cluster_key = f'Cụm {cluster_id}'
                                            if cluster_key in original_feature_means and feature in original_feature_means[cluster_key]:
                                                value = original_feature_means[cluster_key][feature]
                                                row[cluster_key] = f"{value:.2f}"
                                                feature_values[int(cluster_id)] = value
                                        
                                        if feature_values:
                                            feature_values = {int(k): v for k, v in feature_values.items()}
                                            max_cluster = max(feature_values, key=feature_values.get)
                                            min_cluster = min(feature_values, key=feature_values.get)
                                            difference = feature_values[max_cluster] - feature_values[min_cluster]
                                            abs_difference = abs(difference)
                                            row['Cao nhất'] = f"Cụm {max_cluster}"
                                            row['Thấp nhất'] = f"Cụm {min_cluster}"
                                            row['Chênh lệch'] = f"{difference:.2f}"
                                            features_with_differences.append((abs_difference, row))
                                    
                                    max_features_to_show = min(30, len(features_with_differences))
                                    features_with_differences.sort(reverse=True, key=lambda x: x[0])
                                    top_features = features_with_differences[:max_features_to_show]
                                    reconstructed_table = [item[1] for item in top_features]
                                    reconstructed_clusters['comparison_table'] = reconstructed_table
                        except Exception as e:
                            logging.error(f"Error reconstructing original feature data: {str(e)}")
                    
                    # Lưu kết quả với metadata
                    data['cluster_analysis'] = {
                        'selected_model': selected_model,
                        'selected_k': selected_k,
                        'analysis': cluster_analysis,
                        'comparison_table': comparison_table.to_dict('records') if not comparison_table.empty else [],
                        'labels': labels.tolist(),
                        'total_samples': len(labels),
                        'used_pca_for_clustering': use_pca,
                        'analysis_features': original_feature_names if use_pca else feature_names,
                        'clustering_features': feature_names,
                        'available_models': data['models'],
                        'cache_timestamp': pd.Timestamp.now().isoformat(),
                        'data_shape': {
                            'clustering_data': clustering_data.shape,
                            'analysis_data': analysis_data.shape
                        },
                        'pca_component_info': pca_component_info,
                        'original_features_info': original_feature_names,
                        'reconstructed_clusters': reconstructed_clusters
                    }
                    
                    # Lưu kết quả phân tích
                    analysis_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'cluster_analysis.pkl')
                    try:
                        pd.to_pickle(data['cluster_analysis'], analysis_file)
                        logging.info("Saved cluster analysis results")
                    except Exception as e:
                        logging.warning(f"Could not save analysis file: {str(e)}")
                    
                    flash(f"Đã phân tích thành công {selected_k} cụm cho mô hình {selected_model}.", "success")
                    
                    return render_template('cluster_analysis_dashkit.html', data=data)
                
                except Exception as e:
                    logging.error(f"Error in cluster analysis: {str(e)}")
                    flash(f"Lỗi khi phân tích cụm: {str(e)}", "error")
                    return redirect(url_for('cluster_analysis_dashkit'))
            else:
                flash("Vui lòng chọn mô hình hợp lệ.", "warning")
        
        # GET request - load kết quả phân tích nếu có và cache hợp lệ
        analysis_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'cluster_analysis.pkl')
        if os.path.exists(analysis_file):
            try:
                cached_analysis = pd.read_pickle(analysis_file)
                
                cache_valid = True
                cache_reasons = []
                
                try:
                    analysis_time = os.path.getmtime(analysis_file)
                    files_to_check = [
                        original_data_file,
                        pca_data_file, 
                        model_results_file
                    ]
                    
                    config_files = [
                        os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'),
                        os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json'),
                        os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt')
                    ]
                    
                    for config_file in config_files:
                        if os.path.exists(config_file):
                            files_to_check.append(config_file)
                    
                    latest_file_time = max(os.path.getmtime(f) for f in files_to_check if os.path.exists(f))
                    
                    if analysis_time < latest_file_time:
                        cache_valid = False
                        cache_reasons.append("Config or data files newer than analysis")
                except Exception as e:
                    logging.warning(f"Could not check file timestamps: {str(e)}")
                    cache_valid = False
                    cache_reasons.append("Timestamp check failed")
                
                if cache_valid:
                    cached_data_shape = cached_analysis.get('data_shape', None)
                    current_data_shape = (len(clustering_data), len(feature_names))
                    
                    if cached_data_shape != current_data_shape:
                        cache_valid = False
                        cache_reasons.append(f"Data shape changed: {cached_data_shape} -> {current_data_shape}")
                
                if cache_valid and 'used_pca_for_clustering' in cached_analysis:
                    if cached_analysis['used_pca_for_clustering'] != use_pca:
                        cache_valid = False
                        cache_reasons.append(f"PCA config changed: {cached_analysis['used_pca_for_clustering']} -> {use_pca}")
                
                if cache_valid and 'clustering_features' in cached_analysis:
                    current_clustering_features = feature_names
                    if set(cached_analysis['clustering_features']) != set(current_clustering_features):
                        cache_valid = False
                        cache_reasons.append("Clustering features changed")
                
                if cache_valid and 'analysis_features' in cached_analysis:
                    current_analysis_features = original_feature_names if use_pca else feature_names
                    if set(cached_analysis['analysis_features']) != set(current_analysis_features):
                        cache_valid = False
                        cache_reasons.append("Analysis features changed")
                
                if cache_valid and 'available_models' in cached_analysis:
                    current_models = set(data['models'])
                    cached_models = set(cached_analysis.get('available_models', []))
                    if current_models != cached_models:
                        cache_valid = False
                        cache_reasons.append("Available models changed")
                
                if cache_valid:
                    data['cluster_analysis'] = cached_analysis
                    logging.info("Loaded valid cached cluster analysis")
                    flash("Đã tải kết quả phân tích từ cache (dữ liệu không thay đổi).", "info")
                else:
                    logging.info(f"Cache invalid: {', '.join(cache_reasons)}. Clearing cache.")
                    try:
                        os.remove(analysis_file)
                        logging.info("Removed invalid cached analysis file")
                    except Exception as e:
                        logging.warning(f"Could not remove cache file: {str(e)}")
            except Exception as e:
                logging.error(f"Error loading cluster analysis: {str(e)}")
                try:
                    os.remove(analysis_file)
                    logging.info("Removed corrupted analysis file")
                except:
                    pass
        
        return render_template('cluster_analysis_dashkit.html', data=data)
        
    except Exception as e:
        logging.error(f"Error in cluster_analysis_dashkit: {str(e)}")
        flash(f"Lỗi khi tải trang phân tích cụm: {str(e)}", "error")
        return redirect(url_for('bcvi'))