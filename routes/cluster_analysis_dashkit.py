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
        
        # Load model results với error handling  
        try:
            with open(model_results_file, 'r', encoding='utf-8') as f:
                bcvi_data = json.load(f)
            logging.info("Successfully loaded model results data")
        except Exception as e:
            logging.error(f"Error loading model results: {str(e)}")
            flash("File kết quả mô hình bị hỏng. Vui lòng thực hiện BCVI lại.", "error")
            return redirect(url_for('bcvi'))
        
        # Load dữ liệu gốc
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
        
        if os.path.exists(pca_results_file):
            try:
                with open(pca_results_file, 'r', encoding='utf-8') as f:
                    pca_results = json.load(f)
                    if pca_results.get('no_pca', False):
                        use_pca = False
                        # Nếu không dùng PCA, sử dụng features gốc đã chọn
                        selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
                        if os.path.exists(selected_features_file):
                            with open(selected_features_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    selected_features = content.split(',')
                                    selected_features = [f.strip() for f in selected_features]  # Clean whitespace
                                    feature_names = selected_features
                                    # Kiểm tra features có tồn tại trong original_data
                                    available_features = [f for f in selected_features if f in original_data.columns]
                                    if available_features:
                                        clustering_data = original_data[available_features].fillna(original_data[available_features].mean()).values
                                        feature_names = available_features
                                    else:
                                        logging.warning("No valid features found, using PCA data")
                                        use_pca = True
                                        feature_names = list(pca_data.columns)
                                        clustering_data = pca_data.values
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
                      # Chuẩn bị dữ liệu gốc cho phân tích (LUÔN SỬ DỤNG DỮ LIỆU GỐC CHƯA CHUẨN HÓA)
                    if use_pca:
                        # Nếu dùng PCA, sử dụng dữ liệu gốc với features đã chọn
                        selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
                        try:
                            if os.path.exists(selected_features_file):
                                with open(selected_features_file, 'r') as f:
                                    content = f.read().strip()
                                    if content:
                                        original_features = content.split(',')
                                        original_features = [f.strip() for f in original_features]
                                        
                                        # Kiểm tra features có tồn tại
                                        available_original_features = [f for f in original_features if f in original_data.columns]
                                        logging.info(f"Found {len(available_original_features)}/{len(original_features)} features in original data")
                                        
                                        if available_original_features:
                                            # Dùng dữ liệu gốc THỰC SỰ (chưa chuẩn hóa)
                                            analysis_data = original_data[available_original_features].copy()
                                            analysis_data = analysis_data.fillna(analysis_data.mean())
                                            logging.info(f"Using {len(available_original_features)} available ORIGINAL features for analysis")
                                        else:
                                            # Fallback to all numeric columns from original data
                                            numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                            analysis_data = original_data[numeric_cols].copy()
                                            analysis_data = analysis_data.fillna(analysis_data.mean())
                                            logging.info(f"Fallback: using {len(numeric_cols)} numeric columns from ORIGINAL data")
                                    else:
                                        # No content in selected_features.txt
                                        numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                        analysis_data = original_data[numeric_cols].copy()
                                        analysis_data = analysis_data.fillna(analysis_data.mean())
                                        logging.info(f"No selected features found, using {len(numeric_cols)} numeric columns from ORIGINAL data")
                            else:
                                # No selected_features.txt file
                                numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                                analysis_data = original_data[numeric_cols].copy()
                                analysis_data = analysis_data.fillna(analysis_data.mean())
                                logging.info(f"No selected features file found, using {len(numeric_cols)} numeric columns from ORIGINAL data")
                                
                            # Kiểm tra NaN sau khi xử lý
                            if analysis_data.isna().any().any():
                                nan_count = analysis_data.isna().sum().sum()
                                logging.warning(f"Found {nan_count} NaN values in analysis_data after fillna, replacing with zeros")
                                analysis_data = analysis_data.fillna(0)
                                
                        except Exception as e:
                            logging.error(f"Error preparing analysis data: {str(e)}")
                            # Fallback to safe option with original data
                            logging.info("Falling back to numeric columns from original data")
                            numeric_cols = original_data.select_dtypes(include=[np.number]).columns.tolist()
                            analysis_data = original_data[numeric_cols].copy()
                            analysis_data = analysis_data.fillna(analysis_data.mean())
                    else:
                        # Nếu không dùng PCA, dùng chính dữ liệu đang clustering
                            analysis_data = pd.DataFrame(clustering_data, columns=feature_names)
                            
                            # Tạo bảng trung bình các đặc trưng gốc theo cụm (vì không dùng PCA nên là dùng trực tiếp)
                            try:
                                # Sử dụng trực tiếp các đặc trưng gốc đã chọn
                                original_features_info = feature_names
                                
                                # Tạo DataFrame với cả dữ liệu gốc và nhãn cụm
                                orig_features_data = analysis_data.copy()
                                orig_features_data['cluster'] = labels
                                
                                # Tính trung bình của các đặc trưng gốc theo cụm
                                for cluster_id in np.unique(labels):
                                    cluster_data = orig_features_data[orig_features_data['cluster'] == cluster_id]
                                    original_feature_means[f'Cụm {cluster_id}'] = {}
                                    
                                    # Tính trung bình cho mỗi đặc trưng gốc
                                    for feature in original_features_info:
                                        try:
                                            mean_val = cluster_data[feature].mean()
                                            original_feature_means[f'Cụm {cluster_id}'][feature] = mean_val
                                        except Exception as e:
                                            logging.error(f"Error calculating mean for {feature}: {str(e)}")
                                
                                # Tạo bảng so sánh các đặc trưng gốc giữa các cụm
                                reconstructed_table = []
                                features_with_differences = []  # Để sắp xếp theo chênh lệch
                                
                                for feature in original_features_info:
                                    row = {'Feature': feature}                                    
                                    feature_values = {}
                                      # Thêm giá trị trung bình cho từng cụm
                                    for cluster_id in np.unique(labels):
                                        cluster_key = f'Cụm {cluster_id}'
                                        if cluster_key in original_feature_means and feature in original_feature_means[cluster_key]:
                                            value = original_feature_means[cluster_key][feature]
                                            row[cluster_key] = f"{value:.2f}"
                                            feature_values[int(cluster_id)] = value  # Chuyển sang int để đảm bảo đồng nhất
                                    
                                    # Tìm cụm có giá trị cao nhất và thấp nhất
                                    if feature_values:
                                        # Đảm bảo các keys là Python int (thay vì numpy.int32)
                                        feature_values = {int(k): v for k, v in feature_values.items()}
                                        max_cluster = max(feature_values, key=feature_values.get)
                                        min_cluster = min(feature_values, key=feature_values.get)
                                        difference = feature_values[max_cluster] - feature_values[min_cluster]
                                        abs_difference = abs(difference)  # Giá trị tuyệt đối để sắp xếp
                                        
                                        row['Cao nhất'] = f"Cụm {max_cluster}"
                                        row['Thấp nhất'] = f"Cụm {min_cluster}"
                                        row['Chênh lệch'] = f"{difference:.2f}"
                                        
                                        # Lưu thông tin để sắp xếp sau
                                        features_with_differences.append((abs_difference, row))
                                
                                # Sắp xếp và giới hạn số lượng hiển thị nếu quá nhiều
                                max_features_to_show = min(30, len(features_with_differences))  # Tối đa hiển thị 30 features
                                
                                # Sắp xếp theo chênh lệch giảm dần
                                features_with_differences.sort(reverse=True, key=lambda x: x[0])
                                
                                # Chỉ lấy top features có chênh lệch lớn nhất
                                top_features = features_with_differences[:max_features_to_show]
                                reconstructed_table = [item[1] for item in top_features]
                                
                                reconstructed_clusters['comparison_table'] = reconstructed_table
                            except Exception as e:
                                logging.error(f"Error creating original feature comparison when not using PCA: {str(e)}")
                                
                    logging.info(f"Not using PCA, analysis_data = clustering_data with shape {analysis_data.shape}")
                    
                    logging.info(f"Analysis data shape: {analysis_data.shape}")                    # Phân tích đặc trưng cụm
                    # Sử dụng đặc trưng gốc thay vì PCA components
                    if use_pca:
                        # Khi dùng PCA, phân tích trên dữ liệu gốc với original_features_info
                        selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
                        if os.path.exists(selected_features_file):
                            with open(selected_features_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    original_features = content.split(',')
                                    original_features = [f.strip() for f in original_features]
                                    analysis_feature_names = original_features
                                    logging.info(f"Using original features for analysis: {len(analysis_feature_names)} features")
                                else:
                                    # Fallback to PCA if no original features found
                                    analysis_feature_names = list(analysis_data.columns)
                                    logging.info(f"No original features found, using PCA components: {analysis_feature_names}")
                        else:
                            # Fallback to PCA if no file exists
                            analysis_feature_names = list(analysis_data.columns)
                            logging.info(f"No selected features file found, using PCA components: {analysis_feature_names}")
                    else:
                        # Không dùng PCA, dùng trực tiếp feature_names
                        analysis_feature_names = feature_names
                        logging.info(f"Not using PCA, using original feature_names: {feature_names}")                    # Phân tích đặc trưng cụm với dữ liệu gốc thực sự
                    if use_pca:
                        # Khi dùng PCA, sử dụng tên cột từ analysis_data (dữ liệu gốc)
                        analysis_feature_names = list(analysis_data.columns)
                        cluster_analysis = analyze_cluster_characteristics(
                            X=clustering_data,  # Dữ liệu đã chuẩn hóa để phân cụm
                            labels=labels,
                            feature_names=feature_names,  # Tên features của clustering_data (PC1, PC2, etc.)
                            original_data=analysis_data,  # Dữ liệu gốc thực sự
                            top_features=len(analysis_data.columns)
                        )
                        logging.info(f"Analyzed clusters with PCA using {len(analysis_feature_names)} ORIGINAL feature columns")
                    else:
                        # Không dùng PCA, analysis_data chính là clustering_data
                        analysis_feature_names = feature_names
                        cluster_analysis = analyze_cluster_characteristics(
                            X=clustering_data,
                            labels=labels,
                            feature_names=feature_names,
                            original_data=analysis_data,
                            top_features=len(analysis_data.columns)
                        )
                        logging.info(f"Analyzed clusters without PCA using {len(analysis_feature_names)} feature columns")
                    
                    if not cluster_analysis:
                        raise ValueError("Cluster analysis returned empty results")
                    
                    # Xác định feature names cho analysis
                    analysis_feature_names = list(analysis_data.columns)
                    logging.info(f"Analysis feature names: {len(analysis_feature_names)} features")
                      # Tạo bảng so sánh
                    # Nếu dùng PCA, sử dụng đặc trưng gốc để hiển thị
                    if use_pca and 'original_features_info' in locals() and original_features_info:
                        # Dùng đặc trưng gốc để tạo bảng so sánh
                        comparison_table = create_cluster_comparison_table(
                            cluster_analysis,
                            original_features_info  # Sử dụng đặc trưng gốc thay vì PC
                        )
                        logging.info(f"Created comparison table with original features: {len(original_features_info)}")
                    else:
                        # Không dùng PCA hoặc không có đặc trưng gốc
                        comparison_table = create_cluster_comparison_table(
                            cluster_analysis, 
                            analysis_feature_names
                        )
                        logging.info(f"Created comparison table with analysis features: {len(analysis_feature_names)}")# Khởi tạo các biến cho dữ liệu PCA và đặc trưng gốc
                    pca_component_info = {}
                    pca_full_info = {}
                    original_features_info = []
                    original_feature_means = {}
                    reconstructed_clusters = {'comparison_table': []}
                    
                    if use_pca:
                        # Đọc thông tin PCA từ file
                        pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
                        if os.path.exists(pca_results_file):
                            try:
                                with open(pca_results_file, 'r', encoding='utf-8') as f:
                                    pca_results = json.load(f)
                                    
                                    # Lấy thông tin về top features
                                    if 'top_features_by_component' in pca_results:
                                        pca_component_info = pca_results['top_features_by_component']
                                        logging.info(f"Loaded PCA component info with {len(pca_component_info)} components")
                                    
                                    # Lấy thông tin loadings đầy đủ
                                    if 'full_pca_loadings' in pca_results:
                                        pca_full_info = pca_results['full_pca_loadings']
                                    
                                    # Lấy thông tin về các đặc trưng gốc
                                    if 'original_features' in pca_results:
                                        original_features_info = pca_results['original_features']
                                    
                                    # Lấy ma trận components để thực hiện chuyển đổi ngược
                                    components_matrix = None
                                    if 'components_matrix' in pca_results:
                                        components_matrix = np.array(pca_results['components_matrix'])
                                    else:
                                        # Thử đọc từ file npy
                                        components_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_components.npy')
                                        if os.path.exists(components_file):
                                            try:
                                                with open(components_file, 'rb') as f:
                                                    components_matrix = np.load(f)
                                            except Exception as e:
                                                logging.error(f"Error loading PCA components matrix: {str(e)}")
                                    
                                    # Nếu có ma trận components và dữ liệu gốc, thực hiện chuyển đổi ngược
                                    if components_matrix is not None and original_features_info:
                                        try:
                                            # Đọc dữ liệu gốc
                                            original_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')
                                            if os.path.exists(original_data_file):
                                                original_df = pd.read_pickle(original_data_file)
                                                
                                                # Lấy dữ liệu các đặc trưng gốc đã chọn
                                                orig_features_data = original_df[original_features_info].fillna(original_df[original_features_info].mean())
                                                
                                                # Tính trung bình theo cụm cho đặc trưng gốc
                                                orig_features_data['cluster'] = labels
                                                
                                                # Tính trung bình của các đặc trưng gốc theo cụm
                                                for cluster_id in np.unique(labels):
                                                    cluster_data = orig_features_data[orig_features_data['cluster'] == cluster_id]
                                                    original_feature_means[f'Cụm {cluster_id}'] = {}
                                                    
                                                    # Tính trung bình cho mỗi đặc trưng gốc
                                                    for feature in original_features_info:
                                                        try:
                                                            mean_val = cluster_data[feature].mean()
                                                            original_feature_means[f'Cụm {cluster_id}'][feature] = mean_val
                                                        except Exception as e:
                                                            logging.error(f"Error calculating mean for {feature}: {str(e)}")
                                                
                                                # Tạo bảng so sánh các đặc trưng gốc giữa các cụm
                                                reconstructed_table = []
                                                features_with_differences = []  # Để sắp xếp theo chênh lệch
                                                
                                                for feature in original_features_info:
                                                    row = {'Feature': feature}                                                    
                                                    feature_values = {}
                                                      # Thêm giá trị trung bình cho từng cụm
                                                    for cluster_id in np.unique(labels):
                                                        cluster_key = f'Cụm {cluster_id}'
                                                        if cluster_key in original_feature_means and feature in original_feature_means[cluster_key]:
                                                            value = original_feature_means[cluster_key][feature]
                                                            row[cluster_key] = f"{value:.2f}"  # Sử dụng cluster_key thay vì cluster_id
                                                            feature_values[int(cluster_id)] = value  # Chuyển sang int để đảm bảo đồng nhất
                                                    
                                                    # Tìm cụm có giá trị cao nhất và thấp nhất
                                                    if feature_values:
                                                        # Đảm bảo các keys là Python int (thay vì numpy.int32)
                                                        feature_values = {int(k): v for k, v in feature_values.items()}
                                                        max_cluster = max(feature_values, key=feature_values.get)
                                                        min_cluster = min(feature_values, key=feature_values.get)
                                                        difference = feature_values[max_cluster] - feature_values[min_cluster]
                                                        abs_difference = abs(difference)  # Giá trị tuyệt đối để sắp xếp
                                                        
                                                        row['Cao nhất'] = f"Cụm {max_cluster}"
                                                        row['Thấp nhất'] = f"Cụm {min_cluster}"
                                                        row['Chênh lệch'] = f"{difference:.2f}"
                                                        
                                                        # Lưu thông tin để sắp xếp sau
                                                        features_with_differences.append((abs_difference, row))
                                                
                                                # Sắp xếp và giới hạn số lượng hiển thị nếu quá nhiều
                                                max_features_to_show = min(30, len(features_with_differences))  # Tối đa hiển thị 30 features
                                                
                                                # Sắp xếp theo chênh lệch giảm dần
                                                features_with_differences.sort(reverse=True, key=lambda x: x[0])
                                                
                                                # Chỉ lấy top features có chênh lệch lớn nhất
                                                top_features = features_with_differences[:max_features_to_show]
                                                reconstructed_table = [item[1] for item in top_features]
                                                
                                                reconstructed_clusters['comparison_table'] = reconstructed_table
                                        except Exception as e:
                                            logging.error(f"Error reconstructing original feature data: {str(e)}")
                                            
                            except Exception as e:
                                logging.error(f"Error loading PCA component info: {str(e)}")                    # Cập nhật lại feature names trong cluster_analysis với original_features_info nếu dùng PCA
                    if use_pca and original_features_info:
                        # Thay thế tên PC components bằng original features trong kết quả phân tích
                        # Áp dụng đặc biệt cho phần distinctive_features để hiển thị đặc trưng gốc
                        for cluster_id, cluster_data in cluster_analysis.items():
                            if 'distinctive_features' in cluster_data:
                                for feature_idx, feature_info in enumerate(cluster_data['distinctive_features']):
                                    if feature_info['feature'].startswith('PC'):
                                        try:
                                            # Tìm original feature tương ứng với PC component
                                            pc_idx = int(feature_info['feature'][2:]) - 1  # Lấy số từ 'PC1' -> 0
                                            if pc_idx < len(original_features_info):
                                                feature_info['feature'] = original_features_info[pc_idx]
                                                feature_info['is_pc_component'] = True
                                                feature_info['original_pc'] = f"PC{pc_idx+1}"
                                        except Exception as e:
                                            logging.error(f"Error replacing PC name: {str(e)}")
                    
                    # Lưu kết quả với metadata cho cache validation
                    data['cluster_analysis'] = {
                        'selected_model': selected_model,
                        'selected_k': selected_k,
                        'analysis': cluster_analysis,
                        'comparison_table': comparison_table.to_dict('records') if not comparison_table.empty else [],
                        'labels': labels.tolist(),
                        'total_samples': len(labels),
                        'used_pca_for_clustering': use_pca,
                        'analysis_features': analysis_feature_names,
                        'clustering_features': feature_names,
                        'available_models': data['models'],
                        'cache_timestamp': pd.Timestamp.now().isoformat(),
                        'data_shape': {
                            'clustering_data': clustering_data.shape,
                            'analysis_data': analysis_data.shape
                        },
                        'pca_component_info': pca_component_info,
                        'original_features_info': original_features_info,
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
                    
                except Exception as e:
                    logging.error(f"Error in cluster analysis: {str(e)}")
                    flash(f"Lỗi khi phân tích cụm: {str(e)}", "error")
            else:
                flash("Vui lòng chọn mô hình hợp lệ.", "warning")        
        else:
            # GET request - load kết quả phân tích nếu có và cache hợp lệ
            analysis_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'cluster_analysis.pkl')
            if os.path.exists(analysis_file):
                try:
                    cached_analysis = pd.read_pickle(analysis_file)
                    
                    # Kiểm tra cache có hợp lệ không bằng cách so sánh:
                    # 1. Timestamp của data files
                    # 2. Cấu hình PCA 
                    # 3. Selected features
                    # 4. PCA results file
                    # 5. Data shape/hash để phát hiện thay đổi nội dung
                    cache_valid = True
                    cache_reasons = []
                    
                    # Kiểm tra timestamp của tất cả related files
                    try:
                        analysis_time = os.path.getmtime(analysis_file)
                        files_to_check = [
                            original_data_file,
                            pca_data_file, 
                            model_results_file
                        ]
                        
                        # Kiểm tra thêm các config files
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
                    
                    # Kiểm tra data shape có thay đổi
                    if cache_valid:
                        try:
                            cached_data_shape = cached_analysis.get('data_shape', None)
                            current_data_shape = (len(clustering_data), len(feature_names))
                            
                            if cached_data_shape != current_data_shape:
                                cache_valid = False
                                cache_reasons.append(f"Data shape changed: {cached_data_shape} -> {current_data_shape}")
                        except Exception as e:
                            logging.warning(f"Could not check data shape: {str(e)}")
                            cache_valid = False
                            cache_reasons.append("Data shape check failed")
                    
                    # Kiểm tra cấu hình PCA có thay đổi
                    if cache_valid and 'used_pca_for_clustering' in cached_analysis:
                        if cached_analysis['used_pca_for_clustering'] != use_pca:
                            cache_valid = False
                            cache_reasons.append(f"PCA config changed: {cached_analysis['used_pca_for_clustering']} -> {use_pca}")
                    
                    # Kiểm tra clustering features có thay đổi
                    if cache_valid and 'clustering_features' in cached_analysis:
                        current_clustering_features = feature_names
                        if set(cached_analysis['clustering_features']) != set(current_clustering_features):
                            cache_valid = False
                            cache_reasons.append("Clustering features changed")
                    
                    # Kiểm tra analysis features có thay đổi
                    if cache_valid and 'analysis_features' in cached_analysis:
                        # Lấy current analysis features để so sánh
                        current_analysis_features = feature_names
                        if use_pca:
                            selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
                            if os.path.exists(selected_features_file):
                                with open(selected_features_file, 'r') as f:
                                    content = f.read().strip()
                                    if content:
                                        original_features = content.split(',')
                                        available_features = [f.strip() for f in original_features if f.strip() in original_data.columns]
                                        if available_features:
                                            current_analysis_features = available_features
                        
                        if set(cached_analysis['analysis_features']) != set(current_analysis_features):
                            cache_valid = False
                            cache_reasons.append("Analysis features changed")
                    
                    # Kiểm tra models available có thay đổi
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
                    # Xóa file bị hỏng
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