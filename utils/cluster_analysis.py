import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import logging

def analyze_cluster_characteristics(X, labels, feature_names, original_data=None, top_features=10):
    """
    Phân tích đặc trưng của từng cụm
    
    Args:
        X: Dữ liệu đã chuẩn hóa (numpy array hoặc DataFrame)
        labels: Nhãn cụm cho từng điểm dữ liệu
        feature_names: Tên các features (cho dữ liệu clustering)
        original_data: Dữ liệu gốc chưa chuẩn hóa (để hiển thị giá trị thực)
        top_features: Số lượng features đặc trưng nhất để hiển thị
    
    Returns:
        dict: Thông tin đặc trưng của từng cụm
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        logging.info(f"analyze_cluster_characteristics - Input shapes: X={X.shape if hasattr(X, 'shape') else 'unknown'}, "
                      f"labels={len(labels) if labels is not None else 'None'}, feature_names={len(feature_names)}")
                 
        # Kiểm tra NaN
        if isinstance(X, np.ndarray) and np.isnan(X).any():
            logging.warning("NaN values found in X, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0)
        
        # Chuyển đổi về DataFrame nếu cần (cho clustering data)
        if isinstance(X, np.ndarray):
            if X.shape[1] != len(feature_names):
                logging.warning(f"Feature names count ({len(feature_names)}) doesn't match X columns ({X.shape[1]}). Using generic names.")
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
            
        # Thêm cột cluster cho clustering data
        df['cluster'] = labels
        
        # Chuẩn bị dữ liệu gốc
        if original_data is not None:
            if isinstance(original_data, np.ndarray):
                # Tạo DataFrame từ original_data
                original_feature_names = [f'Original_Feature_{i}' for i in range(original_data.shape[1])]
                original_df = pd.DataFrame(original_data, columns=original_feature_names)
            else:
                original_df = original_data.copy()
                
            # Kiểm tra NaN
            if original_df.isna().any().any():
                logging.warning("NaN values found in original_data, filling with column means")
                original_df = original_df.fillna(original_df.mean())
                
            # Thêm cột cluster
            if len(original_df) == len(labels):
                original_df['cluster'] = labels
                logging.info(f"Added cluster labels to original data. Shape: {original_df.shape}")
            else:
                logging.warning(f"Length mismatch: original_df ({len(original_df)}) vs labels ({len(labels)}). Using clustering data instead.")
                original_df = df.copy()
        else:
            original_df = df.copy()
            logging.info("No original data provided, using clustering data")
        
        cluster_analysis = {}
        unique_clusters = sorted(df['cluster'].unique())
        
        # Lấy danh sách feature names từ original data (bỏ qua cột cluster)
        analysis_feature_names = [col for col in original_df.columns if col != 'cluster']
        logging.info(f"Analysis feature names: {analysis_feature_names[:5]}... (total: {len(analysis_feature_names)})")
        
        for cluster_id in unique_clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            original_cluster_data = original_df[original_df['cluster'] == cluster_id]
            
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(df)) * 100
            
            # Tính trung bình từ dữ liệu đã chuẩn hóa (để xếp hạng)
            cluster_means = cluster_data[feature_names].mean()
            overall_means = df[feature_names].mean()
            mean_differences = cluster_means - overall_means
            
            # Tính trung bình từ dữ liệu gốc thực tế (để hiển thị)
            original_cluster_means = original_cluster_data[analysis_feature_names].mean()
            original_overall_means = original_df[analysis_feature_names].mean()
            
            # Tìm features đặc trưng (có độ lệch lớn nhất)
            distinctive_features = []
            
            # Sắp xếp theo độ lệch tuyệt đối từ normalized data
            sorted_features = mean_differences.abs().sort_values(ascending=False)
            
            # Lấy top features từ normalized data nhưng hiển thị bằng original data
            for i, norm_feature in enumerate(sorted_features.head(top_features).index):
                if i < len(analysis_feature_names):
                    # Lấy tên feature gốc tương ứng
                    original_feature_name = analysis_feature_names[i]
                    
                    # Lấy giá trị từ dữ liệu gốc
                    cluster_mean_original = original_cluster_means[original_feature_name]
                    overall_mean_original = original_overall_means[original_feature_name]
                    
                    # Tính difference từ normalized data để classify significance
                    difference_normalized = mean_differences[norm_feature]
                    
                    # Phân loại mức độ đặc trưng dựa trên normalized difference
                    if abs(difference_normalized) > 2:
                        significance = "Rất cao"
                    elif abs(difference_normalized) > 1:
                        significance = "Cao"
                    elif abs(difference_normalized) > 0.5:
                        significance = "Trung bình"
                    else:
                        significance = "Thấp"
                    
                    # Xác định xu hướng
                    if difference_normalized > 0:
                        trend = "cao hơn"
                        trend_icon = "📈"
                    else:
                        trend = "thấp hơn"
                        trend_icon = "📉"
                    
                    # Tính số lượng phần tử có giá trị cao/thấp
                    threshold = overall_mean_original
                    if difference_normalized > 0:
                        count_above_threshold = (original_cluster_data[original_feature_name] > threshold).sum()
                        percentage_above = (count_above_threshold / cluster_size) * 100
                        description = f"{count_above_threshold}/{cluster_size} phần tử ({percentage_above:.1f}%) có {original_feature_name} > {threshold:.2f}"
                    else:
                        count_below_threshold = (original_cluster_data[original_feature_name] < threshold).sum()
                        percentage_below = (count_below_threshold / cluster_size) * 100
                        description = f"{count_below_threshold}/{cluster_size} phần tử ({percentage_below:.1f}%) có {original_feature_name} < {threshold:.2f}"
                    
                    # Tính percentile
                    percentile = (original_cluster_data[original_feature_name] > overall_mean_original).mean() * 100
                    
                    distinctive_features.append({
                        'feature': original_feature_name,  # Hiển thị tên feature gốc
                        'cluster_mean': cluster_mean_original,  # Giá trị gốc
                        'overall_mean': overall_mean_original,  # Giá trị gốc
                        'difference': difference_normalized,  # Normalized difference để sort
                        'abs_difference': abs(difference_normalized),
                        'trend': trend,
                        'trend_icon': trend_icon,
                        'significance': significance,
                        'percentile': percentile,
                        'description': description
                    })
            
            # Tìm features có phương sai thấp nhất trong cụm (ổn định)
            cluster_stds = cluster_data[feature_names].std()
            most_stable_features = cluster_stds.sort_values().head(5)
            
            stable_features = []
            for j, feature in enumerate(most_stable_features.index):
                if j < len(analysis_feature_names):
                    original_feature_name = analysis_feature_names[j]
                    stable_features.append({
                        'feature': original_feature_name,
                        'std': cluster_stds[feature],
                        'mean': original_cluster_means[original_feature_name],
                        'description': f"{original_feature_name} rất ổn định (std: {cluster_stds[feature]:.3f})"
                    })
            
            # Tạo mô tả tổng quan cho cụm
            top_3_features = distinctive_features[:3]
            if top_3_features:
                summary_parts = []
                for feat in top_3_features:
                    summary_parts.append(f"{feat['feature']} {feat['trend']} trung bình")
                summary = f"Cụm đặc trưng bởi: " + ", ".join(summary_parts)
            else:
                summary = "Cụm có đặc điểm trung bình, không có features nổi bật đặc biệt"
            
            cluster_analysis[cluster_id] = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': cluster_percentage,
                'summary': summary,
                'distinctive_features': distinctive_features,
                'stable_features': stable_features,
                'cluster_means': original_cluster_means.to_dict(),
                'statistics': {
                    'min_values': original_cluster_data[analysis_feature_names].min().to_dict(),
                    'max_values': original_cluster_data[analysis_feature_names].max().to_dict(),
                    'std_values': original_cluster_data[analysis_feature_names].std().to_dict()
                }
            }
        
        logging.info(f"Successfully analyzed {len(cluster_analysis)} clusters")
        return cluster_analysis
    
    except Exception as e:
        logging.error(f"Error in analyze_cluster_characteristics: {str(e)}")
        logging.error(f"Input details: X shape = {X.shape if hasattr(X, 'shape') else 'Unknown'}, "
                     f"labels count = {len(labels) if labels is not None else 'Unknown'}, "
                     f"feature_names = {feature_names[:5]}...(total: {len(feature_names)}), "
                     f"original_data = {original_data.shape if hasattr(original_data, 'shape') else 'None'}, "
                     f"top_features = {top_features}")
        return {}

def get_cluster_predictions(X, model_name, k, random_state=42):
    """
    Thực hiện phân cụm với số cụm k cho mô hình cụ thể
    
    Args:
        X: Dữ liệu để phân cụm
        model_name: Tên mô hình ('KMeans', 'GMM', 'Hierarchical', 'FuzzyCMeans')
        k: Số cụm
        random_state: Seed cho reproducibility
    
    Returns:
        numpy array: Nhãn cụm cho từng điểm dữ liệu
    """
    try:
        if model_name == 'KMeans':
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(X)
            
        elif model_name == 'GMM':
            model = GaussianMixture(n_components=k, random_state=random_state)
            labels = model.fit_predict(X)
            
        elif model_name == 'Hierarchical':
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            
        elif model_name == 'FuzzyCMeans':
            # Fuzzy C-means cần được import riêng
            try:
                import skfuzzy as fuzz
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    X.T, k, 2, error=0.005, maxiter=1000, init=None
                )
                labels = np.argmax(u, axis=0)
            except ImportError:
                logging.warning("skfuzzy not available, using KMeans instead")
                model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = model.fit_predict(X)
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return labels
        
    except Exception as e:
        logging.error(f"Error in get_cluster_predictions: {str(e)}")
        # Fallback to KMeans
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        return model.fit_predict(X)

def create_cluster_comparison_table(cluster_analysis, feature_names):
    """
    Tạo bảng so sánh đặc trưng giữa các cụm
    
    Args:
        cluster_analysis: Kết quả phân tích từ analyze_cluster_characteristics
        feature_names: Danh sách tên features
    
    Returns:
        pandas.DataFrame: Bảng so sánh
    """
    try:
        comparison_data = []
        
        for feature in feature_names:
            row = {'Feature': feature}
            
            # Kiểm tra xem feature có tồn tại trong cluster_means không
            cluster_values = {}
            for cluster_id in sorted(cluster_analysis.keys()):
                if feature in cluster_analysis[cluster_id]['cluster_means']:
                    cluster_mean = cluster_analysis[cluster_id]['cluster_means'][feature]
                    row[f'Cụm {cluster_id}'] = f"{cluster_mean:.2f}"
                    cluster_values[cluster_id] = cluster_mean
                else:
                    logging.warning(f"Feature {feature} not found in cluster {cluster_id} means")
            
            # Tìm cụm có giá trị cao nhất và thấp nhất nếu có dữ liệu
            if cluster_values:
                max_cluster = max(cluster_values, key=cluster_values.get)
                min_cluster = min(cluster_values, key=cluster_values.get)
                
                row['Cao nhất'] = f"Cụm {max_cluster}"
                row['Thấp nhất'] = f"Cụm {min_cluster}"
                row['Chênh lệch'] = f"{cluster_values[max_cluster] - cluster_values[min_cluster]:.2f}"
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
        
    except Exception as e:
        logging.error(f"Error in create_cluster_comparison_table: {str(e)}")
        return pd.DataFrame()
