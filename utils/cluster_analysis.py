import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import logging

def analyze_cluster_characteristics(X, labels, feature_names, original_data=None, original_feature_names=None, top_features=10):
    """
    Phân tích đặc trưng của từng cụm.
    
    Args:
        X: Dữ liệu đã chuẩn hóa (numpy array hoặc DataFrame)
        labels: Nhãn cụm cho từng điểm dữ liệu
        feature_names: Tên các features của X (có thể là PC1, PC2... nếu dùng PCA)
        original_data: Dữ liệu gốc chưa chuẩn hóa (để hiển thị giá trị thực)
        original_feature_names: Tên các features gốc nếu dùng PCA
        top_features: Số lượng features đặc trưng nhất để hiển thị
    
    Returns:
        dict: Thông tin đặc trưng của từng cụm
    """
    try:
        logging.debug(f"analyze_cluster_characteristics - Input shapes: X={X.shape if hasattr(X, 'shape') else 'unknown'}, "
                      f"labels={len(labels) if labels is not None else 'None'}, "
                      f"feature_names={len(feature_names)}, "
                      f"original_feature_names={len(original_feature_names) if original_feature_names else 'None'}")

        # Kiểm tra NaN trong X
        if isinstance(X, np.ndarray) and np.isnan(X).any():
            logging.warning("NaN values found in X, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0)
        
        # Chuyển đổi về DataFrame nếu cần
        if isinstance(X, np.ndarray):
            if X.shape[1] != len(feature_names):
                logging.warning(f"Feature names count ({len(feature_names)}) doesn't match X columns ({X.shape[1]}). Using generic names.")
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
            
        # Thêm cột cluster
        df['cluster'] = labels

        # Sử dụng dữ liệu gốc nếu có
        if original_data is not None:
            if isinstance(original_data, np.ndarray):
                if original_feature_names is None:
                    original_feature_names = feature_names[:original_data.shape[1]]
                original_df = pd.DataFrame(original_data, columns=original_feature_names)
            else:
                original_df = original_data.copy()
                
            # Kiểm tra và xử lý NaN trong original_data
            if original_df.isna().any().any():
                nan_counts = original_df.isna().sum()
                for col, count in nan_counts.items():
                    if count > 0:
                        logging.warning(f"Column '{col}' has {count} NaN values")
                original_df = original_df.fillna(original_df.median(numeric_only=True))
                if original_df.isna().any().any():
                    nan_count = original_df.isna().sum().sum()
                    logging.warning(f"Found {nan_count} NaN values in original_data after fillna, dropping rows")
                    original_df = original_df.dropna()
                
            if len(original_df) == len(labels):
                original_df['cluster'] = labels
            else:
                logging.warning(f"Length mismatch: original_df ({len(original_df)}) vs labels ({len(labels)}). Using df instead.")
                original_df = df.copy()
        else:
            original_df = df.copy()
            original_feature_names = feature_names

        cluster_analysis = {}
        unique_clusters = sorted(df['cluster'].unique())
        analysis_feature_names = original_feature_names if original_feature_names else feature_names

        for cluster_id in unique_clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            original_cluster_data = original_df[original_df['cluster'] == cluster_id]
            
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(df)) * 100
            cluster_means = cluster_data[feature_names].mean()
            overall_means = df[feature_names].mean()
            
            original_cluster_means = original_cluster_data[analysis_feature_names].mean()
            original_overall_means = original_df[analysis_feature_names].mean()
            
            distinctive_features = []
            
            for feature in analysis_feature_names:
                try:
                    if feature not in original_cluster_data.columns or feature not in original_df.columns:
                        logging.warning(f"Feature '{feature}' not found in original_data, skipping")
                        continue
                    
                    cluster_mean_original = original_cluster_means[feature]
                    overall_mean_original = original_overall_means[feature]
                    
                    if pd.isna(cluster_mean_original) or pd.isna(overall_mean_original):
                        logging.warning(f"Feature '{feature}' has NaN mean values, skipping")
                        continue
                    
                    # Sử dụng dữ liệu gốc để tính difference
                    difference = cluster_mean_original - overall_mean_original
                    percentile = (original_cluster_data[feature] > overall_mean_original).mean() * 100
                    
                    # Điều chỉnh ngưỡng cho dữ liệu gốc
                    if abs(difference) > 10:
                        significance = "Rất cao"
                    elif abs(difference) > 5:
                        significance = "Cao"
                    elif abs(difference) > 2:
                        significance = "Trung bình"
                    else:
                        significance = "Thấp"
                    
                    if difference > 0:
                        trend = "cao hơn"
                        trend_icon = "📈"
                    else:
                        trend = "thấp hơn"
                        trend_icon = "📉"
                    
                    threshold = overall_mean_original
                    if difference > 0:
                        count_above_threshold = (original_cluster_data[feature] > threshold).sum()
                        percentage_above = (count_above_threshold / cluster_size) * 100
                        description = f"{str(count_above_threshold)}/{str(cluster_size)} phần tử ({percentage_above:.1f}%) có {feature} > {threshold:.2f}"
                    else:
                        count_below_threshold = (original_cluster_data[feature] < threshold).sum()
                        percentage_below = (count_below_threshold / cluster_size) * 100
                        description = f"{str(count_below_threshold)}/{str(cluster_size)} phần tử ({percentage_below:.1f}%) có {feature} < {threshold:.2f}"
                    
                    distinctive_features.append({
                        'feature': feature,
                        'cluster_mean': float(cluster_mean_original),
                        'overall_mean': float(overall_mean_original),
                        'difference': float(difference),
                        'abs_difference': float(abs(difference)),
                        'trend': trend,
                        'trend_icon': trend_icon,
                        'significance': significance,
                        'percentile': float(percentile),
                        'description': description
                    })
                except Exception as e:
                    logging.error(f"Error processing feature '{feature}': {str(e)}")
                    continue
            
            # Sắp xếp và lấy top features
            distinctive_features.sort(key=lambda x: x['abs_difference'], reverse=True)
            distinctive_features = distinctive_features[:top_features]
            
            # Tìm features có phương sai thấp nhất dựa trên dữ liệu gốc
            if original_cluster_data is not None and analysis_feature_names:
                cluster_stds = original_cluster_data[analysis_feature_names].std()
                most_stable_features = cluster_stds.sort_values().head(5)
                
                stable_features = []
                for feature in most_stable_features.index:
                    try:
                        stable_features.append({
                            'feature': feature,
                            'std': float(cluster_stds[feature]),
                            'mean': float(original_cluster_means.get(feature, original_cluster_data[feature].mean())),
                            'description': f"{feature} rất ổn định (std: {cluster_stds[feature]:.3f})"
                        })
                    except Exception as e:
                        logging.error(f"Error processing stable feature '{feature}': {str(e)}")
                        continue
            
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
        
        if not cluster_analysis:
            logging.warning("No clusters analyzed successfully")
            return {}
        
        return cluster_analysis
    
    except Exception as e:
        logging.error(f"Error in analyze_cluster_characteristics: {str(e)}")
        logging.error(f"Input details: X shape = {X.shape if hasattr(X, 'shape') else 'Unknown'}, "
                     f"labels count = {len(labels) if labels is not None else 'Unknown'}, "
                     f"feature_names = {feature_names[:5]}...(total: {len(feature_names)}), "
                     f"original_data = {original_data.shape if hasattr(original_data, 'shape') else 'None'}, "
                     f"original_feature_names = {original_feature_names[:5] if original_feature_names else 'None'}...(total: {len(original_feature_names) if original_feature_names else 0}), "
                     f"top_features = {top_features}")
        return {}

def get_cluster_predictions(X, model_name, k, random_state=42):
    """
    Thực hiện phân cụm với số cụm k cho mô hình cụ thể.
    
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
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        return model.fit_predict(X)

def create_cluster_comparison_table(cluster_analysis, feature_names):
    """
    Tạo bảng so sánh đặc trưng giữa các cụm.
    
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
            
            for cluster_id in sorted(cluster_analysis.keys()):
                cluster_mean = cluster_analysis[cluster_id]['cluster_means'].get(feature, 0)
                row[f'Cụm {cluster_id}'] = f"{cluster_mean:.2f}"
            
            cluster_values = {}
            for cluster_id in cluster_analysis.keys():
                cluster_values[cluster_id] = cluster_analysis[cluster_id]['cluster_means'].get(feature, 0)
            
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