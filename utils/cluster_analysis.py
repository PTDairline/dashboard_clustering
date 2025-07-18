import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import logging

def analyze_cluster_characteristics(X, labels, feature_names, original_data=None, top_features=10):
    """
    Phân tích đặc trưng của từng cụm
    
    Args:
        X: Dữ liệu đã chuẩn hóa (numpy array hoặc DataFrame)
        labels: Nhãn cụm cho từng điểm dữ liệu
        feature_names: Tên các features
        original_data: Dữ liệu gốc chưa chuẩn hóa (để hiển thị giá trị thực)
        top_features: Số lượng features đặc trưng nhất để hiển thị
    
    Returns:
        dict: Thông tin đặc trưng của từng cụm
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        logging.debug(f"analyze_cluster_characteristics - Input shapes: X={X.shape if hasattr(X, 'shape') else 'unknown'}, "
                      f"labels={len(labels) if labels is not None else 'None'}, feature_names={len(feature_names)}") 
                 
        # Kiểm tra NaN
        if isinstance(X, np.ndarray) and np.isnan(X).any():
            logging.warning("NaN values found in X, replacing with zeros")
            X = np.nan_to_num(X, nan=0.0)
          # Chuyển đổi về DataFrame nếu cần
        if isinstance(X, np.ndarray):
            # Đảm bảo số lượng feature_names trùng khớp với số cột của X
            if X.shape[1] != len(feature_names):
                logging.warning(f"Feature names count ({len(feature_names)}) doesn't match X columns ({X.shape[1]}). Using generic names.")
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()
            
        # Kiểm tra feature_names cho PCA components
        pc_pattern = re.compile(r'^PC\d+$')
        is_using_pca = all(pc_pattern.match(f) for f in feature_names if isinstance(f, str))
        if is_using_pca:
            logging.info("Detected PCA components in feature_names, will use original_data for analysis")
            
        # Thêm cột cluster
        df['cluster'] = labels
          # Sử dụng dữ liệu gốc nếu có
        if original_data is not None:
            if isinstance(original_data, np.ndarray):
                # Đảm bảo số lượng feature_names trùng khớp với số cột của original_data
                if original_data.shape[1] != len(feature_names):
                    logging.warning(f"Feature names count ({len(feature_names)}) doesn't match original_data columns ({original_data.shape[1]}). Using generic names.")
                    original_feature_names = [f'Feature_{i}' for i in range(original_data.shape[1])]
                    original_df = pd.DataFrame(original_data, columns=original_feature_names)
                else:
                    original_df = pd.DataFrame(original_data, columns=feature_names)
            else:
                original_df = original_data.copy()
                
            # Kiểm tra NaN
            if original_df.isna().any().any():
                logging.warning("NaN values found in original_data, filling with column means")
                original_df = original_df.fillna(original_df.mean())
                
            # Thêm cột cluster
            if len(original_df) == len(labels):
                original_df['cluster'] = labels
            else:
                logging.warning(f"Length mismatch: original_df ({len(original_df)}) vs labels ({len(labels)}). Using df instead.")
                original_df = df.copy()
        else:
            original_df = df.copy()
        
        cluster_analysis = {}
        unique_clusters = sorted(df['cluster'].unique())
        
        for cluster_id in unique_clusters:
            cluster_data = df[df['cluster'] == cluster_id]
            original_cluster_data = original_df[original_df['cluster'] == cluster_id]
            
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(df)) * 100
            
            # Tính trung bình của cụm và tổng thể
            cluster_means = cluster_data[feature_names].mean()
            overall_means = df[feature_names].mean()
            original_cluster_means = original_cluster_data[feature_names].mean()
            
            # Tính độ lệch so với trung bình tổng thể
            mean_differences = cluster_means - overall_means
            
            # Tìm features đặc trưng (có độ lệch lớn nhất)
            distinctive_features = []
            
            # Sắp xếp theo độ lệch tuyệt đối
            sorted_features = mean_differences.abs().sort_values(ascending=False)
            
            for feature in sorted_features.head(top_features).index:
                difference = mean_differences[feature]
                cluster_mean_original = original_cluster_means[feature]
                overall_mean_original = df[feature_names].mean()[feature] if original_data is None else original_data[feature_names].mean()[feature]
                
                # Tính percentile để hiểu vị trí của cụm
                percentile = (original_cluster_data[feature] > overall_mean_original).mean() * 100
                
                # Phân loại mức độ đặc trưng
                if abs(difference) > 2:  # Rất đặc trưng
                    significance = "Rất cao"
                elif abs(difference) > 1:  # Khá đặc trưng
                    significance = "Cao"
                elif abs(difference) > 0.5:  # Trung bình
                    significance = "Trung bình"
                else:
                    significance = "Thấp"
                
                # Xác định xu hướng
                if difference > 0:
                    trend = "cao hơn"
                    trend_icon = "📈"
                else:
                    trend = "thấp hơn"
                    trend_icon = "📉"
                
                # Tính số lượng phần tử có giá trị cao/thấp
                threshold = overall_mean_original
                if difference > 0:
                    count_above_threshold = (original_cluster_data[feature] > threshold).sum()
                    percentage_above = (count_above_threshold / cluster_size) * 100
                    description = f"{count_above_threshold}/{cluster_size} phần tử ({percentage_above:.1f}%) có {feature} > {threshold:.2f}"
                else:
                    count_below_threshold = (original_cluster_data[feature] < threshold).sum()
                    percentage_below = (count_below_threshold / cluster_size) * 100
                    description = f"{count_below_threshold}/{cluster_size} phần tử ({percentage_below:.1f}%) có {feature} < {threshold:.2f}"
                
                distinctive_features.append({
                    'feature': feature,
                    'cluster_mean': cluster_mean_original,
                    'overall_mean': overall_mean_original,
                    'difference': difference,
                    'abs_difference': abs(difference),
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
            for feature in most_stable_features.index:
                stable_features.append({
                    'feature': feature,
                    'std': cluster_stds[feature],
                    'mean': original_cluster_means[feature],
                    'description': f"{feature} rất ổn định (std: {cluster_stds[feature]:.3f})"
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
                    'min_values': original_cluster_data[feature_names].min().to_dict(),
                    'max_values': original_cluster_data[feature_names].max().to_dict(),
                    'std_values': original_cluster_data[feature_names].std().to_dict()
                }
            }
        
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
            
            for cluster_id in sorted(cluster_analysis.keys()):
                cluster_mean = cluster_analysis[cluster_id]['cluster_means'][feature]
                row[f'Cụm {cluster_id}'] = f"{cluster_mean:.2f}"
            
            # Tìm cụm có giá trị cao nhất và thấp nhất
            cluster_values = {}
            for cluster_id in cluster_analysis.keys():
                cluster_values[cluster_id] = cluster_analysis[cluster_id]['cluster_means'][feature]
            
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