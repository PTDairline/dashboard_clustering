import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from utils.metrics import starczewski_index, wiroonsri_index
import skfuzzy as fuzz
import logging

def compute_bcvi(cvi_values, k_range, alpha, n, opt_type='max'):
    """
    Tính toán BCVI dựa trên phân phối Dirichlet tiên nghiệm.
    
    Parameters:
    - cvi_values: List các giá trị CVI cho k từ k_range[0] đến k_range[-1].
    - k_range: List các giá trị k (số cụm).
    - alpha: List các tham số tiên nghiệm Dirichlet (alpha_k).
    - n: Tham số điều chỉnh (cố định).
    - opt_type: 'max' nếu CVI tối ưu khi giá trị lớn nhất, 'min' nếu nhỏ nhất.
    
    Returns:
    - bcvi: List các giá trị BCVI(k).
    """
    try:
        # Kiểm tra độ dài đầu vào
        if len(cvi_values) != len(k_range) or len(alpha) != len(k_range):
            raise ValueError(f"Length mismatch: cvi_values ({len(cvi_values)}), k_range ({len(k_range)}), alpha ({len(alpha)})")
        
        if any(a < 0 for a in alpha):
            raise ValueError("Tất cả alpha_k phải không âm")
        
        # Kiểm tra và làm sạch cvi_values
        cleaned_cvi_values = []
        for value in cvi_values:
            if value is None or np.isinf(value) or np.isnan(value):
                cleaned_cvi_values.append(0.0)
            else:
                cleaned_cvi_values.append(float(value))
        logging.debug(f"Cleaned CVI values: {cleaned_cvi_values}")
        
        # Tính r_k
        rk = []
        if opt_type == 'min':
            max_cvi = max(cleaned_cvi_values)
            denominator = sum(max_cvi - cvi for cvi in cleaned_cvi_values)
            if denominator > 0:
                rk = [(max_cvi - cvi) / denominator for cvi in cleaned_cvi_values]
            else:
                rk = [1 / len(k_range)] * len(k_range)
        else:  # opt_type == 'max'
            min_cvi = min(cleaned_cvi_values)
            denominator = sum(cvi - min_cvi for cvi in cleaned_cvi_values)
            if denominator > 0:
                rk = [(cvi - min_cvi) / denominator for cvi in cleaned_cvi_values]
            else:
                rk = [1 / len(k_range)] * len(k_range)
        
        # Kiểm tra tổng r_k
        rk_sum = sum(rk)
        if not abs(rk_sum - 1.0) < 1e-6:
            raise ValueError(f"Tổng r_k không bằng 1: {rk_sum}")
        logging.debug(f"r_k values: {rk}")
        
        # Tính alpha_0
        alpha_0 = sum(alpha)
        
        # Tính BCVI(k)
        bcvi = [(alpha[k_idx] + n * rk[k_idx]) / (alpha_0 + n) for k_idx in range(len(k_range))]
        
        return bcvi
    except Exception as e:
        raise Exception(f"Error computing BCVI: {str(e)}")

def generate_clustering_plots(X, model_name, k_range, selected_k, use_pca, selected_features, explained_variance_ratio):
    try:
        # Kiểm tra dữ liệu đầu vào
        if X.empty or len(X.columns) < 2:
            return {'error': 'Dữ liệu không đủ để chạy phân cụm (cần ít nhất 2 cột số và không rỗng).'}
        
        # Kiểm tra giá trị NaN trong dữ liệu
        if X.isna().any().any():
            return {'error': 'Dữ liệu chứa giá trị NaN. Vui lòng xử lý dữ liệu trước khi chạy phân cụm.'}
        
        # Kiểm tra kiểu dữ liệu: Chỉ giữ các cột số
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty or len(X_numeric.columns) < 2:
            return {'error': 'Dữ liệu không chứa đủ cột số (cần ít nhất 2 cột số).'}
        
        # Kiểm tra giá trị vô cực (inf)
        if np.isinf(X_numeric.values).any():
            return {'error': 'Dữ liệu chứa giá trị vô cực (inf). Vui lòng xử lý dữ liệu trước khi chạy phân cụm.'}
        
        # Chuyển đổi dữ liệu thành numpy array với kiểu float
        X_array = X_numeric.values.astype(float)
        
        # Kiểm tra kích thước dữ liệu
        n_samples, n_features = X_array.shape
        if n_samples < 2:
            return {'error': 'Số lượng mẫu quá nhỏ (ít nhất 2 mẫu để phân cụm).'}
        if n_features < 2:
            return {'error': 'Số lượng feature quá nhỏ (ít nhất 2 feature để phân cụm).'}
        
        # Kiểm tra giá trị k_range
        if not k_range:
            return {'error': 'Phạm vi số cụm (k_range) rỗng. Vui lòng chọn số cụm hợp lệ.'}
        
        for k in k_range:
            if k < 2:
                return {'error': f'Số cụm k={k} không hợp lệ. Số cụm phải lớn hơn hoặc bằng 2.'}
            if k >= n_samples:
                return {'error': f'Số cụm k={k} không hợp lệ. Số cụm phải nhỏ hơn số mẫu ({n_samples}).'}
        
        # Chuẩn bị kết quả
        plots = {
            'silhouette': {'scores': [], 'plot': None},
            'elbow': {'inertias': [], 'plot': None},
            'scatter': [],
            'cvi': []
        }
        
        # Tính các chỉ số đánh giá cụm
        silhouette_scores = []
        inertias = []
        cvi_scores = []
        
        for k in k_range:
            # Khởi tạo mô hình phân cụm
            if model_name == 'KMeans':
                model = KMeans(n_clusters=k, n_init=1, max_iter=300, random_state=42)
                labels = model.fit_predict(X_array)
                centroids = model.cluster_centers_
                membership = None
            elif model_name == 'GMM':
                model = GaussianMixture(n_components=k, n_init=1, max_iter=100, random_state=42)
                labels = model.fit_predict(X_array)
                centroids = model.means_
                membership = model.predict_proba(X_array)
            elif model_name == 'Hierarchical':
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X_array)
                centroids = np.array([X_array[labels == i].mean(axis=0) for i in range(k)])
                membership = None
            elif model_name == 'FuzzyCMeans':
                # Sử dụng Fuzzy C-Means từ skfuzzy
                try:
                    logging.info(f"Running Fuzzy C-Means with k={k}, shape of X.T: {X_array.T.shape}")
                    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        X_array.T, k, m=2, error=0.01, maxiter=100, init=None, seed=42
                    )
                    labels = np.argmax(u, axis=0)
                    centroids = cntr
                    membership = u.T
                except Exception as e:
                    logging.error(f"Error in Fuzzy C-Means with k={k}: {str(e)}")
                    return {'error': f'Fuzzy C-Means failed for k={k}: {str(e)}. Kiểm tra dữ liệu đầu vào (kích thước: {X_array.shape}, kiểu dữ liệu: {X_array.dtype})'}
            else:
                return {'error': f'Mô hình {model_name} không được hỗ trợ.'}
            
            # Tính các chỉ số đánh giá cụm trên toàn bộ dữ liệu
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X_array, labels)
                calinski = calinski_harabasz_score(X_array, labels)
                davies = davies_bouldin_score(X_array, labels)
            else:
                silhouette = 0
                calinski = 0
                davies = float('inf')
            
            # Tính các chỉ số tùy chỉnh
            starczewski = starczewski_index(X_array, labels, centroids)
            wiroonsri = wiroonsri_index(X_array, labels, centroids)
            
            cvi_scores.append({
                'k': k,
                'Silhouette': silhouette,
                'Calinski-Harabasz': calinski,
                'Davies-Bouldin': davies,
                'Starczewski': starczewski,
                'Wiroonsri': wiroonsri
            })
            
            silhouette_scores.append(silhouette)
            if model_name in ['KMeans', 'FuzzyCMeans']:
                if model_name == 'KMeans':
                    inertias.append(model.inertia_)
                else:
                    # Tối ưu hóa tính inertia cho Fuzzy C-Means bằng vector hóa
                    distances = np.linalg.norm(X_array[:, np.newaxis] - centroids, axis=2) ** 2  # Shape: (n, k)
                    inertia = np.sum((u.T ** 2) * distances)
                    inertias.append(inertia)
            
            # Tạo biểu đồ scatter cho tất cả giá trị k
            plt.figure(figsize=(6, 4))
            plt.scatter(X_array[:, 0], X_array[:, 1], c=labels, cmap='viridis')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'{model_name} Clustering (k={k})')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=75)  # Giảm dpi để tăng tốc độ
            buf.seek(0)
            scatter_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close('all')
            plots['scatter'].append({'k': k, 'plot': scatter_plot})
        
        # Tạo biểu đồ Silhouette
        plt.figure(figsize=(6, 4))
        plt.plot(list(k_range), silhouette_scores, marker='o')
        plt.xlabel('Số cụm (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score theo số cụm')
        plt.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        plots['silhouette']['scores'] = silhouette_scores
        plots['silhouette']['plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close('all')
        
        # Tạo biểu đồ Elbow (chỉ có ý nghĩa với KMeans và Fuzzy C-Means)
        if model_name in ['KMeans', 'FuzzyCMeans']:
            plt.figure(figsize=(6, 4))
            plt.plot(list(k_range), inertias, marker='o')
            plt.xlabel('Số cụm (k)')
            plt.ylabel('WCSS')
            plt.title('Elbow Curve')
            plt.grid(True)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=75)
            buf.seek(0)
            plots['elbow']['inertias'] = inertias
            plots['elbow']['plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close('all')
        else:
            # Không tạo biểu đồ Elbow cho GMM và Hierarchical
            plots['elbow']['inertias'] = [0] * len(k_range)  # Điền giá trị 0 để tránh lỗi
            plots['elbow']['plot'] = None
        
        plots['cvi'] = cvi_scores
        
        return plots
    except Exception as e:
        logging.error(f"Error in generate_clustering_plots: {str(e)}")
        return {'error': f'Lỗi khi chạy phân cụm: {str(e)}'}