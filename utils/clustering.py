import matplotlib
matplotlib.use('Agg')  # Comment: Sử dụng backend 'Agg' cho matplotlib để vẽ biểu đồ trong môi trường không có GUI (phù hợp với Flask).
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import skfuzzy as fuzz
import logging
import concurrent.futures  # Thêm concurrent.futures để hỗ trợ đa luồng
from utils.metrics import starczewski_index, wiroonsri_index

# Comment: Import các thư viện và module cần thiết.
# - matplotlib: Dùng để vẽ biểu đồ (Silhouette, Elbow, Scatter).
# - pandas, numpy: Dùng cho xử lý dữ liệu.
# - sklearn.cluster: Các mô hình phân cụm (KMeans, AgglomerativeClustering, GaussianMixture).
# - sklearn.metrics: Các chỉ số đánh giá cụm (Silhouette, Calinski-Harabasz, Davies-Bouldin).
# - starczewski_index, wiroonsri_index: Chỉ số CVI tùy chỉnh từ module utils.metrics.
# - skfuzzy: Dùng cho thuật toán Fuzzy C-means.
# - logging: Dùng để ghi log.
# - sklearn.decomposition.PCA: Dùng để giảm chiều dữ liệu khi vẽ Scatter.

# Comment: Thiết lập logging để ghi lại thông tin debug.
# - Định dạng log: thời gian, mức độ, thông điệp.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        # Tối ưu hiệu suất bằng cách giảm logging và sử dụng numpy
        # Kiểm tra độ dài của các tham số đầu vào
        if len(cvi_values) != len(k_range) or len(alpha) != len(k_range):
            raise ValueError(f"Length mismatch: cvi_values ({len(cvi_values)}), k_range ({len(k_range)}), alpha ({len(alpha)})")
        
        # Kiểm tra tham số `alpha` không âm
        if any(a < 0 for a in alpha):
            raise ValueError("Tất cả alpha_k phải không âm")
        
        # Chuyển dữ liệu sang numpy để tính toán nhanh hơn
        cvi_array = np.array(cvi_values, dtype=float)
        alpha_array = np.array(alpha, dtype=float)
        
        # Thay các giá trị không hợp lệ bằng 0
        cvi_array = np.nan_to_num(cvi_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Tính r_k bằng vectorized operations (nhanh hơn loops)
        rk = np.zeros_like(cvi_array)
        if opt_type == 'min':
            max_cvi = np.max(cvi_array)
            differences = max_cvi - cvi_array
            sum_diff = np.sum(differences)
            if sum_diff > 0:
                rk = differences / sum_diff
            else:
                rk.fill(1.0 / len(k_range))
        else:  # opt_type == 'max'
            min_cvi = np.min(cvi_array)
            differences = cvi_array - min_cvi
            sum_diff = np.sum(differences)
            if sum_diff > 0:
                rk = differences / sum_diff
            else:
                rk.fill(1.0 / len(k_range))
        
        # Kiểm tra tổng r_k ≈ 1
        if not np.isclose(np.sum(rk), 1.0):
            rk = rk / np.sum(rk)  # Chuẩn hóa nếu tổng không bằng 1
        
        # Tính alpha_0 và BCVI
        alpha_0 = np.sum(alpha_array)
        bcvi = (alpha_array + n * rk) / (alpha_0 + n)
        
        # Chỉ log kết quả cuối cùng để tối ưu hiệu suất
        logging.debug(f"BCVI calculation completed with {len(bcvi)} values")
        
        return bcvi
    except Exception as e:
        # Comment: Xử lý lỗi nếu tính BCVI thất bại.
        logging.error(f"Lỗi tính BCVI: {str(e)}")
        raise Exception(f"Error computing BCVI: {str(e)}")

def generate_clustering_plots(X, model_name, k_range, selected_k, use_pca, selected_features, explained_variance_ratio):
    try:
        # Comment: Ghi log bắt đầu hàm `generate_clustering_plots`.
        # - `X`: Dữ liệu đầu vào.
        # - `model_name`: Tên mô hình (KMeans, GMM, Hierarchical, FuzzyCMeans).
        # - `k_range`: Phạm vi số cụm để thử.
        # - `selected_k`: Số cụm tối đa được chọn.
        # - `use_pca`: Trạng thái sử dụng PCA.
        # - `selected_features`: Danh sách feature đã chọn.
        # - `explained_variance_ratio`: Tỷ lệ phương sai giải thích (nếu dùng PCA).
        logging.debug(f"Bắt đầu generate_clustering_plots: model={model_name}, k_range={k_range}, selected_k={selected_k}")
        
        # Comment: Kiểm tra dữ liệu đầu vào có rỗng hoặc ít hơn 2 cột không.
        if X.empty or len(X.columns) < 2:
            logging.error("Dữ liệu không đủ để chạy phân cụm")
            return {'error': 'Dữ liệu không đủ để chạy phân cụm (cần ít nhất 2 cột số và không rỗng).'}
        
        # Comment: Kiểm tra dữ liệu có giá trị NaN không.
        if X.isna().any().any():
            logging.error("Dữ liệu chứa giá trị NaN")
            return {'error': 'Dữ liệu chứa giá trị NaN. Vui lòng xử lý dữ liệu trước khi chạy phân cụm.'}
        
        # Comment: Lọc các cột số từ dữ liệu.
        # - Đảm bảo chỉ sử dụng các cột số để chạy phân cụm.
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty or len(X_numeric.columns) < 2:
            logging.error("Dữ liệu không chứa đủ cột số")
            return {'error': 'Dữ liệu không chứa đủ cột số (cần ít nhất 2 cột số).'}
        
        # Comment: Kiểm tra dữ liệu có giá trị vô cực (inf) không.
        if np.isinf(X_numeric.values).any():
            logging.error("Dữ liệu chứa giá trị vô cực")
            return {'error': 'Dữ liệu chứa giá trị vô cực (inf). Vui lòng xử lý dữ liệu trước khi chạy phân cụm.'}
        
        # Comment: Chuyển đổi dữ liệu thành mảng numpy với kiểu float.
        X_array = X_numeric.values.astype(float)
        
        # Comment: Kiểm tra kích thước dữ liệu.
        # - Đảm bảo có ít nhất 2 mẫu và 2 feature để chạy phân cụm.
        n_samples, n_features = X_array.shape
        logging.debug(f"Kích thước dữ liệu: samples={n_samples}, features={n_features}")
        if n_samples < 2:
            logging.error("Số mẫu quá nhỏ")
            return {'error': 'Số lượng mẫu quá nhỏ (ít nhất 2 mẫu để phân cụm).'}
        if n_features < 2:
            logging.error("Số feature quá nhỏ")
            return {'error': 'Số lượng feature quá nhỏ (ít nhất 2 feature để phân cụm).'}
        
        # Comment: Kiểm tra tính hợp lệ của `k_range`.
        # - Đảm bảo `k_range` không rỗng và các giá trị `k` nằm trong khoảng hợp lệ.
        if not k_range:
            logging.error("k_range rỗng")
            return {'error': 'Phạm vi số cụm (k_range) rỗng. Vui lòng chọn số cụm hợp lệ.'}
        
        for k in k_range:
            if k < 2:
                logging.error(f"Số cụm k={k} không hợp lệ")
                return {'error': f'Số cụm k={k} không hợp lệ. Số cụm phải lớn hơn hoặc bằng 2.'}
            if k >= n_samples:
                logging.error(f"Số cụm k={k} lớn hơn số mẫu {n_samples}")
                return {'error': f'Số cụm k={k} không hợp lệ. Số cụm phải nhỏ hơn số mẫu ({n_samples}).'}
        
        # Comment: Khởi tạo dictionary `plots` để lưu kết quả phân cụm và biểu đồ.
        # - `silhouette`: Điểm Silhouette và biểu đồ.
        # - `elbow`: Điểm WCSS (Within-Cluster Sum of Squares) và biểu đồ Elbow.
        # - `scatter`: Biểu đồ phân cụm (Scatter).
        # - `cvi`: Các chỉ số CVI (Silhouette, Calinski-Harabasz, v.v.).
        plots = {
            'silhouette': {'scores': [], 'plot': None},
            'elbow': {'inertias': [], 'plot': None},
            'scatter': [],
            'cvi': []
        }
        
        # Comment: Khởi tạo các list để lưu điểm Silhouette, WCSS, và chỉ số CVI.
        silhouette_scores = []
        inertias = []
        cvi_scores = []
        
        # Comment: Giảm chiều dữ liệu để vẽ Scatter nếu số chiều lớn hơn 2.
        # - Sử dụng PCA để giảm xuống 2 chiều (cho biểu đồ 2D).
        X_plot = X_array
        if n_features > 2:
            logging.debug("Giảm chiều dữ liệu để vẽ scatter")
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X_array)
        
        # Comment: Lặp qua từng giá trị `k` trong `k_range` để chạy phân cụm và tính toán các chỉ số.
        for k in k_range:
            logging.debug(f"Chạy phân cụm với k={k}, model={model_name}")
              # Tối ưu hóa: Sử dụng tham số được tối ưu cho tốc độ
            if model_name == 'KMeans':
                model = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42, algorithm='lloyd')
                labels = model.fit_predict(X_array)                
                centroids = model.cluster_centers_
                membership = None
            elif model_name == 'GMM':
                model = GaussianMixture(n_components=k, n_init=3, max_iter=50, random_state=42, 
                                      covariance_type='full', warm_start=True)
                labels = model.fit_predict(X_array)
                centroids = model.means_
                membership = model.predict_proba(X_array)
            elif model_name == 'Hierarchical':
                # Tối ưu hóa: Sử dụng linkage tốt nhất cho tốc độ
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = model.fit_predict(X_array)
                # Tối ưu hóa: Tính centroids nhanh hơn
                centroids = np.array([X_array[labels == i].mean(axis=0) for i in range(k)])
                membership = None
            elif model_name == 'FuzzyCMeans':
                try:
                    # Tối ưu hóa: Giảm số iterations và error tolerance
                    logging.info(f"Running Fuzzy C-Means with k={k}, shape of X.T: {X_array.T.shape}")
                    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        X_array.T, k, m=1.5, error=0.1, maxiter=150, init=None, seed=42
                    )
                    labels = np.argmax(u, axis=0)
                    centroids = cntr
                    membership = u.T
                    logging.debug(f"Fuzzy C-Means completed: fpc={fpc}")
                except Exception as e:
                    logging.error(f"Lỗi Fuzzy C-Means với k={k}: {str(e)}")
                    return {'error': f'Fuzzy C-Means failed for k={k}: {str(e)}. Kiểm tra dữ liệu đầu vào (kích thước: {X_array.shape}, kiểu dữ liệu: {X_array.dtype})'}
            else:
                logging.error(f"Mô hình {model_name} không hỗ trợ")
                return {'error': f'Mô hình {model_name} không được hỗ trợ.'}
            
            # Comment: Sử dụng toàn bộ dữ liệu để vẽ Scatter.
            sample_labels = labels
            
            # Comment: Kiểm tra kích thước của `X_plot` và `sample_labels` trước khi vẽ.
            # - Đảm bảo số lượng nhãn khớp với số lượng điểm để vẽ Scatter.
            logging.debug(f"Kích thước X_plot: {X_plot.shape}, Kích thước sample_labels: {len(sample_labels)}")
            if len(sample_labels) != X_plot.shape[0]:
                logging.error(f"Kích thước không khớp: sample_labels ({len(sample_labels)}) không bằng X_plot ({X_plot.shape[0]})")
                return {'error': f"Kích thước không khớp: sample_labels ({len(sample_labels)}) không bằng X_plot ({X_plot.shape[0]})"}
            
            # Comment: Tính các chỉ số đánh giá cụm (CVI) trên toàn bộ dữ liệu.
            # - Nếu chỉ có 1 cụm duy nhất, đặt các chỉ số về giá trị mặc định.
            logging.debug(f"Tính chỉ số CVI cho k={k}")
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X_array, labels)
                calinski = calinski_harabasz_score(X_array, labels)
                davies = davies_bouldin_score(X_array, labels)
            else:
                logging.warning(f"Chỉ có 1 cụm duy nhất cho k={k}, đặt chỉ số CVI mặc định")
                silhouette = 0
                calinski = 0
                davies = float('inf')
            
            # Comment: Tính các chỉ số CVI tùy chỉnh (Starczewski, Wiroonsri).
            starczewski = starczewski_index(X_array, labels, centroids)
            wiroonsri = wiroonsri_index(X_array, labels, centroids)
            
            # Comment: Lưu các chỉ số CVI vào danh sách `cvi_scores`.
            cvi_scores.append({
                'k': k,
                'Silhouette': silhouette,
                'Calinski-Harabasz': calinski,
                'Davies-Bouldin': davies,
                'Starczewski': starczewski,
                'Wiroonsri': wiroonsri
            })
            
            silhouette_scores.append(silhouette)
            
            # Comment: Tính WCSS (inertia) cho KMeans và Fuzzy C-means.
            # - KMeans: Sử dụng `model.inertia_`.
            # - Fuzzy C-means: Tính thủ công dựa trên ma trận thành viên `u`.
            if model_name in ['KMeans', 'FuzzyCMeans']:
                if model_name == 'KMeans':
                    inertias.append(model.inertia_)
                else:
                    logging.debug(f"Tính inertia cho Fuzzy C-Means với k={k}")
                    distances = np.linalg.norm(X_array[:, np.newaxis] - centroids, axis=2) ** 2  # Shape: (n, k)
                    inertia = np.sum((u.T ** 2) * distances)
                    inertias.append(inertia)
              # Tối ưu hóa: Tạo biểu đồ Scatter với kích thước và DPI tối ưu
            logging.debug(f"Tạo biểu đồ scatter cho k={k}")
            plt.figure(figsize=(5, 3))  # Giảm kích thước để tăng tốc
            scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=sample_labels, cmap='viridis', s=15, alpha=0.8)
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title(f'{model_name} (k={k})')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Giảm DPI để tăng tốc
            buf.seek(0)
            scatter_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close('all')
            plots['scatter'].append({'k': k, 'plot': scatter_plot})
          # Tối ưu hóa: Tạo biểu đồ Silhouette với kích thước tối ưu
        logging.debug("Tạo biểu đồ Silhouette")
        plt.figure(figsize=(7, 4))
        plt.plot(list(k_range), silhouette_scores, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Số cụm (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score theo số cụm')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        plots['silhouette']['scores'] = silhouette_scores
        plots['silhouette']['plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close('all')
        
        # Tối ưu hóa: Tạo biểu đồ Elbow với hiệu suất cao hơn
        if model_name in ['KMeans', 'FuzzyCMeans']:
            logging.debug("Tạo biểu đồ Elbow")
            plt.figure(figsize=(7, 4))
            plt.plot(list(k_range), inertias, marker='o', linewidth=2, markersize=6)
            plt.xlabel('Số cụm (k)')
            plt.ylabel('WCSS')
            plt.title('Elbow Method')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            plots['elbow']['inertias'] = inertias
            plots['elbow']['plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close('all')
        else:
            # Comment: Nếu không phải KMeans hoặc Fuzzy C-means, đặt giá trị mặc định.
            plots['elbow']['inertias'] = [0] * len(k_range)
            plots['elbow']['plot'] = None
        
        # Comment: Lưu các chỉ số CVI vào `plots` và ghi log hoàn thành.
        plots['cvi'] = cvi_scores
        logging.debug("Hoàn thành generate_clustering_plots")
        
        return plots
    except Exception as e:
        # Comment: Xử lý lỗi nếu quá trình phân cụm hoặc tạo biểu đồ thất bại.
        logging.error(f"Lỗi trong generate_clustering_plots: {str(e)}")
        return {'error': f'Lỗi khi chạy phân cụm: {str(e)}'}

def run_cluster_for_k(X_array, X_plot, k, model_name):
    """
    Chạy mô hình phân cụm với số cụm k và trả về kết quả
    
    Args:
        X_array (ndarray): Dữ liệu đầu vào dạng mảng numpy
        X_plot (ndarray): Dữ liệu chiếu 2D để vẽ biểu đồ scatter
        k (int): Số cụm
        model_name (str): Tên mô hình ('KMeans', 'GMM', 'Hierarchical', 'FuzzyCMeans')
    
    Returns:
        dict: Kết quả phân cụm bao gồm labels, centroids, membership và các chỉ số
    """
    try:
        logging.debug(f"Chạy phân cụm với k={k}, model={model_name}")
        
        # Khởi tạo và chạy mô hình tương ứng
        if model_name == 'KMeans':
            model = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42, algorithm='lloyd')
            labels = model.fit_predict(X_array)
            centroids = model.cluster_centers_
            membership = None
            inertia = model.inertia_
        elif model_name == 'GMM':
            model = GaussianMixture(n_components=k, n_init=3, max_iter=50, random_state=42,
                                  covariance_type='full', warm_start=True)
            labels = model.fit_predict(X_array)
            centroids = model.means_
            membership = model.predict_proba(X_array)
            inertia = None
        elif model_name == 'Hierarchical':
            model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = model.fit_predict(X_array)
            centroids = np.array([X_array[labels == i].mean(axis=0) for i in range(k)])
            membership = None
            inertia = None
        elif model_name == 'FuzzyCMeans':
            try:
                logging.info(f"Running Fuzzy C-Means with k={k}, shape of X.T: {X_array.T.shape}")
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    X_array.T, k, m=1.5, error=0.1, maxiter=150, init=None, seed=42
                )
                labels = np.argmax(u, axis=0)
                centroids = cntr
                membership = u.T
                # Tính inertia cho FCM
                distances = np.linalg.norm(X_array[:, np.newaxis] - centroids, axis=2) ** 2
                inertia = np.sum((u.T ** 2) * distances)
                logging.debug(f"Fuzzy C-Means completed: fpc={fpc}")
            except Exception as e:
                logging.error(f"Lỗi Fuzzy C-Means với k={k}: {str(e)}")
                return {'error': f'Fuzzy C-Means failed for k={k}: {str(e)}'}
        else:
            logging.error(f"Mô hình {model_name} không hỗ trợ")
            return {'error': f'Mô hình {model_name} không được hỗ trợ.'}
        
        # Tính các chỉ số đánh giá cụm (CVI)
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X_array, labels)
            calinski = calinski_harabasz_score(X_array, labels)
            davies = davies_bouldin_score(X_array, labels)
        else:
            silhouette = 0
            calinski = 0
            davies = float('inf')
        
        # Tính các chỉ số CVI tùy chỉnh
        starczewski = starczewski_index(X_array, labels, centroids)
        wiroonsri = wiroonsri_index(X_array, labels, centroids)
        
        # Tạo biểu đồ scatter
        plt.figure(figsize=(5, 3))
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap='viridis', s=15, alpha=0.8)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{model_name} (k={k})')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        scatter_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close('all')
        
        result = {
            'k': k,
            'labels': labels,
            'centroids': centroids,
            'membership': membership,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies,
            'starczewski': starczewski,
            'wiroonsri': wiroonsri,
            'inertia': inertia,
            'scatter_plot': scatter_plot
        }
        
        return result
    
    except Exception as e:
        logging.error(f"Lỗi trong run_cluster_for_k với k={k}, model={model_name}: {str(e)}")
        return {'error': f'Lỗi khi chạy phân cụm với k={k}: {str(e)}'}