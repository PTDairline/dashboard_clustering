import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging
import concurrent.futures  # Thêm concurrent.futures cho đa luồng

# Comment: Import các thư viện cần thiết.
# - numpy: Dùng cho tính toán số học.
# - scipy.spatial.distance: Dùng để tính khoảng cách (pdist, squareform).
# - sklearn.metrics: Dùng để tính các chỉ số CVI (silhouette_score, davies_bouldin_score).
# - logging: Dùng để ghi log.

# Comment: Thiết lập logging để ghi lại thông tin debug.
# - Định dạng log: thời gian, mức độ, thông điệp.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calinski_harabasz_index(X, labels, centroids):
    """Tính toán chỉ số Calinski-Harabasz (CH) với tối đa 1000 điểm."""
    try:
        # Lấy mẫu 1000 điểm nếu dữ liệu lớn hơn
        n = len(X)
        if n > 1000:
            indices = np.random.choice(n, size=1000, replace=False)
            X = X[indices]
            labels = labels[indices]
        
        if len(np.unique(labels)) < 2:
            return 0.0
        
        # Tính CH index
        return calinski_harabasz_score(X, labels)
    except Exception:
        return 0.0

def silhouette_index(X, labels):
    """Tính toán chỉ số Silhouette (SH) với tối đa 1000 điểm."""
    try:
        # Lấy mẫu 1000 điểm nếu dữ liệu lớn hơn
        n = len(X)
        if n > 1000:
            indices = np.random.choice(n, size=1000, replace=False)
            X = X[indices]
            labels = labels[indices]
        
        if len(np.unique(labels)) < 2:
            return 0.0
        
        return silhouette_score(X, labels)
    except Exception:
        return 0.0

def davies_bouldin_index(X, labels, centroids):
    """Tính toán chỉ số Davies-Bouldin (DB) với tối đa 1000 điểm."""
    try:
        # Lấy mẫu 1000 điểm nếu dữ liệu lớn hơn
        n = len(X)
        if n > 1000:
            indices = np.random.choice(n, size=1000, replace=False)
            X = X[indices]
            labels = labels[indices]
        
        if len(np.unique(labels)) < 2:
            return float('inf')
        
        db = davies_bouldin_score(X, labels)
        return db if np.isfinite(db) else float('inf')
    except Exception:
        return float('inf')

def starczewski_index(X, labels, centroids):
    """Tính toán chỉ số Starczewski (STR) với tối đa 100 điểm."""
    try:
        # Lấy mẫu 100 điểm nếu dữ liệu lớn hơn
        n = len(X)
        if n > 100:
            indices = np.random.choice(n, size=100, replace=False)
            X = X[indices]
            labels = labels[indices]
        
        if len(np.unique(labels)) < 2:
            return 0.0
        
        k = len(np.unique(labels))
        
        # Tính D(k)
        centroid_dists = pdist(centroids)
        max_dist = np.max(centroid_dists)
        min_dist = np.min(centroid_dists)
        D_k = max_dist / min_dist if min_dist > 0 else 0.0
        
        # Tính E(k)
        v_0 = np.mean(X, axis=0)
        num = np.sum(np.linalg.norm(X - v_0, axis=1))
        den = 0.0
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                den += np.sum(np.linalg.norm(points - centroids[j], axis=1))
        
        E_k = num / den if den > 0 else 0.0
        return E_k * D_k
    except Exception:
        return 0.0

def wiroonsri_index(X, labels, centroids):
    """Tính toán chỉ số Wiroonsri (WI) với tối đa 100 điểm."""
    try:
        # Lấy mẫu 100 điểm nếu dữ liệu lớn hơn
        n = len(X)
        if n > 100:
            indices = np.random.choice(n, size=100, replace=False)
            X_sample = X[indices]
            labels_sample = labels[indices]
        else:
            X_sample = X
            labels_sample = labels
        
        if len(np.unique(labels)) < 2:
            return 0.0
        
        # Tính vector d và c
        d = pdist(X_sample)
        n_pairs = len(d)
        c = np.zeros(n_pairs)
        
        pair_idx = 0
        for i in range(len(X_sample)):
            for j in range(i + 1, len(X_sample)):
                c[pair_idx] = np.linalg.norm(centroids[labels_sample[i]] - centroids[labels_sample[j]])
                pair_idx += 1
        
        # Tính NC(k)
        std_d = np.std(d)
        std_c = np.std(c)
        
        if std_d == 0 or std_c == 0:
            nc_k = 0.0
        else:
            nc_k = np.corrcoef(d, c)[0, 1]
            if np.isnan(nc_k):
                nc_k = 0.0
        
        # Tính NC(1)
        if std_d == 0:
            nc_1 = 0.0
        else:
            d_range = np.max(d) - np.min(d)
            nc_1 = std_d / d_range if d_range > 0 else 0.0
        
        return max(nc_k, nc_1)
    except Exception:
        return 0.0

def suggest_optimal_k(plots, k_range, use_wiroonsri_starczewski=False, use_pca=True):
    """
    Gợi ý số k tối ưu dựa trên các chỉ số CVI và biểu đồ Elbow.
    
    Parameters:
    -----------
    plots : dict
        Dictionary chứa kết quả phân cụm (CVI, Elbow, v.v.)
    k_range : list
        Danh sách các giá trị k cần xét
    use_wiroonsri_starczewski : bool, optional
        Nếu True, sử dụng Wiroonsri và Starczewski để gợi ý số cụm tối ưu
        Nếu False, sử dụng Silhouette và Elbow để gợi ý số cụm tối ưu
    use_pca : bool, optional
        Nếu True, dùng PCA làm phương pháp giảm chiều
        Nếu False, dùng trực tiếp các features được chọn
        
    Returns:
    --------
    tuple : (optimal_k, reasoning)
        optimal_k : int
            Số cụm tối ưu
        reasoning : str
            Lý do gợi ý số cụm tối ưu
    """
    # Comment: Ghi log bắt đầu hàm gợi ý số cụm tối ưu.
    logging.debug(f"Bắt đầu suggest_optimal_k, use_wiroonsri_starczewski={use_wiroonsri_starczewski}, use_pca={use_pca}")
    
    # Comment: Khởi tạo biến để lưu kết quả.
    # - `optimal_k`: Số cụm tối ưu (mặc định là giá trị nhỏ nhất trong `k_range`).
    # - `reasoning`: Danh sách lý do cho từng gợi ý.
    optimal_k = k_range[0]  # Giá trị mặc định
    reasoning = []    # Tối ưu hóa: Lấy các chỉ số CVI từ `plots` với xử lý nhanh hơn
    cvi_data = plots['cvi']
    if not cvi_data:
        return k_range[0], "Không có dữ liệu CVI để gợi ý số cụm tối ưu."
    
    # Tối ưu hóa: Sử dụng numpy arrays để xử lý nhanh hơn
    silhouette_scores = np.array([entry['Silhouette'] for entry in cvi_data])
    calinski_scores = np.array([entry['Calinski-Harabasz'] for entry in cvi_data])
    davies_scores = np.array([entry['Davies-Bouldin'] for entry in cvi_data])
    starczewski_scores = np.array([entry['Starczewski'] for entry in cvi_data])
    wiroonsri_scores = np.array([entry['Wiroonsri'] for entry in cvi_data])    # Tối ưu hóa: Phương pháp Elbow với numpy operations
    wcss = plots['elbow']['inertias']
    elbow_k = k_range[0]  # Mặc định
    if wcss and len(wcss) > 2:  # Cần ít nhất 3 điểm để tính elbow
        wcss_array = np.array(wcss)
        if np.any(wcss_array):  # Kiểm tra xem WCSS có giá trị hợp lệ không
            wcss_diff = np.diff(wcss_array)  # Đạo hàm bậc nhất
            if len(wcss_diff) > 1:
                wcss_diff2 = np.diff(wcss_diff)  # Đạo hàm bậc hai
                if np.any(wcss_diff2 != 0):  # Kiểm tra wcss_diff2 không toàn 0
                    elbow_idx = np.argmax(wcss_diff2) + 2
                    if elbow_idx < len(k_range):
                        elbow_k = k_range[elbow_idx]
                        reasoning.append(f"Phương pháp Elbow gợi ý k={elbow_k} dựa trên sự thay đổi lớn nhất trong WCSS.")
                    else:
                        reasoning.append("Phương pháp Elbow không áp dụng được do chỉ số nằm ngoài phạm vi k_range.")
                else:
                    reasoning.append("Phương pháp Elbow không áp dụng được do WCSS không thay đổi.")
            else:
                reasoning.append("Phương pháp Elbow không áp dụng được do không đủ điểm dữ liệu.")
        else:
            reasoning.append("Phương pháp Elbow không áp dụng được do WCSS không có giá trị hợp lệ.")
    else:
        reasoning.append("Phương pháp Elbow không áp dụng được do không có WCSS hoặc không đủ dữ liệu.")    # Tối ưu hóa: Gợi ý dựa trên chỉ số Silhouette với numpy
    silhouette_k = None
    if np.any(silhouette_scores != 0):
        silhouette_idx = np.argmax(silhouette_scores)
        silhouette_k = k_range[silhouette_idx]
        reasoning.append(f"Silhouette Score gợi ý k={silhouette_k} với giá trị cao nhất: {silhouette_scores[silhouette_idx]:.3f}.")
    else:
        reasoning.append("Silhouette Score không khả dụng (tất cả giá trị bằng 0).")

    # Tối ưu hóa: Gợi ý dựa trên chỉ số Calinski-Harabasz
    calinski_k = None
    if np.any(calinski_scores != 0):
        calinski_idx = np.argmax(calinski_scores)
        calinski_k = k_range[calinski_idx]
        reasoning.append(f"Calinski-Harabasz gợi ý k={calinski_k} với giá trị cao nhất: {calinski_scores[calinski_idx]:.3f}.")
    else:
        reasoning.append("Calinski-Harabasz không khả dụng (tất cả giá trị bằng 0).")

    # Tối ưu hóa: Gợi ý dựa trên chỉ số Davies-Bouldin
    davies_k = None
    valid_davies = davies_scores[davies_scores != float('inf')]
    if len(valid_davies) > 0:
        davies_idx = np.argmin(davies_scores)
        davies_k = k_range[davies_idx]
        reasoning.append(f"Davies-Bouldin gợi ý k={davies_k} với giá trị thấp nhất: {davies_scores[davies_idx]:.3f}.")
    else:
        reasoning.append("Davies-Bouldin không khả dụng (tất cả giá trị không hợp lệ).")

    # Tối ưu hóa: Gợi ý dựa trên chỉ số Starczewski
    starczewski_k = None
    if np.any(starczewski_scores != 0):
        starczewski_idx = np.argmax(starczewski_scores)
        starczewski_k = k_range[starczewski_idx]
        reasoning.append(f"Starczewski gợi ý k={starczewski_k} với giá trị cao nhất: {starczewski_scores[starczewski_idx]:.3f}.")
    else:
        reasoning.append("Starczewski không khả dụng (tất cả giá trị bằng 0).")

    # Tối ưu hóa: Gợi ý dựa trên chỉ số Wiroonsri
    wiroonsri_k = None
    if np.any(wiroonsri_scores != 0):
        wiroonsri_idx = np.argmax(wiroonsri_scores)
        wiroonsri_k = k_range[wiroonsri_idx]
        reasoning.append(f"Wiroonsri gợi ý k={wiroonsri_k} với giá trị cao nhất: {wiroonsri_scores[wiroonsri_idx]:.3f}.")
    else:
        reasoning.append("Wiroonsri không khả dụng (tất cả giá trị bằng 0).")
        reasoning.append("Calinski-Harabasz không khả dụng (tất cả giá trị bằng 0).")    # Comment: Gợi ý dựa trên chỉ số Davies-Bouldin (giá trị thấp nhất).
    if np.any(davies_scores != float('inf')):
        # Tìm chỉ số có giá trị nhỏ nhất (không phải inf)
        valid_davies = np.where(davies_scores != float('inf'), davies_scores, float('inf'))
        davies_idx = np.argmin(valid_davies)
        davies_k = k_range[davies_idx]
        reasoning.append(f"Davies-Bouldin gợi ý k={davies_k} với giá trị thấp nhất: {davies_scores[davies_idx]:.3f}.")
    else:
        reasoning.append("Davies-Bouldin không khả dụng (tất cả giá trị không hợp lệ).")

    # Comment: Gợi ý dựa trên chỉ số Starczewski (giá trị cao nhất).
    starczewski_k = None
    if np.any(starczewski_scores != 0):
        starczewski_idx = np.argmax(starczewski_scores)
        starczewski_k = k_range[starczewski_idx]
        reasoning.append(f"Starczewski gợi ý k={starczewski_k} với giá trị cao nhất: {starczewski_scores[starczewski_idx]:.3f}.")
    else:
        reasoning.append("Starczewski không khả dụng (tất cả giá trị bằng 0).")
        
    # Comment: Gợi ý dựa trên chỉ số Wiroonsri (giá trị cao nhất).
    wiroonsri_k = None
    if np.any(wiroonsri_scores != 0):
        wiroonsri_idx = np.argmax(wiroonsri_scores)
        wiroonsri_k = k_range[wiroonsri_idx]
        reasoning.append(f"Wiroonsri gợi ý k={wiroonsri_k} với giá trị cao nhất: {wiroonsri_scores[wiroonsri_idx]:.3f}.")
    else:
        reasoning.append("Wiroonsri không khả dụng (tất cả giá trị bằng 0).")
        
    if use_wiroonsri_starczewski:
        # Comment: Sau khi tính BCVI - Gợi ý số cụm tối ưu dựa trên Starczewski và Wiroonsri, ưu tiên Starczewski
        if starczewski_k is not None and wiroonsri_k is not None:
            if starczewski_k == wiroonsri_k:
                optimal_k = starczewski_k
                reasoning.append(f"\n👉 GỢI Ý: Cả hai chỉ số Starczewski và Wiroonsri đều đồng ý số cụm tối ưu là k={optimal_k}.")
            else:
                # Ưu tiên Starczewski theo yêu cầu
                optimal_k = starczewski_k
                reasoning.append(f"\n👉 GỢI Ý: Sau khi tính BCVI - Ưu tiên sử dụng chỉ số Starczewski với k={optimal_k} (Wiroonsri gợi ý k={wiroonsri_k}).")
        elif wiroonsri_k is not None:
            optimal_k = wiroonsri_k
            reasoning.append(f"\n👉 GỢI Ý: Sau khi tính BCVI - Sử dụng chỉ số Wiroonsri với k={optimal_k} (Starczewski không khả dụng).")
        elif starczewski_k is not None:
            optimal_k = starczewski_k
            reasoning.append(f"\n👉 GỢI Ý: Sau khi tính BCVI - Sử dụng chỉ số Starczewski với k={optimal_k} (Wiroonsri không khả dụng).")
        else:
            # Kết hợp các gợi ý nếu không có Wiroonsri và Starczewski
            reasoning.append(f"\n👉 GỢI Ý: Sau khi tính BCVI - Không có Wiroonsri và Starczewski khả dụng, hãy xem xét các chỉ số khác.")
    else:
        # Comment: Trước khi tính BCVI - Gợi ý số cụm tối ưu dựa trên Silhouette và Elbow
        elbow_valid = any(wcss) and any(diff != 0 for diff in wcss_diff2)
        silhouette_valid = silhouette_k is not None
        
        if silhouette_valid and elbow_valid:
            if silhouette_k == elbow_k:
                optimal_k = silhouette_k
                reasoning.append(f"\n👉 GỢI Ý: Cả hai phương pháp Silhouette và Elbow đều đồng ý số cụm tối ưu là k={optimal_k}.")
            else:
                # Ưu tiên Silhouette vì nó là chỉ số đánh giá thường xuyên hơn
                optimal_k = silhouette_k
                reasoning.append(f"\n👉 GỢI Ý: Ưu tiên sử dụng Silhouette Score với k={optimal_k} (Elbow gợi ý k={elbow_k}).")
        elif silhouette_valid:
            optimal_k = silhouette_k
            reasoning.append(f"\n👉 GỢI Ý: Sử dụng Silhouette Score với k={optimal_k} (Elbow không khả dụng).")
        elif elbow_valid:
            optimal_k = elbow_k
            reasoning.append(f"\n👉 GỢI Ý: Sử dụng phương pháp Elbow với k={optimal_k} (Silhouette không khả dụng).")
        else:
            # Sử dụng phương pháp đếm tất cả các chỉ số
            k_counts = {}
            for k in k_range:
                k_counts[k] = 0
            
            # Đếm số lần mỗi giá trị `k` được gợi ý
            if any(score != 0 for score in calinski_scores):
                k_counts[calinski_k] += 1
            if any(score != float('inf') for score in davies_scores):
                k_counts[davies_k] += 1
            
            # Chọn giá trị `k` được gợi ý nhiều nhất
            max_count = max(k_counts.values())
            if max_count > 0:
                optimal_k = max(k for k, count in k_counts.items() if count == max_count)
                reasoning.append(f"\n👉 GỢI Ý: Silhouette và Elbow không khả dụng, chọn k={optimal_k} dựa trên các chỉ số khác.")
            else:
                optimal_k = k_range[0]
                reasoning.append(f"\n👉 GỢI Ý: Không có chỉ số CVI nào khả dụng, mặc định chọn k={optimal_k}.")
    
    # Comment: Ghi log giá trị `k` tối ưu và trả về kết quả.
    logging.debug(f"Optimal k: {optimal_k}")
    return optimal_k, "\n".join(reasoning)

def evaluate_cluster_indices_parallel(X, max_k, models=('KMeans', 'GMM', 'Hierarchical', 'FuzzyCMeans'), max_workers=None):
    """
    Đánh giá nhiều mô hình phân cụm với nhiều giá trị k đồng thời sử dụng đa luồng.
    
    Args:
        X (ndarray): Dữ liệu đầu vào
        max_k (int): Giá trị k tối đa để thử
        models (tuple): Danh sách các mô hình cần đánh giá
        max_workers (int): Số luồng tối đa. Nếu None, sẽ sử dụng số lõi CPU
        
    Returns:
        dict: Kết quả đánh giá cho tất cả các mô hình và giá trị k
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    import skfuzzy as fuzz
    
    logging.debug(f"Bắt đầu đánh giá song song cho {len(models)} mô hình và k từ 2 đến {max_k}")
    
    # Tạo các task cần chạy (mô hình x k)
    tasks = []
    for model_name in models:
        for k in range(2, max_k + 1):
            tasks.append((model_name, k))
    
    # Xác định số luồng tối đa
    if max_workers is None:
        import multiprocessing
        max_workers = multiprocessing.cpu_count()
    max_workers = min(max_workers, len(tasks))
    
    results = {}
    for model_name in models:
        results[model_name] = {}
    
    def run_single_model(task):
        """Chạy một mô hình với một giá trị k"""
        model_name, k = task
        try:
            logging.debug(f"Đánh giá {model_name} với k={k}")
            
            if model_name == 'KMeans':
                model = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
                labels = model.fit_predict(X)
                centroids = model.cluster_centers_
                inertia = model.inertia_
            elif model_name == 'GMM':
                model = GaussianMixture(n_components=k, n_init=3, max_iter=50, random_state=42)
                labels = model.fit_predict(X)
                centroids = model.means_
                inertia = None
            elif model_name == 'Hierarchical':
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = model.fit_predict(X)
                centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
                inertia = None
            elif model_name == 'FuzzyCMeans':
                try:
                    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        X.T, k, m=1.5, error=0.1, maxiter=150, init=None, seed=42
                    )
                    labels = np.argmax(u, axis=0)
                    centroids = cntr
                    # Tính inertia cho FCM
                    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) ** 2
                    inertia = np.sum((u.T ** 2) * distances)
                except Exception as e:
                    logging.error(f"Lỗi Fuzzy C-Means với k={k}: {str(e)}")
                    return model_name, k, None
            
            # Tính các chỉ số đánh giá
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
                davies = davies_bouldin_score(X, labels)
                starczewski = starczewski_index(X, labels, centroids)
                wiroonsri = wiroonsri_index(X, labels, centroids)
            else:
                silhouette = 0
                calinski = 0
                davies = float('inf')
                starczewski = 0
                wiroonsri = 0
            
            return model_name, k, {
                'labels': labels,
                'centroids': centroids,
                'silhouette': silhouette,
                'calinski': calinski,
                'davies': davies,
                'starczewski': starczewski,
                'wiroonsri': wiroonsri,
                'inertia': inertia
            }
        except Exception as e:
            logging.error(f"Lỗi khi đánh giá {model_name} với k={k}: {str(e)}")
            return model_name, k, None
    
    # Chạy các task đồng thời
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_results = list(executor.map(run_single_model, tasks))
        
    # Xử lý kết quả
    for result in future_results:
        if result is not None:
            model_name, k, data = result
            if data is not None:
                results[model_name][k] = data
    
    logging.debug(f"Hoàn thành đánh giá song song")
    return results

def suggest_optimal_k_parallel(model_results, use_wiroonsri_starczewski=True, max_workers=None):
    """
    Gợi ý giá trị k tối ưu cho nhiều mô hình song song
    
    Args:
        model_results (dict): Từ điển chứa kết quả của các mô hình
        use_wiroonsri_starczewski (bool): Có ưu tiên chỉ số Wiroonsri và Starczewski không
        max_workers (int): Số luồng tối đa
    
    Returns:
        dict: Gợi ý k tối ưu cho mỗi mô hình {'model_name': {'k': optimal_k, 'reasoning': reasoning}}
    """
    logging.debug(f"Bắt đầu gợi ý k tối ưu song song cho {len(model_results['models'])} mô hình")
    
    if 'models' not in model_results or not model_results['models']:
        logging.error("Không có mô hình nào được chạy")
        return {}      # Xác định số luồng tối đa - tăng lên tối đa 3 luồng
    if max_workers is None:
        import multiprocessing
        max_workers = min(3, multiprocessing.cpu_count())
    max_workers = min(max_workers, len(model_results['models']), 3)  # Tăng giới hạn lên 3 luồng
    
    suggestions = {}
    
    def process_model(model_name):
        """Xử lý một mô hình và trả về gợi ý k tối ưu"""
        try:
            plots = model_results['plots'].get(model_name, {})
            if not plots:
                logging.warning(f"Không tìm thấy dữ liệu cho mô hình {model_name}")
                return model_name, None, "Không tìm thấy dữ liệu"
            
            optimal_k, reasoning = suggest_optimal_k(
                plots=plots,
                k_range=model_results['k_range'],
                use_wiroonsri_starczewski=use_wiroonsri_starczewski
            )
            
            return model_name, optimal_k, reasoning
        except Exception as e:
            logging.error(f"Lỗi khi gợi ý k tối ưu cho {model_name}: {str(e)}")
            return model_name, 3, f"Lỗi: {str(e)}"
    
    # Chạy gợi ý song song cho mỗi mô hình
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_results = list(executor.map(process_model, model_results['models']))
    
    # Xử lý kết quả
    for result in future_results:
        model_name, optimal_k, reasoning = result
        suggestions[model_name] = {
            'k': optimal_k,
            'reasoning': reasoning,
            'method': 'wiroonsri_starczewski' if use_wiroonsri_starczewski else 'traditional'
        }
    
    logging.debug(f"Hoàn thành gợi ý k tối ưu song song")
    return suggestions