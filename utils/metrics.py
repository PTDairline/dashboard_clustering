import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

# Comment: Import các thư viện cần thiết.
# - numpy: Dùng cho tính toán số học.
# - scipy.spatial.distance: Dùng để tính khoảng cách (pdist, squareform).
# - sklearn.metrics: Dùng để tính các chỉ số CVI (silhouette_score, davies_bouldin_score).
# - logging: Dùng để ghi log.

# Comment: Thiết lập logging để ghi lại thông tin debug.
# - Định dạng log: thời gian, mức độ, thông điệp.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def calinski_harabasz_index(X, labels, centroids):
    """
    Tính toán chỉ số Calinski-Harabasz (CH).
    Giá trị lớn nhất là tối ưu.
    """
    try:
        # Comment: Ghi log bắt đầu tính toán chỉ số Calinski-Harabasz.
        logging.debug("Bắt đầu tính Calinski-Harabasz")
        
        # Comment: Lấy số mẫu `n` và số cụm `k` từ dữ liệu.
        # - Kiểm tra điều kiện: cần ít nhất 2 cụm và số mẫu phải lớn hơn số cụm.
        n = len(X)
        k = len(np.unique(labels))
        if k < 2 or n <= k:
            logging.warning("Số cụm không hợp lệ để tính Calinski-Harabasz")
            return 0.0
        
        # Comment: Tính trung tâm toàn bộ tập dữ liệu (`v_0`).
        v_0 = np.mean(X, axis=0)
        
        # Comment: Tính tử số: Between-cluster sum of squares (BSS).
        # - Tổng bình phương khoảng cách từ trung tâm cụm đến trung tâm toàn bộ, nhân với kích thước cụm.
        bss = 0.0
        for j in range(k):
            cluster_size = np.sum(labels == j)
            bss += cluster_size * np.linalg.norm(centroids[j] - v_0) ** 2
        
        # Comment: Tính mẫu số: Within-cluster sum of squares (WSS).
        # - Tổng bình phương khoảng cách từ các điểm trong cụm đến trung tâm cụm.
        wss = 0.0
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                distances = np.linalg.norm(points - centroids[j], axis=1) ** 2
                wss += np.sum(distances)
        
        # Comment: Kiểm tra WSS, nếu bằng 0 thì trả về giá trị mặc định.
        if wss == 0:
            logging.warning("WSS bằng 0, trả về Calinski-Harabasz mặc định")
            return 0.0
        
        # Comment: Tính chỉ số Calinski-Harabasz theo công thức: (n-k)*BSS / ((k-1)*WSS).
        ch = (n - k) * bss / ((k - 1) * wss)
        logging.debug(f"Calinski-Harabasz: {ch}")
        return ch
    except Exception as e:
        # Comment: Xử lý lỗi nếu tính toán thất bại.
        logging.error(f"Lỗi tính Calinski-Harabasz: {str(e)}")
        return 0.0

def silhouette_index(X, labels):
    """
    Tính toán chỉ số Silhouette (SH) bằng sklearn.
    Giá trị lớn nhất là tối ưu.
    """
    try:
        # Comment: Ghi log bắt đầu tính toán chỉ số Silhouette.
        logging.debug("Bắt đầu tính Silhouette")
        
        # Comment: Kiểm tra số cụm, cần ít nhất 2 cụm để tính Silhouette.
        if len(np.unique(labels)) < 2:
            logging.warning("Số cụm không hợp lệ để tính Silhouette")
            return 0.0
        
        # Comment: Tính chỉ số Silhouette bằng `sklearn.metrics.silhouette_score`.
        sh = silhouette_score(X, labels)
        logging.debug(f"Silhouette: {sh}")
        return sh
    except Exception as e:
        # Comment: Xử lý lỗi nếu tính toán thất bại.
        logging.error(f"Lỗi tính Silhouette: {str(e)}")
        return 0.0

def davies_bouldin_index(X, labels, centroids, q=2, t=2):
    """
    Tính toán chỉ số Davies-Bouldin (DB) bằng sklearn.
    Giá trị nhỏ nhất là tối ưu.
    """
    try:
        # Comment: Ghi log bắt đầu tính toán chỉ số Davies-Bouldin.
        logging.debug("Bắt đầu tính Davies-Bouldin")
        
        # Comment: Kiểm tra số cụm, cần ít nhất 2 cụm để tính Davies-Bouldin.
        if len(np.unique(labels)) < 2:
            logging.warning("Số cụm không hợp lệ để tính Davies-Bouldin")
            return float('inf')
        
        # Comment: Tính chỉ số Davies-Bouldin bằng `sklearn.metrics.davies_bouldin_score`.
        # - Tham số `centroids`, `q`, `t` không được sử dụng vì dùng `sklearn`.
        db = davies_bouldin_score(X, labels)
        logging.debug(f"Davies-Bouldin: {db}")
        
        # Comment: Kiểm tra giá trị hợp lệ, trả về vô cực nếu không hợp lệ.
        return db if np.isfinite(db) else float('inf')
    except Exception as e:
        # Comment: Xử lý lỗi nếu tính toán thất bại.
        logging.error(f"Lỗi tính Davies-Bouldin: {str(e)}")
        return float('inf')

def starczewski_index(X, labels, centroids):
    """
    Tính toán chỉ số Starczewski (STR).
    Giá trị lớn nhất là tối ưu.
    Chỉ tính E(k) và D(k) tại k hiện tại.
    """
    try:
        # Comment: Ghi log bắt đầu tính toán chỉ số Starczewski.
        logging.debug("Bắt đầu tính Starczewski")
        
        # Comment: Kiểm tra số cụm, cần ít nhất 2 cụm để tính Starczewski.
        if len(np.unique(labels)) < 2:
            logging.warning("Số cụm không hợp lệ để tính Starczewski")
            return 0.0
        
        k = len(np.unique(labels))
        
        # Comment: Tính D(k): Tỷ lệ giữa khoảng cách lớn nhất và nhỏ nhất giữa các trung tâm cụm.
        # - Sử dụng `pdist` để tính khoảng cách giữa các trung tâm cụm.
        centroid_dists = pdist(centroids)
        max_dist = np.max(centroid_dists)
        min_dist = np.min(centroid_dists)
        if min_dist == 0:
            D_k = 0.0
        else:
            D_k = max_dist / min_dist
        
        # Comment: Tính E(k): Tỷ lệ giữa tổng khoảng cách từ các điểm đến trung tâm toàn bộ và tổng khoảng cách trong cụm.
        v_0 = np.mean(X, axis=0)  # Trung tâm toàn bộ tập dữ liệu
        num = np.sum(np.linalg.norm(X - v_0, axis=1))
        den = 0.0
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                den += np.sum(np.linalg.norm(points - centroids[j], axis=1))
        if den == 0:
            logging.warning("Denominator bằng 0, trả về Starczewski mặc định")
            E_k = 0.0
        else:
            E_k = num / den
        
        # Comment: Tính chỉ số Starczewski: D(k) * E(k).
        str_idx = E_k * D_k
        logging.debug(f"Starczewski: {str_idx}")
        return str_idx
    except Exception as e:
        # Comment: Xử lý lỗi nếu tính toán thất bại.
        logging.error(f"Lỗi tính Starczewski: {str(e)}")
        return 0.0

def wiroonsri_index(X, labels, centroids):
    """
    Tính toán chỉ số Wiroonsri (WI) bằng cách lấy mẫu ngẫu nhiên - Phiên bản tối ưu.
    Giá trị lớn nhất là tối ưu.
    Chỉ tính NC(k) tại k hiện tại.
    """
    try:
        # Comment: Ghi log bắt đầu tính toán chỉ số Wiroonsri.
        logging.debug("Bắt đầu tính Wiroonsri")
        
        # Comment: Kiểm tra số cụm, cần ít nhất 2 cụm để tính Wiroonsri.
        if len(np.unique(labels)) < 2:
            logging.warning("Số cụm không hợp lệ để tính Wiroonsri")
            return 0.0
        
        # Tối ưu hóa: Giảm sample size để tăng tốc độ
        n = len(X)
        if n < 1000:
            sample_size = n
        else:
            sample_size = min(1000, n)  # Giảm từ 2000 xuống 1000
        
        # Comment: Lấy mẫu ngẫu nhiên các điểm và nhãn tương ứng.
        indices = np.random.choice(n, size=sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
        
        # Tối ưu hóa: Sử dụng numpy operations nhanh hơn
        # Comment: Tính vector `d`: Khoảng cách giữa các cặp điểm trong mẫu.
        d = pdist(X_sample)
        
        # Comment: Tính vector `c(k)`: Khoảng cách giữa các trung tâm cụm tương ứng với các cặp điểm.
        # Tối ưu hóa: Vectorized computation thay vì nested loops
        n_pairs = len(d)
        c = np.zeros(n_pairs)
        
        pair_idx = 0
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                c[pair_idx] = np.linalg.norm(centroids[labels_sample[i]] - centroids[labels_sample[j]])
                pair_idx += 1
        
        # Comment: Tính NC(k): Hệ số tương quan giữa `d` và `c(k)`.
        # Tối ưu hóa: Kiểm tra nhanh và an toàn hơn
        std_d = np.std(d)
        std_c = np.std(c)
        
        if std_d == 0 or std_c == 0:
            logging.warning("Độ lệch chuẩn của d hoặc c bằng 0, trả về Wiroonsri mặc định")
            nc_k = 0.0
        else:
            nc_k = np.corrcoef(d, c)[0, 1]
            if np.isnan(nc_k):
                nc_k = 0.0
        
        # Comment: Tính NC(1): Độ chuẩn hóa của độ lệch chuẩn `d`.
        if std_d == 0:
            logging.warning("Độ lệch chuẩn của d bằng 0, trả về NC(1) mặc định")
            nc_1 = 0.0
        else:
            d_range = np.max(d) - np.min(d)
            nc_1 = std_d / d_range if d_range > 0 else 0.0
        
        # Comment: Tính chỉ số Wiroonsri: Lấy giá trị lớn hơn giữa `nc_k` và `nc_1`.
        wi = max(nc_k, nc_1)
        logging.debug(f"Wiroonsri: {wi}")
        return wi
    except Exception as e:
        # Comment: Xử lý lỗi nếu tính toán thất bại.
        logging.error(f"Lỗi tính Wiroonsri: {str(e)}")
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