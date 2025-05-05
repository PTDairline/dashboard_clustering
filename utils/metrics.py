import numpy as np
from scipy.spatial.distance import pdist, squareform

def calinski_harabasz_index(X, labels, centroids):
    """
    Tính toán chỉ số Calinski-Harabasz (CH).
    Giá trị lớn nhất là tối ưu.
    """
    try:
        n = len(X)
        k = len(np.unique(labels))
        if k < 2 or n <= k:
            return 0.0
        
        # Tính trung tâm toàn bộ tập dữ liệu (v_0)
        v_0 = np.mean(X, axis=0)
        
        # Tính tử số: Between-cluster sum of squares (BSS)
        bss = 0.0
        for j in range(k):
            cluster_size = np.sum(labels == j)
            bss += cluster_size * np.linalg.norm(centroids[j] - v_0) ** 2
        
        # Tính mẫu số: Within-cluster sum of squares (WSS)
        wss = 0.0
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                distances = np.linalg.norm(points - centroids[j], axis=1) ** 2
                wss += np.sum(distances)
        
        if wss == 0:
            return 0.0
        
        # Tính CH index
        ch = (n - k) * bss / ((k - 1) * wss)
        return ch
    except Exception as e:
        return 0.0

def silhouette_index(X, labels):
    """
    Tính toán chỉ số Silhouette (SH).
    Giá trị lớn nhất là tối ưu.
    """
    try:
        if len(np.unique(labels)) < 2:
            return 0.0
        
        n = len(X)
        k = len(np.unique(labels))
        
        # Tính khoảng cách giữa tất cả các cặp điểm
        distances = squareform(pdist(X))
        
        silhouette_scores = np.zeros(n)
        for i in range(n):
            # Cụm của điểm i
            cluster_i = labels[i]
            cluster_points = np.where(labels == cluster_i)[0]
            cluster_size = len(cluster_points)
            
            if cluster_size <= 1:
                silhouette_scores[i] = 0
                continue
            
            # Tính a(i): Khoảng cách trung bình đến các điểm trong cùng cụm
            a_i = np.sum(distances[i, cluster_points]) / (cluster_size - 1)
            
            # Tính b(i): Khoảng cách trung bình đến các điểm trong cụm gần nhất khác
            b_i = float('inf')
            for r in range(k):
                if r == cluster_i:
                    continue
                other_cluster_points = np.where(labels == r)[0]
                if len(other_cluster_points) == 0:
                    continue
                b_i_r = np.mean(distances[i, other_cluster_points])
                b_i = min(b_i, b_i_r)
            
            # Tính silhouette score cho điểm i
            if max(a_i, b_i) == 0:
                silhouette_scores[i] = 0
            else:
                silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        
        # Tính chỉ số Silhouette trung bình
        sh = np.mean(silhouette_scores)
        return sh
    except Exception as e:
        return 0.0

def davies_bouldin_index(X, labels, centroids, q=2, t=2):
    """
    Tính toán chỉ số Davies-Bouldin (DB).
    Giá trị nhỏ nhất là tối ưu.
    """
    try:
        if len(np.unique(labels)) < 2:
            return float('inf')
        
        k = len(np.unique(labels))
        
        # Tính S_i,q cho mỗi cụm
        S = np.zeros(k)
        for i in range(k):
            points = X[labels == i]
            if len(points) == 0:
                S[i] = float('inf')
                continue
            distances = np.linalg.norm(points - centroids[i], axis=1)
            S[i] = (np.sum(distances ** q) / len(points)) ** (1/q)
        
        # Tính M_ij,t giữa các cặp cụm
        M = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    M[i, j] = float('inf')
                else:
                    diff = centroids[i] - centroids[j]
                    M[i, j] = (np.sum(np.abs(diff) ** t)) ** (1/t)
        
        # Tính R_i,q
        R = np.zeros(k)
        for i in range(k):
            max_val = -float('inf')
            for j in range(k):
                if i == j:
                    continue
                val = (S[i] + S[j]) / M[i, j]
                max_val = max(max_val, val)
            R[i] = max_val
        
        # Tính DB index
        db = np.mean(R)
        return db if np.isfinite(db) else float('inf')
    except Exception as e:
        return float('inf')

def starczewski_index(X, labels, centroids):
    """
    Tính toán chỉ số Starczewski (STR).
    Giá trị lớn nhất là tối ưu.
    Chỉ tính E(k) và D(k) tại k hiện tại.
    """
    try:
        if len(np.unique(labels)) < 2:
            return 0.0
        
        k = len(np.unique(labels))
        
        # Tính D(k)
        max_dist = -float('inf')
        min_dist = float('inf')
        for i in range(k):
            for j in range(i + 1, k):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                max_dist = max(max_dist, dist)
                min_dist = min(min_dist, dist)
        if min_dist == 0:
            D_k = 0.0
        else:
            D_k = max_dist / min_dist
        
        # Tính E(k)
        v_0 = np.mean(X, axis=0)  # Trung tâm toàn bộ tập dữ liệu
        num = np.sum(np.linalg.norm(X - v_0, axis=1))
        den = 0.0
        for j in range(k):
            points = X[labels == j]
            if len(points) > 0:
                den += np.sum(np.linalg.norm(points - centroids[j], axis=1))
        if den == 0:
            E_k = 0.0
        else:
            E_k = num / den
        
        # Tính STR(k) (bỏ qua E(k-1) và D(k+1))
        return E_k * D_k
    except Exception as e:
        return 0.0

def wiroonsri_index(X, labels, centroids):
    """
    Tính toán chỉ số Wiroonsri (WI).
    Giá trị lớn nhất là tối ưu.
    Chỉ tính NC(k) tại k hiện tại.
    """
    try:
        if len(np.unique(labels)) < 2:
            return 0.0
        
        n = len(X)
        
        # Tính vector d: Khoảng cách giữa tất cả các cặp điểm
        d = pdist(X)
        
        # Tính vector c(k): Khoảng cách giữa các trung tâm cụm tương ứng
        c = np.zeros(len(d))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c[idx] = np.linalg.norm(centroids[labels[i]] - centroids[labels[j]])
                idx += 1
        
        # Tính NC(k) = Corr(d, c(k))
        if np.std(d) == 0 or np.std(c) == 0:
            nc_k = 0.0
        else:
            nc_k = np.corrcoef(d, c)[0, 1]
        
        # Tính NC(1)
        std_d = np.std(d)
        if std_d == 0:
            nc_1 = 0.0
        else:
            nc_1 = std_d / (np.max(d) - np.min(d))
        
        # Tính WI(k) (bỏ qua NCI1 và NCI2 do cần k-1 và k+1)
        return nc_k if nc_k >= nc_1 else nc_1
    except Exception as e:
        return 0.0

def suggest_optimal_k(plots, k_range):
    """
    Gợi ý số k tối ưu dựa trên các chỉ số CVI và biểu đồ Elbow.
    """
    # Khởi tạo biến để lưu kết quả
    optimal_k = k_range[0]  # Giá trị mặc định
    reasoning = []

    # Lấy các chỉ số CVI
    cvi_data = plots['cvi']
    silhouette_scores = [entry['Silhouette'] for entry in cvi_data]
    calinski_scores = [entry['Calinski-Harabasz'] for entry in cvi_data]
    davies_scores = [entry['Davies-Bouldin'] for entry in cvi_data]
    starczewski_scores = [entry['Starczewski'] for entry in cvi_data]
    wiroonsri_scores = [entry['Wiroonsri'] for entry in cvi_data]

    # Phương pháp Elbow (chỉ áp dụng cho KMeans và Fuzzy C-Means)
    wcss = plots['elbow']['inertias']
    elbow_k = k_range[0]  # Mặc định
    if any(wcss):  # Kiểm tra xem WCSS có giá trị hợp lệ không
        wcss_diff = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
        wcss_diff2 = [wcss_diff[i] - wcss_diff[i+1] for i in range(len(wcss_diff)-1)]
        if wcss_diff2 and any(diff != 0 for diff in wcss_diff2):  # Kiểm tra wcss_diff2 không rỗng và không toàn 0
            elbow_idx = wcss_diff2.index(max(wcss_diff2)) + 2
            elbow_k = k_range[elbow_idx]
            reasoning.append(f"Phương pháp Elbow gợi ý k={elbow_k} dựa trên sự thay đổi lớn nhất trong WCSS.")
        else:
            reasoning.append("Phương pháp Elbow không áp dụng được do WCSS không thay đổi hoặc không có giá trị hợp lệ.")
    else:
        reasoning.append("Phương pháp Elbow không áp dụng được do không có WCSS (thuật toán không hỗ trợ).")

    # Gợi ý dựa trên các chỉ số CVI
    # 1. Silhouette Score (giá trị cao nhất)
    if any(score != 0 for score in silhouette_scores):
        silhouette_idx = silhouette_scores.index(max(silhouette_scores))
        silhouette_k = k_range[silhouette_idx]
        reasoning.append(f"Silhouette Score gợi ý k={silhouette_k} với giá trị cao nhất: {silhouette_scores[silhouette_idx]:.3f}.")
    else:
        reasoning.append("Silhouette Score không khả dụng (tất cả giá trị bằng 0).")

    # 2. Calinski-Harabasz (giá trị cao nhất)
    if any(score != 0 for score in calinski_scores):
        calinski_idx = calinski_scores.index(max(calinski_scores))
        calinski_k = k_range[calinski_idx]
        reasoning.append(f"Calinski-Harabasz gợi ý k={calinski_k} với giá trị cao nhất: {calinski_scores[calinski_idx]:.3f}.")
    else:
        reasoning.append("Calinski-Harabasz không khả dụng (tất cả giá trị bằng 0).")

    # 3. Davies-Bouldin (giá trị thấp nhất)
    if any(score != float('inf') for score in davies_scores):
        davies_idx = davies_scores.index(min(score for score in davies_scores if score != float('inf')))
        davies_k = k_range[davies_idx]
        reasoning.append(f"Davies-Bouldin gợi ý k={davies_k} với giá trị thấp nhất: {davies_scores[davies_idx]:.3f}.")
    else:
        reasoning.append("Davies-Bouldin không khả dụng (tất cả giá trị không hợp lệ).")

    # 4. Starczewski (giá trị cao nhất)
    if any(score != 0 for score in starczewski_scores):
        starczewski_idx = starczewski_scores.index(max(starczewski_scores))
        starczewski_k = k_range[starczewski_idx]
        reasoning.append(f"Starczewski gợi ý k={starczewski_k} với giá trị cao nhất: {starczewski_scores[starczewski_idx]:.3f}.")
    else:
        reasoning.append("Starczewski không khả dụng (tất cả giá trị bằng 0).")

    # 5. Wiroonsri (giá trị cao nhất)
    if any(score != 0 for score in wiroonsri_scores):
        wiroonsri_idx = wiroonsri_scores.index(max(wiroonsri_scores))
        wiroonsri_k = k_range[wiroonsri_idx]
        reasoning.append(f"Wiroonsri gợi ý k={wiroonsri_k} với giá trị cao nhất: {wiroonsri_scores[wiroonsri_idx]:.3f}.")
    else:
        reasoning.append("Wiroonsri không khả dụng (tất cả giá trị bằng 0).")

    # Kết hợp các gợi ý để chọn k tối ưu
    k_counts = {}
    for k in k_range:
        k_counts[k] = 0
    
    # Đếm số lần k được gợi ý
    if any(wcss) and any(diff != 0 for diff in wcss_diff2):
        k_counts[elbow_k] += 1
    if any(score != 0 for score in silhouette_scores):
        k_counts[silhouette_k] += 1
    if any(score != 0 for score in calinski_scores):
        k_counts[calinski_k] += 1
    if any(score != float('inf') for score in davies_scores):
        k_counts[davies_k] += 1
    if any(score != 0 for score in starczewski_scores):
        k_counts[starczewski_k] += 1
    if any(score != 0 for score in wiroonsri_scores):
        k_counts[wiroonsri_k] += 1
    
    # Chọn k được gợi ý nhiều nhất
    max_count = max(k_counts.values())
    if max_count > 0:
        optimal_k = max(k for k, count in k_counts.items() if count == max_count)
    else:
        optimal_k = k_range[0]  # Nếu không có gợi ý nào, chọn k nhỏ nhất
        reasoning.append(f"Không có chỉ số CVI nào khả dụng, mặc định chọn k={optimal_k}.")

    return optimal_k, "\n".join(reasoning)