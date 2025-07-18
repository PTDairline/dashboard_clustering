import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import skfuzzy as fuzz
import base64
import io
import pandas as pd
import logging
import concurrent.futures  # Thêm concurrent.futures để hỗ trợ đa luồng
import time  # Thêm thư viện time để đo thời gian xử lý
import threading  # Thêm threading để tạo khóa đồng bộ
from utils.metrics import starczewski_index, wiroonsri_index

# Tạo khóa đồng bộ toàn cục cho matplotlib để tránh xung đột khi vẽ biểu đồ song song
plot_lock = threading.RLock()

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
        with plot_lock:
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

def generate_clustering_plots(X, model_name, k_range, selected_k, use_pca=True, selected_features=None, explained_variance_ratio=0.95, max_workers=None):
    """
    Tạo biểu đồ phân cụm với các mô hình khác nhau, sử dụng đa luồng để tăng tốc.
    
    Args:
        X (DataFrame): Dữ liệu đầu vào.
        model_name (str): Tên mô hình (KMeans, GMM, Hierarchical, FuzzyCMeans).
        k_range (list): Phạm vi số cụm để thử.
        selected_k (int): Số cụm tối đa được chọn.
        use_pca (bool, optional): Có sử dụng PCA hay không. Mặc định là True.
        selected_features (list, optional): Danh sách feature đã chọn. Mặc định là None.
        explained_variance_ratio (float, optional): Tỷ lệ phương sai giải thích. Mặc định là 0.95.
        max_workers (int, optional): Số lượng luồng tối đa. Mặc định là None (số luồng CPU).
    
    Returns:
        dict: Dictionary chứa các biểu đồ và điểm đánh giá.
    """
    try:
        # Logging bắt đầu và thông tin đầu vào.
        logging.debug(f"Bắt đầu generate_clustering_plots: model={model_name}, k_range={k_range}, selected_k={selected_k}")
        
        # Kiểm tra dữ liệu đầu vào
        if X.empty or len(X.columns) < 2:
            logging.error("Dữ liệu không đủ để chạy phân cụm")
            return {'error': 'Dữ liệu không đủ để chạy phân cụm (cần ít nhất 2 cột số và không rỗng).'}
        
        if X.isna().any().any():
            logging.error("Dữ liệu chứa giá trị NaN")
            return {'error': 'Dữ liệu chứa giá trị NaN. Vui lòng xử lý dữ liệu trước khi chạy phân cụm.'}
        
        # Lọc các cột số từ dữ liệu
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty or len(X_numeric.columns) < 2:
            logging.error("Dữ liệu không chứa đủ cột số")
            return {'error': 'Dữ liệu không chứa đủ cột số (cần ít nhất 2 cột số).'}
        
        if np.isinf(X_numeric.values).any():
            logging.error("Dữ liệu chứa giá trị vô cực")
            return {'error': 'Dữ liệu chứa giá trị vô cực (inf). Vui lòng xử lý dữ liệu trước khi chạy phân cụm.'}
        
        # Chuyển đổi dữ liệu thành mảng numpy
        X_array = X_numeric.values.astype(float)
        
        # Kiểm tra kích thước dữ liệu
        n_samples, n_features = X_array.shape
        logging.debug(f"Kích thước dữ liệu: samples={n_samples}, features={n_features}")
        if n_samples < 2:
            logging.error("Số mẫu quá nhỏ")
            return {'error': 'Số lượng mẫu quá nhỏ (ít nhất 2 mẫu để phân cụm).'}
        if n_features < 2:
            logging.error("Số feature quá nhỏ")
            return {'error': 'Số lượng feature quá nhỏ (ít nhất 2 feature để phân cụm).'}
        
        # Kiểm tra tính hợp lệ của k_range
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
        
        # Khởi tạo dictionary kết quả
        plots = {
            'silhouette': {'scores': [], 'plot': None},
            'elbow': {'inertias': [], 'plot': None},
            'scatter': [],
            'cvi': []
        }
        
        # Giảm chiều dữ liệu cho việc vẽ biểu đồ nếu cần
        X_plot = X_array
        if n_features > 2:
            logging.debug("Giảm chiều dữ liệu để vẽ scatter")
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X_array)
        
        # Chạy phân cụm với đa luồng
        logging.debug(f"Bắt đầu chạy phân cụm đa luồng cho {len(k_range)} giá trị k")
        cluster_results = []
        
        # Xác định số luồng tối đa dựa trên CPU hoặc giá trị được cung cấp
        if max_workers is None:
            import multiprocessing
            max_workers = multiprocessing.cpu_count()
        max_workers = min(max_workers, len(k_range))  # Không cần nhiều luồng hơn số k_range
        
        # Sử dụng ThreadPoolExecutor để chạy song song các giá trị k
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Tạo các task và thêm vào executor
            future_to_k = {
                executor.submit(run_cluster_for_k, X_array, X_plot, k, model_name): k 
                for k in k_range
            }
            
            # Thu thập kết quả khi hoàn thành
            for future in concurrent.futures.as_completed(future_to_k):
                k = future_to_k[future]
                try:
                    result = future.result()
                    if 'error' in result:
                        logging.error(f"Lỗi khi chạy k={k}: {result['error']}")
                        return {'error': result['error']}
                    cluster_results.append(result)
                    logging.debug(f"Hoàn thành phân cụm với k={k}")
                except Exception as e:
                    logging.error(f"Lỗi xử lý kết quả cho k={k}: {str(e)}")
                    return {'error': f"Lỗi xử lý kết quả cho k={k}: {str(e)}"}
        
        # Sắp xếp kết quả theo k để đảm bảo thứ tự đúng
        cluster_results.sort(key=lambda x: x['k'])
        
        # Chuẩn bị dữ liệu cho các biểu đồ và chỉ số
        silhouette_scores = []
        inertias = []
        cvi_scores = []
        
        # Xử lý kết quả từ đa luồng
        for result in cluster_results:
            k = result['k']
            
            # Thêm thông tin scatter plot
            plots['scatter'].append({
                'k': k, 
                'plot': result['scatter_plot']
            })
            
            # Thêm thông tin CVI
            cvi_scores.append({
                'k': k,
                'Silhouette': result['silhouette'],
                'Calinski-Harabasz': result['calinski'],
                'Davies-Bouldin': result['davies'],
                'Starczewski': result['starczewski'],
                'Wiroonsri': result['wiroonsri']
            })
            
            # Thêm điểm silhouette
            silhouette_scores.append(result['silhouette'])
            
            # Thêm inertia nếu có
            if result['inertia'] is not None:
                inertias.append(result['inertia'])
          # Tạo biểu đồ Silhouette tổng hợp
        logging.debug("Tạo biểu đồ Silhouette tổng hợp")
        with plot_lock:
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
          # Tạo biểu đồ Elbow nếu có dữ liệu inertias
        if model_name in ['KMeans', 'FuzzyCMeans'] and inertias:
            logging.debug("Tạo biểu đồ Elbow")
            with plot_lock:
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
            plots['elbow']['inertias'] = [0] * len(k_range)
            plots['elbow']['plot'] = None
        
        # Cập nhật các chỉ số CVI
        plots['cvi'] = cvi_scores
        logging.debug("Hoàn thành generate_clustering_plots với đa luồng")
        
        return plots
    except Exception as e:
        logging.error(f"Lỗi trong generate_clustering_plots: {str(e)}")
        return {'error': f'Lỗi khi chạy phân cụm: {str(e)}'}
def run_model_parallel(model_name, X, k_range, selected_k, use_pca=True, selected_features=None, explained_variance_ratio=0.95, threads_per_model=None):
    """
    Chạy một mô hình phân cụm cụ thể với nhiều giá trị k song song
    
    Args:
        model_name (str): Tên mô hình ('KMeans', 'GMM', 'Hierarchical', 'FuzzyCMeans')
        X (DataFrame): Dữ liệu đầu vào
        k_range (list): Phạm vi giá trị k cần thử
        selected_k (int): Giá trị k tối đa
        use_pca (bool): Có sử dụng PCA không
        selected_features (list): Danh sách feature được chọn
        explained_variance_ratio (float): Tỷ lệ phương sai giải thích cho PCA
        threads_per_model (int): Số luồng được phân bổ cho mỗi mô hình
        
    Returns:
        dict: Kết quả phân cụm của mô hình
    """
    try:
        logging.debug(f"Bắt đầu chạy model {model_name} với {threads_per_model} luồng")
        start_time = time.time()
        
        # Gọi hàm generate_clustering_plots với số luồng được chỉ định
        result = generate_clustering_plots(
            X=X,
            model_name=model_name,
            k_range=k_range,
            selected_k=selected_k,
            use_pca=use_pca,
            selected_features=selected_features,
            explained_variance_ratio=explained_variance_ratio,
            max_workers=threads_per_model
        )
        
        # Thêm thông tin về thời gian xử lý
        elapsed_time = time.time() - start_time
        logging.debug(f"Hoàn thành model {model_name} trong {elapsed_time:.2f} giây")
        
        # Thêm thời gian xử lý vào kết quả
        if isinstance(result, dict) and 'error' not in result:
            result['processing_time'] = elapsed_time
        
        return model_name, result
    except Exception as e:
        logging.error(f"Lỗi khi chạy model {model_name}: {str(e)}")
        return model_name, {'error': f"Lỗi khi chạy model {model_name}: {str(e)}"}

def run_multiple_models(X, models, k_range, selected_k, use_pca=True, selected_features=None, explained_variance_ratio=0.95, max_workers=None, threads_per_model=None):
    """
    Chạy nhiều mô hình phân cụm song song
    
    Args:
        X (DataFrame): Dữ liệu đầu vào
        models (list): Danh sách các mô hình cần chạy ('KMeans', 'GMM', 'Hierarchical', 'FuzzyCMeans')
        k_range (list): Phạm vi giá trị k cần thử
        selected_k (int): Giá trị k tối đa
        use_pca (bool): Có sử dụng PCA không
        selected_features (list): Danh sách feature được chọn
        explained_variance_ratio (float): Tỷ lệ phương sai giải thích cho PCA
        max_workers (int): Số luồng tối đa cho việc chạy song song các mô hình
        threads_per_model (int): Số luồng được phân bổ cho mỗi mô hình
        
    Returns:
        dict: Kết quả phân cụm của tất cả mô hình
    """
    if not models:
        logging.error("Không có mô hình nào được chọn")
        return {}
    
    logging.debug(f"Bắt đầu chạy {len(models)} mô hình song song")
      # Xác định số luồng cho các mô hình
    import multiprocessing
      # Giới hạn tùy chỉnh tài nguyên CPU - tăng từ 6 lên 8 nhân
    MAX_CPU_CORES = 8  # Tăng số CPU tối đa
    MAX_PARALLEL_MODELS = 2  # Giữ nguyên số mô hình song song
    MIN_THREADS_PER_MODEL = 2  # Mỗi mô hình cần ít nhất 2 luồng
    MAX_THREADS_PER_MODEL = 4  # Tăng số luồng tối đa cho mỗi mô hình
    
    cpu_count = min(multiprocessing.cpu_count(), MAX_CPU_CORES)
    logging.debug(f"Giới hạn sử dụng tối đa {cpu_count} nhân CPU")
    
    # Giới hạn số mô hình chạy song song
    if max_workers is None:
        max_workers = min(len(models), MAX_PARALLEL_MODELS)
    else:
        max_workers = min(max_workers, len(models), MAX_PARALLEL_MODELS)
    
    # Nếu không chỉ định threads_per_model, phân bổ dựa trên CPU
    if threads_per_model is None:
        # Tăng số luồng từ 3 lên 4 cho mỗi mô hình
        threads_per_model = max(MIN_THREADS_PER_MODEL, min(MAX_THREADS_PER_MODEL, cpu_count // max_workers))
    
    logging.debug(f"Phân bổ {max_workers} luồng để chạy song song {len(models)} mô hình, mỗi mô hình sử dụng {threads_per_model} luồng")
    
    # Kết quả
    results = {
        'models': [],
        'k_range': k_range,
        'selected_k': selected_k,
        'cvi_scores': {},
        'plots': {},
        'optimal_k_suggestions': {},
        'processing_times': {}
    }
      # Chạy các mô hình song song
    # 1. Sử dụng ThreadPoolExecutor thay vì ProcessPoolExecutor để tránh Flask restart
    # 2. Đặt daemon=True để đảm bảo các thread sẽ thoát khi thread chính kết thúc
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, 
                                             thread_name_prefix='model_thread') as executor:
        # Chuẩn bị tham số cho các mô hình
        future_to_model = {
            executor.submit(
                run_model_parallel, 
                model_name, 
                X, 
                k_range, 
                selected_k, 
                use_pca, 
                selected_features, 
                explained_variance_ratio, 
                threads_per_model
            ): model_name for model_name in models
        }
        
        # Thu thập kết quả khi hoàn thành
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                model_name, result = future.result()
                if 'error' in result:
                    logging.error(f"Lỗi khi chạy mô hình {model_name}: {result['error']}")
                    continue
                
                results['models'].append(model_name)
                
                # Chuyển đổi cvi_scores về format mong muốn
                cvi_dict = {}
                for cvi_entry in result['cvi']:
                    k = str(cvi_entry['k'])
                    cvi_dict[k] = {
                        'silhouette': cvi_entry['Silhouette'],
                        'calinski_harabasz': cvi_entry['Calinski-Harabasz'],
                        'davies_bouldin': cvi_entry['Davies-Bouldin'],
                        'starczewski': cvi_entry['Starczewski'],
                        'wiroonsri': cvi_entry['Wiroonsri']
                    }
                
                results['cvi_scores'][model_name] = cvi_dict
                results['plots'][model_name] = result
                
                # Lưu thời gian xử lý
                if 'processing_time' in result:
                    results['processing_times'][model_name] = result['processing_time']
                
            except Exception as e:
                logging.error(f"Lỗi khi xử lý kết quả của mô hình {model_name}: {str(e)}")
    
    logging.debug(f"Hoàn thành chạy {len(results['models'])}/{len(models)} mô hình song song")
    return results
