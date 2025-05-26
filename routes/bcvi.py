from flask import render_template, request, redirect, url_for, flash, send_file, current_app, session, jsonify
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering import compute_bcvi
from utils.metrics import suggest_optimal_k  # Thêm dòng này
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz

# Comment: Import các thư viện và module cần thiết.
# - Flask: Dùng để xử lý request, render giao diện, và quản lý ứng dụng web.
# - os, pandas, numpy: Dùng cho xử lý file và dữ liệu.
# - json, logging: Dùng để lưu kết quả và ghi log.
# - compute_bcvi: Hàm từ module utils.clustering để tính chỉ số BCVI.
# - sklearn.cluster, skfuzzy: Dùng để chạy phân cụm với k tối ưu.

# Comment: Thiết lập logging để ghi lại thông tin debug.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_clusters(X, model_name, optimal_k):
    """
    Chạy phân cụm với k tối ưu và phân tích đặc trưng của từng cụm.
    
    Parameters:
    - X: Dữ liệu đầu vào (numpy array).
    - model_name: Tên mô hình (KMeans, GMM, Hierarchical, FuzzyCMeans).
    - optimal_k: Số cụm tối ưu (từ BCVI).
    
    Returns:
    - cluster_stats: Thống kê đặc trưng của từng cụm (trung bình, độ lệch chuẩn).
    - cluster_sizes: Số lượng điểm trong từng cụm.
    """
    try:
        # Comment: Chạy phân cụm với k tối ưu để lấy nhãn cụm.
        if model_name == 'KMeans':
            model = KMeans(n_clusters=optimal_k, n_init=1, max_iter=300, random_state=42)
            labels = model.fit_predict(X)
        elif model_name == 'GMM':
            model = GaussianMixture(n_components=optimal_k, n_init=1, max_iter=100, random_state=42)
            labels = model.fit_predict(X)
        elif model_name == 'Hierarchical':
            model = AgglomerativeClustering(n_clusters=optimal_k)
            labels = model.fit_predict(X)
        elif model_name == 'FuzzyCMeans':
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                X.T, optimal_k, m=1.5, error=0.05, maxiter=300, init=None, seed=42
            )
            labels = np.argmax(u, axis=0)
        else:
            raise ValueError(f"Mô hình {model_name} không hỗ trợ")
        
        # Comment: Chuyển dữ liệu thành DataFrame để dễ phân tích.
        df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
        df['Cluster'] = labels
        
        # Comment: Tính thống kê cho từng cụm: trung bình và độ lệch chuẩn.
        cluster_stats = {}
        cluster_sizes = {}
        for cluster in range(optimal_k):
            cluster_data = df[df['Cluster'] == cluster].drop(columns=['Cluster'])
            cluster_sizes[cluster] = len(cluster_data)
            cluster_stats[cluster] = {
                'mean': cluster_data.mean().to_dict(),
                'std': cluster_data.std().to_dict()
            }
        
        # Comment: Ghi log thông tin thống kê.
        logging.debug(f"Thống kê cụm: {cluster_stats}")
        logging.debug(f"Kích thước cụm: {cluster_sizes}")
        
        return cluster_stats, cluster_sizes
    
    except Exception as e:
        # Comment: Xử lý lỗi nếu phân cụm hoặc phân tích thất bại.
        logging.error(f"Lỗi phân tích cụm: {str(e)}")
        return None, None

def bcvi():
    # Kiểm tra session trước để tránh đọc file không cần thiết
    if 'bcvi_calculated' in session and request.method != 'POST':
        bcvi_cache_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_cache.pkl')
        bcvi_flag_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_calculated.flag')
        
        if os.path.exists(bcvi_flag_file) and os.path.exists(bcvi_cache_file):
            logging.debug("Sử dụng kết quả BCVI từ cache (session check)")
            try:
                # Đọc cache nhanh hơn với compression
                import time
                start_time = time.time()
                data = pd.read_pickle(bcvi_cache_file)
                load_time = time.time() - start_time
                logging.debug(f"Cache loaded in {load_time:.3f} seconds")
                
                # Kiểm tra tính hợp lệ của cache
                if 'bcvi_results' in data and data['bcvi_results']:
                    return render_template('bcvi.html', data=data)
                else:
                    logging.warning("Cache không hợp lệ, xóa và tính toán lại")
                    os.remove(bcvi_flag_file)
                    if os.path.exists(bcvi_cache_file):
                        os.remove(bcvi_cache_file)
                    session.pop('bcvi_calculated', None)
            except Exception as e:
                logging.error(f"Lỗi khi đọc bcvi_cache.pkl: {str(e)}")
                # Xóa file flag nếu có lỗi để tính toán lại
                if os.path.exists(bcvi_flag_file):
                    os.remove(bcvi_flag_file)
                if os.path.exists(bcvi_cache_file):
                    os.remove(bcvi_cache_file)
                session.pop('bcvi_calculated', None)
    # Comment: Khởi tạo dictionary `data` để lưu trạng thái và dữ liệu cho giao diện.
    # - `k_range`: Phạm vi số cụm (mặc định từ 2 đến 10).
    # - `selected_k`: Số cụm tối đa được chọn (mặc định là 2).
    # - `models`: Danh sách các mô hình phân cụm.
    # - `plots`: Kết quả phân cụm (từ `clustering_results.json`).
    # - `bcvi_results`: Kết quả BCVI cho từng mô hình.
    # - `optimal_k`: Số cụm tối ưu dựa trên BCVI.
    # - `alpha`: Danh sách tham số alpha cho BCVI.
    # - `cluster_stats`: Thống kê đặc trưng của từng cụm (thêm mới).
    # - `cluster_sizes`: Số lượng điểm trong từng cụm (thêm mới).
      # Kiểm tra xem đã có kết quả BCVI được lưu trong file chưa
    bcvi_cache_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_cache.pkl')
    bcvi_flag_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_calculated.flag')
    
    if os.path.exists(bcvi_flag_file) and not request.method == 'POST':
        logging.debug("Sử dụng kết quả BCVI từ cache")
        try:
            # Đọc cache nhanh hơn với compression
            import time
            start_time = time.time()
            data = pd.read_pickle(bcvi_cache_file)
            load_time = time.time() - start_time
            logging.debug(f"Cache loaded in {load_time:.3f} seconds")
            
            # Kiểm tra tính hợp lệ của cache
            if 'bcvi_results' in data and data['bcvi_results']:
                return render_template('bcvi.html', data=data)
            else:
                logging.warning("Cache không hợp lệ, xóa và tính toán lại")
                os.remove(bcvi_flag_file)
                if os.path.exists(bcvi_cache_file):
                    os.remove(bcvi_cache_file)
        except Exception as e:
            logging.error(f"Lỗi khi đọc bcvi_cache.pkl: {str(e)}")
            # Xóa file flag nếu có lỗi để tính toán lại
            if os.path.exists(bcvi_flag_file):
                os.remove(bcvi_flag_file)
            if os.path.exists(bcvi_cache_file):
                os.remove(bcvi_cache_file)
    
    data = {
        'k_range': list(range(2, 11)),
        'selected_k': 2,
        'models': [],
        'plots': {},
        'bcvi_results': {},
        'optimal_k': {},
        'alpha': [],
        'cluster_stats': {},  # Thêm để lưu thống kê cụm
        'cluster_sizes': {}   # Thêm để lưu kích thước cụm
    }
    
    # Comment: Kiểm tra file kết quả phân cụm (`clustering_results.json`) có tồn tại không.
    # - Nếu không, thông báo lỗi và chuyển hướng về trang chọn mô hình (`select_model`).
    clustering_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'clustering_results.json')
    if not os.path.exists(clustering_results_file):
        flash("Vui lòng chạy phân cụm trước.")
        return redirect(url_for('select_model'))
    
    # Comment: Đọc kết quả phân cụm từ file `clustering_results.json`.
    # - Cập nhật `data` với danh sách mô hình, số cụm tối đa, kết quả phân cụm, và gợi ý số cụm tối ưu.
    try:
        with open(clustering_results_file, 'r') as f:
            clustering_results = json.load(f)
        data['models'] = clustering_results.get('models', [])
        data['selected_k'] = clustering_results.get('selected_k', 2)
        data['plots'] = clustering_results.get('plots', {})
        data['optimal_k_suggestions'] = clustering_results.get('optimal_k_suggestions', {})
    except Exception as e:
        logging.error(f"Error reading clustering_results.json: {str(e)}")
        flash("Lỗi khi đọc kết quả phân cụm trước đó. Vui lòng chạy lại mô hình.")
        return redirect(url_for('select_model'))
    
    # Comment: Kiểm tra xem có mô hình nào được chạy thành công không.
    if not data['models']:
        flash("Không có mô hình nào được chạy thành công. Vui lòng chạy lại phân cụm.")
        return redirect(url_for('select_model'))
    
    # Comment: Cập nhật phạm vi số cụm (`k_range`) dựa trên `selected_k`.
    data['k_range'] = list(range(2, data['selected_k'] + 1))
    
    if request.method == 'POST':
        # Comment: Xử lý yêu cầu POST khi người dùng gửi form để tính BCVI.
        # - Lấy tham số `alpha` từ form, mỗi `alpha_k` tương ứng với một giá trị `k` trong `k_range`.
        alpha = []
        for k in data['k_range']:
            alpha_k = request.form.get(f'alpha_{k}')
            try:
                alpha_k = float(alpha_k)
                if alpha_k < 0:
                    raise ValueError
                alpha.append(alpha_k)
            except (ValueError, TypeError):
                flash(f"Tham số alpha_{k} không hợp lệ. Vui lòng nhập số không âm.")
                return render_template('bcvi.html', data=data)
        
        # Comment: Định nghĩa số lượng mẫu `n` (cố định là 100) để tính BCVI.
        n = 100
        
        # Comment: Tính toán BCVI cho từng mô hình và chỉ số CVI.
        try:
            bcvi_results = {}
            optimal_k = {}
            
            for model in data['models']:
                cvi_indices = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Starczewski', 'Wiroonsri']
                model_results = []
                
                # Comment: Làm sạch dữ liệu CVI trước khi tính BCVI.
                cleaned_cvi_data = []
                for cvi in data['plots'][model]['cvi']:
                    cleaned_cvi = {}
                    for key, value in cvi.items():
                        if key == 'k':
                            cleaned_cvi[key] = value
                        else:
                            try:
                                cleaned_cvi[key] = float(value) if value is not None else 0
                            except (ValueError, TypeError):
                                logging.warning(f"Invalid CVI value for {key} in model {model}: {value}. Setting to 0.")
                                cleaned_cvi[key] = 0
                    cleaned_cvi_data.append(cleaned_cvi)
                
                # Comment: Lấy danh sách các giá trị `k` thực sự có trong dữ liệu CVI.
                actual_k_values = [cvi['k'] for cvi in cleaned_cvi_data]
                logging.debug(f"Model: {model}, Actual k values in CVI data: {actual_k_values}")
                
                # Comment: Lọc `k_range` và `alpha` để chỉ giữ các giá trị `k` có trong dữ liệu CVI.
                filtered_alpha = []
                filtered_k_range = []
                alpha_index = 0
                for k in data['k_range']:
                    if k in actual_k_values:
                        filtered_k_range.append(k)
                        filtered_alpha.append(alpha[alpha_index])
                    alpha_index += 1
                logging.debug(f"Model: {model}, Filtered k range: {filtered_k_range}, Filtered alpha: {filtered_alpha}")
                
                if not filtered_k_range:
                    flash(f"Không có dữ liệu CVI cho mô hình {model} trong khoảng k đã chọn.")
                    continue
                  # Tối ưu hóa bằng cách tính BCVI một lần cho mỗi chỉ số CVI thay vì cho từng k
                cvi_to_bcvi_values = {}
                
                # Tính BCVI cho mỗi chỉ số CVI
                for cvi_index in cvi_indices:
                    cvi_values = [cvi.get(cvi_index, 0) for cvi in cleaned_cvi_data if cvi['k'] in filtered_k_range]
                    logging.debug(f"Model: {model}, CVI Index: {cvi_index}, CVI Values: {cvi_values}")
                    
                    if all(v == 0 for v in cvi_values):
                        logging.warning(f"All CVI values for {cvi_index} in model {model} are 0. Skipping BCVI calculation for this index.")
                        cvi_to_bcvi_values[cvi_index] = [0.0] * len(filtered_k_range)
                        continue
                    
                    if cvi_index in ['Silhouette', 'Calinski-Harabasz', 'Starczewski', 'Wiroonsri']:
                        opt_type = 'max'
                    else:
                        opt_type = 'min'
                    
                    # Tính BCVI một lần cho tất cả k
                    cvi_to_bcvi_values[cvi_index] = compute_bcvi(cvi_values, filtered_k_range, filtered_alpha, n, opt_type=opt_type)
                
                # Tạo kết quả cho từng k
                for i, k in enumerate(filtered_k_range):
                    cvi_entry = next((cvi for cvi in cleaned_cvi_data if cvi['k'] == k), None)
                    if not cvi_entry:
                        continue
                    
                    bcvi_entry = {'k': k, 'bcvi': {}}
                    for cvi_index in cvi_indices:
                        if cvi_index in cvi_to_bcvi_values:
                            bcvi_entry['bcvi'][cvi_index] = cvi_to_bcvi_values[cvi_index][i]
                    
                    model_results.append(bcvi_entry)
                
                if not model_results:
                    flash(f"Không có kết quả BCVI cho mô hình {model}.")
                    continue
                  # Lưu trữ kết quả BCVI
                bcvi_results[model] = model_results
                
                # Tìm số cụm tối ưu sau khi tính BCVI dựa trên chỉ số Wiroonsri và Starczewski
                for cvi_data in data['plots'][model]['cvi']:
                    k = cvi_data['k']
                    k_result = next((r for r in model_results if r['k'] == k), None)
                    if k_result:
                        # Cập nhật chỉ số CVI với giá trị BCVI 
                        for cvi_index in ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Starczewski', 'Wiroonsri']:
                            if cvi_index in k_result['bcvi']:
                                cvi_data[f"{cvi_index}_BCVI"] = k_result['bcvi'][cvi_index]
                  # Tìm k tối ưu dựa trên giá trị BCVI cao nhất của Wiroonsri và Starczewski
                optimal_k_value = None
                reasoning = []
                
                # Lấy BCVI values cho Wiroonsri và Starczewski
                wiroonsri_bcvi = [(result['k'], result['bcvi'].get('Wiroonsri', 0)) for result in model_results if result['bcvi'].get('Wiroonsri', 0) > 0]
                starczewski_bcvi = [(result['k'], result['bcvi'].get('Starczewski', 0)) for result in model_results if result['bcvi'].get('Starczewski', 0) > 0]
                
                if wiroonsri_bcvi and starczewski_bcvi:
                    # Tìm k có BCVI Wiroonsri cao nhất
                    wiroonsri_optimal = max(wiroonsri_bcvi, key=lambda x: x[1])
                    # Tìm k có BCVI Starczewski cao nhất  
                    starczewski_optimal = max(starczewski_bcvi, key=lambda x: x[1])
                    
                    if wiroonsri_optimal[0] == starczewski_optimal[0]:
                        optimal_k_value = wiroonsri_optimal[0]
                        reasoning.append(f"Cả Wiroonsri (BCVI={wiroonsri_optimal[1]:.4f}) và Starczewski (BCVI={starczewski_optimal[1]:.4f}) đều gợi ý k={optimal_k_value}")
                    else:
                        # Ưu tiên Starczewski nếu khác nhau
                        optimal_k_value = wiroonsri_optimal[0]
                        reasoning.append(f"Ưu tiên wiroonsri: k={wiroonsri_optimal[0]} (BCVI={wiroonsri_optimal[1]:.4f}) thay vì Starczewski: k={starczewski_optimal[0]} (BCVI={starczewski_optimal[1]:.4f})")
                        
                elif wiroonsri_bcvi:
                    optimal_k_value = max(wiroonsri_bcvi, key=lambda x: x[1])[0]
                    reasoning.append(f"Chỉ có Wiroonsri khả dụng, k={optimal_k_value}")
                elif starczewski_bcvi:
                    optimal_k_value = max(starczewski_bcvi, key=lambda x: x[1])[0]
                    reasoning.append(f"Chỉ có Starczewski khả dụng, k={optimal_k_value}")
                else:
                    # Dự phòng: sử dụng BCVI Silhouette
                    silhouette_bcvi = [(result['k'], result['bcvi'].get('Silhouette', 0)) for result in model_results if result['bcvi'].get('Silhouette', 0) > 0]
                    optimal_k_value = max(silhouette_bcvi, key=lambda x: x[1])[0] if silhouette_bcvi else filtered_k_range[0]
                    reasoning.append(f"Không có Wiroonsri/Starczewski, sử dụng Silhouette, k={optimal_k_value}")
                
                optimal_k[model] = optimal_k_value
                data['optimal_k_suggestions'][model] = {'k': optimal_k_value, 'reasoning': '\n'.join(reasoning)}
                
                logging.debug(f"Model: {model}, Optimal k based on BCVI: {optimal_k_value}")
                logging.debug(f"Model: {model}, Reasoning: {reasoning}")
            
            # Cập nhật data với kết quả BCVI (DÒNG QUAN TRỌNG BỊ THIẾU!)
            data['bcvi_results'] = bcvi_results
            
            if not bcvi_results:
                flash("Không có kết quả BCVI nào được tạo. Vui lòng kiểm tra dữ liệu phân cụm.")
                return render_template('bcvi.html', data=data)
            
            data['optimal_k'] = optimal_k
            data['alpha'] = alpha
              # Lưu kết quả BCVI vào file cache với compression để giảm kích thước
            try:
                import time
                start_time = time.time()
                
                bcvi_cache_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_cache.pkl')
                bcvi_flag_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_calculated.flag')
                
                # Tối ưu hóa dữ liệu trước khi lưu cache
                cache_data = data.copy()
                
                # Chỉ lưu các key cần thiết để giảm kích thước cache
                essential_keys = ['k_range', 'selected_k', 'models', 'plots', 'bcvi_results', 
                                'optimal_k', 'alpha', 'optimal_k_suggestions']
                cache_data = {key: data[key] for key in essential_keys if key in data}
                
                # Sử dụng compression level cao hơn
                pd.to_pickle(cache_data, bcvi_cache_file, compression='gzip')
                
                # Tạo file flag
                with open(bcvi_flag_file, 'w') as f:
                    f.write('1')
                
                save_time = time.time() - start_time
                logging.debug(f"Cache saved in {save_time:.3f} seconds với compression")
                
                # Chỉ lưu thông tin cơ bản vào session
                session['bcvi_calculated'] = True
                session['bcvi_models'] = list(bcvi_results.keys()) if bcvi_results else []
            except Exception as e:
                logging.error(f"Lỗi khi lưu bcvi_cache.pkl: {str(e)}")
            
            # Comment: Lưu kết quả BCVI vào file CSV (`bcvi_result.csv`).
            result_data = []
            for model in data['models']:
                if model not in bcvi_results:
                    continue
                for result in bcvi_results[model]:
                    row = {'Model': model, 'k': result['k']}
                    for cvi_index, value in result['bcvi'].items():
                        row[f'{cvi_index} BCVI'] = value
                    result_data.append(row)
            result_df = pd.DataFrame(result_data)
            result_df.to_csv(os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_result.csv'), index=False)
              # Comment: Phân tích đặc trưng của các cụm với k tối ưu.
            # - Đọc dữ liệu đã xử lý từ `processed_data.pkl`.
            # - Chạy phân cụm với k tối ưu cho từng mô hình và phân tích đặc trưng.
            try:
                processed_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')
                if os.path.exists(processed_data_file):
                    X = pd.read_pickle(processed_data_file)
                    X_array = X.select_dtypes(include=[np.number]).values.astype(float)
                    for model in data['models']:
                        if model in optimal_k:
                            k_opt = optimal_k[model]
                            cluster_stats, cluster_sizes = analyze_clusters(X_array, model, k_opt)
                            if cluster_stats and cluster_sizes:
                                data['cluster_stats'][model] = cluster_stats
                                data['cluster_sizes'][model] = cluster_sizes
                            else:
                                flash(f"Không thể phân tích đặc trưng cụm cho mô hình {model}.")
            except Exception as e:
                # Xử lý lỗi khi phân tích đặc trưng cụm
                flash(f"Lỗi khi phân tích đặc trưng cụm: {str(e)}")
                logging.error(f"Error in cluster analysis: {str(e)}")
                
        except Exception as e:
            # Xử lý lỗi chính khi tính toán BCVI
            flash(f"Lỗi khi tính toán BCVI: {str(e)}")
            # Đảm bảo xóa cache và flag để tránh lưu dữ liệu lỗi
            bcvi_cache_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_cache.pkl')
            bcvi_flag_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_calculated.flag')
            if os.path.exists(bcvi_flag_file):
                os.remove(bcvi_flag_file)
            if os.path.exists(bcvi_cache_file):
                os.remove(bcvi_cache_file)
            if 'bcvi_calculated' in session:
                session.pop('bcvi_calculated')
            logging.error(f"Error in BCVI calculation: {str(e)}")
        
        # Debug: Kiểm tra dữ liệu trước khi render
        logging.debug(f"Data keys before render: {list(data.keys())}")
        logging.debug(f"BCVI results available: {'bcvi_results' in data and bool(data['bcvi_results'])}")
        if 'bcvi_results' in data:
            logging.debug(f"Models with BCVI results: {list(data['bcvi_results'].keys())}")
        
        return render_template('bcvi.html', data=data)
    # Đảm bảo luôn có return statement cho GET request
    return render_template('bcvi.html', data=data)

def clean_bcvi_cache():
    """Xóa cache BCVI khi có thay đổi ở các bước trước đó"""
    bcvi_cache_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_cache.pkl')
    bcvi_flag_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_calculated.flag')
    if os.path.exists(bcvi_flag_file):
        os.remove(bcvi_flag_file)
    if os.path.exists(bcvi_cache_file):
        os.remove(bcvi_cache_file)
    if 'bcvi_calculated' in session:
        session.pop('bcvi_calculated')
    logging.debug("Đã xóa cache BCVI cũ")

def download_bcvi():
    # Comment: Hàm cho phép người dùng tải kết quả BCVI (`bcvi_result.csv`).
    bcvi_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_result.csv')
    if os.path.exists(bcvi_file):
        return send_file(bcvi_file, as_attachment=True)
    else:
        flash("Chưa có kết quả BCVI.")
        return redirect(url_for('bcvi'))