from flask import render_template, request, redirect, url_for, flash, current_app, session
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.clustering import generate_clustering_plots
from utils.metrics import suggest_optimal_k
from routes.bcvi import clean_bcvi_cache
from joblib import Parallel, delayed

# Comment: Import các thư viện và module cần thiết.
# - Flask: Dùng để xử lý request, render giao diện, và quản lý ứng dụng web.
# - os, pandas, numpy: Dùng cho xử lý file và dữ liệu.
# - json, logging: Dùng để lưu kết quả và ghi log.
# - generate_clustering_plots: Hàm từ module utils.clustering để chạy phân cụm và tạo biểu đồ.
# - suggest_optimal_k: Hàm từ module utils.metrics để gợi ý số cụm tối ưu.
# - joblib.Parallel, delayed: Dùng để chạy song song các tác vụ phân cụm.

# Comment: Thiết lập logging để ghi lại thông tin debug.
# - Định dạng log: thời gian, mức độ, thông điệp.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_clustering(model, X, k_range, selected_k, use_pca, selected_features, explained_variance):
    """Hàm hỗ trợ chạy phân cụm cho một mô hình, dùng trong joblib."""
    # Comment: Hàm hỗ trợ để chạy phân cụm cho một mô hình cụ thể, được gọi song song bởi `joblib`.
    # - `model`: Tên mô hình (KMeans, FuzzyCMeans, v.v.).
    # - `X`: Dữ liệu đầu vào.
    # - `k_range`: Phạm vi số cụm để thử.
    # - `selected_k`: Số cụm tối đa được chọn.
    # - `use_pca`: Trạng thái sử dụng PCA.
    # - `selected_features`: Danh sách feature đã chọn.
    # - `explained_variance`: Tỷ lệ phương sai giải thích (nếu dùng PCA).
    logging.debug(f"Chạy phân cụm cho mô hình {model} với selected_k={selected_k}")
    plots = generate_clustering_plots(X, model, k_range, selected_k, use_pca, selected_features, explained_variance)
    return model, plots

def select_model():
    # Comment: Bắt đầu hàm `select_model` để xử lý việc chọn mô hình phân cụm.
    logging.debug("Bắt đầu hàm select_model")
    
    # Comment: Khởi tạo dictionary `data` để lưu trạng thái và dữ liệu cho giao diện.
    # - `k_range`: Phạm vi số cụm (mặc định từ 2 đến 10).
    # - `models`: Danh sách các mô hình phân cụm.
    # - `selected_k`: Số cụm tối đa được chọn (mặc định là 2).
    # - `plots`: Kết quả phân cụm (biểu đồ, chỉ số CVI, v.v.).
    # - `optimal_k_suggestions`: Gợi ý số cụm tối ưu cho từng mô hình.
    # - `use_pca`: Trạng thái sử dụng PCA.
    # - `explained_variance_ratio`: Tỷ lệ phương sai giải thích (nếu dùng PCA).
    # - `selected_features`: Danh sách feature đã chọn.
    data = {
        'k_range': list(range(2, 11)),
        'models': [],
        'selected_k': 2,
        'plots': {},
        'optimal_k_suggestions': {},
        'use_pca': False,
        'explained_variance_ratio': None,
        'selected_features': []
    }
    
    # Comment: Kiểm tra file dữ liệu đã xử lý (`processed_data.pkl`) có tồn tại không.
    # - Nếu không, thông báo lỗi và chuyển hướng về trang xử lý dữ liệu (`process_data`).
    processed_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')
    if not os.path.exists(processed_data_file):
        logging.error("Không tìm thấy processed_data.pkl")
        flash("Vui lòng xử lý dữ liệu trước.")
        return redirect(url_for('process_data'))
    
    # Comment: Đọc dữ liệu từ file `processed_data.pkl` vào DataFrame `X`.
    # - Ghi log kích thước dữ liệu để debug.
    logging.debug(f"Đọc dữ liệu từ {processed_data_file}")
    X = pd.read_pickle(processed_data_file)
    logging.debug(f"Kích thước dữ liệu: {X.shape}")
    
    # Comment: Kiểm tra xem có sử dụng PCA không.
    # - Đọc file `use_pca.txt` để cập nhật trạng thái `use_pca`.
    use_pca_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt')
    if os.path.exists(use_pca_file):
        with open(use_pca_file, 'r') as f:
            data['use_pca'] = f.read().strip() == 'True'
        logging.debug(f"Use PCA: {data['use_pca']}")
    
    # Comment: Nếu sử dụng PCA, đọc tỷ lệ phương sai giải thích từ file `explained_variance.txt`.
    if data['use_pca']:
        explained_variance_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt')
        if os.path.exists(explained_variance_file):
            with open(explained_variance_file, 'r') as f:
                data['explained_variance_ratio'] = float(f.read().strip())
            logging.debug(f"Explained variance ratio: {data['explained_variance_ratio']}")
    
    # Comment: Đọc danh sách feature đã chọn từ file `selected_features.txt`.
    # - Nếu file tồn tại, cập nhật vào `data['selected_features']`.
    # - Xử lý lỗi nếu đọc file thất bại.
    selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
    if os.path.exists(selected_features_file):
        try:
            with open(selected_features_file, 'r') as f:
                data['selected_features'] = f.read().split(',')
            logging.debug(f"Selected features: {data['selected_features']}")
        except Exception as e:
            logging.error(f"Lỗi đọc selected_features.txt: {str(e)}")
            flash("Lỗi khi đọc danh sách feature đã chọn. Vui lòng chọn lại feature.")
    
    # Comment: Đọc kết quả phân cụm đã lưu từ file `clustering_results.json`.
    # - Cập nhật `data` với danh sách mô hình, số cụm tối đa, kết quả phân cụm, và gợi ý số cụm tối ưu.
    # - Xử lý lỗi nếu đọc file thất bại.
    clustering_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'clustering_results.json')
    if os.path.exists(clustering_results_file):
        try:
            with open(clustering_results_file, 'r') as f:
                clustering_results = json.load(f)
            data['models'] = clustering_results.get('models', [])
            data['selected_k'] = clustering_results.get('selected_k', 2)
            data['plots'] = clustering_results.get('plots', {})
            data['optimal_k_suggestions'] = clustering_results.get('optimal_k_suggestions', {})
            logging.debug(f"Đã đọc clustering_results.json: {data['models']}, selected_k={data['selected_k']}")
        except Exception as e:
            logging.error(f"Lỗi đọc clustering_results.json: {str(e)}")
            flash("Lỗi khi đọc kết quả phân cụm trước đó. Vui lòng chạy lại mô hình.")
    
    if request.method == 'POST':
        # Comment: Xử lý yêu cầu POST khi người dùng gửi form để chạy phân cụm.
        logging.debug("Nhận yêu cầu POST")
        models = request.form.getlist('models')  # Danh sách mô hình từ form
        selected_k = int(request.form.get('k'))  # Số cụm tối đa từ form
        
        # Comment: Kiểm tra tính hợp lệ của `selected_k`.
        # - Yêu cầu `selected_k >= 2` để áp dụng phương pháp khuỷu tay.
        if selected_k < 2:
            logging.error(f"selected_k={selected_k} nhỏ hơn 2")
            flash("Số cụm tối đa (k) phải lớn hơn hoặc bằng 2 để áp dụng phương pháp khuỷu tay.")
            return render_template('select_model.html', data=data)
        logging.debug(f"Selected_k: {selected_k}, Models: {models}")
        
        # Comment: Kiểm tra xem có mô hình nào được chọn không.
        if not models:
            logging.error("Không có mô hình nào được chọn")
            flash("Vui lòng chọn ít nhất một mô hình.")
            return render_template('select_model.html', data=data)
        
        # Comment: Kiểm tra dữ liệu đầu vào trước khi chạy phân cụm.
        # - Lọc các cột số, kiểm tra số cột tối thiểu (>= 2), giá trị NaN, và giá trị vô cực.
        X_numeric = X.select_dtypes(include=[np.number])
        logging.debug(f"Kích thước X_numeric: {X_numeric.shape}")
        if X_numeric.empty or len(X_numeric.columns) < 2:
            logging.error("Dữ liệu không đủ cột số")
            flash("Dữ liệu không chứa đủ cột số (cần ít nhất 2 cột số). Vui lòng kiểm tra và xử lý lại dữ liệu.")
            return render_template('select_model.html', data=data)
        
        if X_numeric.isna().any().any():
            logging.error("Dữ liệu chứa giá trị NaN")
            flash("Dữ liệu chứa giá trị NaN. Vui lòng xử lý dữ liệu trước khi chạy phân cụm.")
            return render_template('select_model.html', data=data)
        
        if np.isinf(X_numeric.values).any():
            logging.error("Dữ liệu chứa giá trị vô cực")
            flash("Dữ liệu chứa giá trị vô cực (inf). Vui lòng xử lý dữ liệu trước khi chạy phân cụm.")
            return render_template('select_model.html', data=data)
        
        # Comment: Cảnh báo nếu số chiều của dữ liệu lớn (>20), có thể làm chậm phân cụm.
        if X_numeric.shape[1] > 20:
            logging.warning(f"Dữ liệu có {X_numeric.shape[1]} chiều, có thể làm chậm phân cụm")
            flash(f"Dữ liệu có {X_numeric.shape[1]} chiều, có thể làm chậm phân cụm. Hãy giảm số chiều bằng PCA hoặc chọn ít feature hơn.")
        
        # Comment: Reset danh sách mô hình và cập nhật `selected_k`.
        data['models'] = []
        data['selected_k'] = selected_k
        
        # Comment: Kiểm tra kết quả phân cụm đã lưu trước khi chạy mới.
        # - Nếu kết quả đã có trong `data['plots']`, sử dụng lại để tránh chạy lại.
        for model in models:
            cache_key = f"{model}_{selected_k}"
            if cache_key in data['plots']:
                logging.debug(f"Sử dụng kết quả đã lưu cho {cache_key}")
                plots = data['plots'][cache_key]
                if 'error' in plots:
                    flash(plots['error'])
                    continue
                data['plots'][model] = plots
                optimal_k, reasoning = suggest_optimal_k(plots, list(range(2, selected_k + 1)))
                data['optimal_k_suggestions'][model] = {'k': optimal_k, 'reasoning': reasoning}
                if 'cvi' in plots:
                    data['models'].append(model)
                continue
        
        # Comment: Chạy phân cụm song song cho các mô hình chưa có kết quả.
        # - Sử dụng `joblib` để chạy đồng thời, tăng tốc độ xử lý.
        logging.debug(f"Chạy phân cụm song song cho các mô hình: {[m for m in models if f'{m}_{selected_k}' not in data['plots']]}")
        results = Parallel(n_jobs=-1)(
            delayed(run_clustering)(
                model, X, range(2, selected_k + 1), selected_k, data['use_pca'],
                data['selected_features'], data['explained_variance_ratio']
            )
            for model in models if f"{model}_{selected_k}" not in data['plots']
        )
        
        # Comment: Xử lý kết quả phân cụm từ các tác vụ song song.
        # - Lưu kết quả vào `data['plots']`, gợi ý số cụm tối ưu, và thêm mô hình thành công vào danh sách.
        for model, plots in results:
            logging.debug(f"Kết quả phân cụm cho {model}: {'error' in plots}")
            if 'error' in plots:
                flash(plots['error'])
                continue
            data['plots'][model] = plots
            optimal_k, reasoning = suggest_optimal_k(plots, list(range(2, selected_k + 1)), use_wiroonsri_starczewski=False)
            data['optimal_k_suggestions'][model] = {'k': optimal_k, 'reasoning': reasoning}
            if 'cvi' in plots:
                data['models'].append(model)
            else:
                flash(f"Mô hình {model} không thể chạy thành công. Vui lòng kiểm tra dữ liệu hoặc thử lại.")
          # Comment: Lưu kết quả phân cụm vào file `clustering_results.json` nếu có mô hình thành công.
        if data['models']:
            clustering_results = {
                'models': data['models'],
                'selected_k': data['selected_k'],
                'plots': data['plots'],
                'optimal_k_suggestions': data['optimal_k_suggestions']
            }
            logging.debug("Lưu kết quả phân cụm vào clustering_results.json")
            with open(clustering_results_file, 'w') as f:
                json.dump(clustering_results, f)
            
            # Xóa cache BCVI cũ khi có sự thay đổi trong mô hình phân cụm
            try:
                clean_bcvi_cache()
                logging.debug("Đã xóa cache BCVI cũ sau khi cập nhật mô hình")
            except Exception as e:
                logging.error(f"Lỗi khi xóa cache BCVI: {str(e)}")
        else:
            logging.warning("Không có mô hình nào chạy thành công")
        
        # Comment: Trả về giao diện `select_model.html` với kết quả phân cụm.
        logging.debug("Hoàn thành xử lý POST, trả về giao diện")
        return render_template('select_model.html', data=data)
    
    # Comment: Trả về giao diện mặc định nếu không có yêu cầu POST.
    logging.debug("Trả về giao diện mặc định (GET)")
    return render_template('select_model.html', data=data)