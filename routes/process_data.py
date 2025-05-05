from flask import render_template, request, redirect, url_for, flash, send_file, current_app
import os
import pandas as pd
import numpy as np
import json
import logging
from utils.pca import perform_pca
from utils.data_processing import handle_null_values

# Comment: Import các thư viện cần thiết.
# - Flask: Dùng để xử lý request, render giao diện, và quản lý ứng dụng web.
# - os, pandas, numpy: Dùng cho xử lý file và dữ liệu.
# - json, logging: Dùng để lưu kết quả và ghi log.
# - perform_pca, handle_null_values: Các hàm từ module utils, phù hợp với pipeline.
# Đánh giá: Import đầy đủ và đúng, không có vấn đề.

DEFAULT_FEATURES = [
    "Age", "Overall", "Potential", "Value", "Wage", "Height", "Weight",
    "Crossing", "Finishing", "HeadingAccuracy", "ShortPassing", "Volleys",
    "Dribbling", "Curve", "FKAccuracy", "LongPassing", "BallControl",
    "Acceleration", "SprintSpeed", "Agility", "Reactions", "Balance",
    "ShotPower", "Jumping", "Stamina", "Strength", "LongShots", "Aggression",
    "Interceptions", "Positioning", "Vision", "Penalties", "Composure",
    "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling",
    "GKKicking", "GKPositioning", "GKReflexes", "Release Clause"
]

# Comment: Định nghĩa danh sách các feature mặc định (DEFAULT_FEATURES).
# - Danh sách này bao gồm các feature liên quan đến cầu thủ bóng đá, phù hợp với dữ liệu bạn đang xử lý.
# - Các feature này được sử dụng khi người dùng chọn tùy chọn "default" trong giao diện.
# Đánh giá: Danh sách hợp lý, bao gồm các feature quan trọng của cầu thủ (tuổi, kỹ năng, giá trị, v.v.).
# Đề xuất: Có thể thêm comment mô tả ý nghĩa của từng feature hoặc nhóm feature (ví dụ: "Value", "Wage" là giá trị tài chính, "GKDiving", "GKHandling" là kỹ năng thủ môn).

def clean_currency(value):
    """Chuyển đổi giá trị tiền tệ (ví dụ: '€500K', '€500M', hoặc '€0') thành số."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace('€', '').replace(',', '')
        if value == '0' or value.strip() == '':
            return 0.0
        if 'K' in value:
            return float(value.replace('K', '')) * 1000
        elif 'M' in value:
            return float(value.replace('M', '')) * 1000000
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan

# Comment: Hàm clean_currency chuyển đổi giá trị tiền tệ dạng chuỗi thành số.
# - Xử lý các trường hợp:
#   + Giá trị NaN: Trả về np.nan.
#   + Chuỗi: Loại bỏ ký tự '€' và ',', chuyển đổi đơn vị 'K' (x1000), 'M' (x1,000,000), hoặc chuyển trực tiếp thành float.
#   + Số (int, float): Chuyển thành float.
#   + Các trường hợp khác: Trả về np.nan.
# Đánh giá: Logic đúng và hợp lý, xử lý tốt các định dạng tiền tệ phổ biến trong dữ liệu cầu thủ.
# Đề xuất: Có thể thêm xử lý các trường hợp lỗi khác (ví dụ: chuỗi không hợp lệ như "€abc") bằng cách kiểm tra thêm trước khi thay thế 'K' hoặc 'M'.

def clean_height(value):
    """Chuyển đổi chiều cao (ví dụ: '5'7"' hoặc '170cm') thành cm."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        try:
            if 'cm' in value.lower():
                return float(value.lower().replace('cm', ''))
            feet, inches = map(int, value.split("'"))
            total_inches = feet * 12 + inches
            return total_inches * 2.54  # 1 inch = 2.54 cm
        except:
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan

# Comment: Hàm clean_height chuyển đổi chiều cao thành đơn vị cm.
# - Xử lý các trường hợp:
#   + Giá trị NaN: Trả về np.nan.
#   + Chuỗi:
#     * Nếu có 'cm': Loại bỏ 'cm' và chuyển thành float.
#     * Nếu dạng "feet'inches" (ví dụ: "5'7""): Chuyển feet và inches thành cm (1 inch = 2.54 cm).
#     * Nếu lỗi (try-except): Trả về np.nan.
#   + Số (int, float): Chuyển thành float.
#   + Các trường hợp khác: Trả về np.nan.
# Đánh giá: Logic đúng, xử lý được cả định dạng chiều cao dạng cm và feet'inches.
# Đề xuất:
# - Có thể thêm kiểm tra giá trị hợp lệ (ví dụ: chiều cao âm hoặc quá lớn) trước khi trả về.
# - Nếu dữ liệu có định dạng chiều cao khác (ví dụ: "1.75m"), nên thêm xử lý cho trường hợp này.

def clean_weight(value):
    """Chuyển đổi cân nặng (ví dụ: '159lbs') thành kg."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        try:
            pounds = float(value.replace('lbs', ''))
            return pounds * 0.453592  # 1 lbs = 0.453592 kg
        except:
            return np.nan
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan

# Comment: Hàm clean_weight chuyển đổi cân nặng thành đơn vị kg.
# - Xử lý các trường hợp:
#   + Giá trị NaN: Trả về np.nan.
#   + Chuỗi: Loại bỏ 'lbs', chuyển thành float, và đổi từ lbs sang kg (1 lbs = 0.453592 kg).
#   + Số (int, float): Chuyển thành float.
#   + Các trường hợp khác: Trả về np.nan.
# Đánh giá: Logic đúng, xử lý tốt định dạng cân nặng dạng lbs.
# Đề xuất:
# - Nếu dữ liệu có cân nặng đã ở dạng kg (ví dụ: "70kg"), nên thêm logic để xử lý trường hợp này:
#   ```python
#   if 'kg' in value.lower():
#       return float(value.lower().replace('kg', ''))
#   ```
# - Có thể thêm kiểm tra giá trị hợp lệ (ví dụ: cân nặng âm hoặc quá lớn).

def process_data():
    data = {
        'features': [],
        'num_features': 0,
        'feature_types': {},
        'selected_features': [],
        'preview_data': None,
        'data_stats': None,
        'pca_result': None,
        'pca_message': '',
        'pca_plot': None,
        'variance_details': [],
        'file_uploaded': False,
        'proceed_to_model': False,
        'data_processed': False
    }
    
    # Comment: Khởi tạo dictionary `data` để lưu trạng thái và dữ liệu cho giao diện.
    # - Các key đại diện cho trạng thái và kết quả của pipeline (danh sách feature, dữ liệu preview, kết quả PCA, v.v.).
    # Đánh giá: Cấu trúc dictionary hợp lý, phù hợp với giao diện Flask để truyền dữ liệu vào template.
    # Đề xuất: Có thể thêm comment giải thích ý nghĩa của từng key trong `data` (ví dụ: 'pca_result' là trạng thái PCA đã chạy hay chưa).

    if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
        flash("Vui lòng tải file dữ liệu.")
        return redirect(url_for('index'))
    
    # Comment: Kiểm tra xem file dữ liệu `data.pkl` đã được tải lên chưa.
    # - Nếu không có file, thông báo lỗi và chuyển hướng về trang `index`.
    # Đánh giá: Logic đúng, đảm bảo dữ liệu đã được tải trước khi xử lý.
    # Đề xuất: Có thể thêm logging để ghi lại trạng thái:
    #   ```python
    #   logging.debug("Kiểm tra file data.pkl")
    #   ```

    df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
    
    # Comment: Đọc dữ liệu từ file `data.pkl` vào DataFrame `df`.
    # Đánh giá: Đúng, sử dụng `pd.read_pickle` để đọc file pickle, phù hợp với dữ liệu đã lưu trước đó.
    # Đề xuất: Có thể thêm kiểm tra lỗi khi đọc file:
    #   ```python
    #   try:
    #       df = pd.read_pickle(...)
    #   except Exception as e:
    #       logging.error(f"Lỗi đọc data.pkl: {str(e)}")
    #       flash("Lỗi khi đọc file dữ liệu.")
    #       return redirect(url_for('index'))
    #   ```

    # Kiểm tra và chuyển đổi các cột đặc biệt thành dạng số nếu chúng tồn tại
    required_columns = ['Value', 'Wage', 'Release Clause', 'Height', 'Weight']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        flash(f"Bộ dữ liệu thiếu các cột: {', '.join(missing_columns)}. Một số tính năng có thể không hoạt động chính xác.")
    
    # Comment: Kiểm tra các cột bắt buộc (`required_columns`) có trong DataFrame không.
    # - Nếu thiếu cột, thông báo cho người dùng nhưng không dừng pipeline.
    # Đánh giá: Logic đúng, cảnh báo người dùng về các cột thiếu mà không làm gián đoạn quá trình xử lý.
    # Đề xuất:
    # - Có thể thêm logging để ghi lại các cột thiếu:
    #   ```python
    #   logging.warning(f"Các cột thiếu: {missing_columns}")
    #   ```
    # - Nếu các cột này bắt buộc cho các bước sau (như PCA), có thể cân nhắc dừng pipeline và yêu cầu người dùng tải lại dữ liệu.

    # Chuyển đổi các cột nếu chúng tồn tại
    if 'Value' in df.columns:
        df['Value'] = df['Value'].apply(clean_currency)
    if 'Wage' in df.columns:
        df['Wage'] = df['Wage'].apply(clean_currency)
    if 'Release Clause' in df.columns:
        df['Release Clause'] = df['Release Clause'].apply(clean_currency)
    if 'Height' in df.columns:
        df['Height'] = df['Height'].apply(clean_height)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].apply(clean_weight)
    
    # Comment: Chuyển đổi các cột đặc biệt ('Value', 'Wage', 'Release Clause', 'Height', 'Weight') thành dạng số.
    # - Sử dụng các hàm `clean_currency`, `clean_height`, `clean_weight` đã định nghĩa ở trên.
    # Đánh giá: Logic đúng, áp dụng các hàm chuyển đổi phù hợp để chuẩn hóa dữ liệu.
    # Đề xuất:
    # - Có thể thêm logging để ghi lại số lượng giá trị NaN sau khi chuyển đổi:
    #   ```python
    #   logging.debug(f"Số giá trị NaN trong 'Value' sau chuyển đổi: {df['Value'].isna().sum()}")
    #   ```
    # - Có thể kiểm tra tỷ lệ giá trị NaN sau chuyển đổi, nếu quá cao (ví dụ: >50%), thông báo cho người dùng:
    #   ```python
    #   if df['Value'].isna().mean() > 0.5:
    #       flash("Cột 'Value' có quá nhiều giá trị NaN sau chuyển đổi, kết quả có thể không chính xác.")
    #   ```

    # Kiểm tra và chuyển các cột sang kiểu số nếu có thể
    for col in df.columns:
        if col in ['Value', 'Wage', 'Release Clause', 'Height', 'Weight']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Comment: Chuyển các cột đặc biệt sang kiểu số, với `errors='coerce'` để ép các giá trị không hợp lệ thành NaN.
    # Đánh giá: Logic đúng, đảm bảo các cột này ở dạng số để có thể sử dụng trong PCA và các bước sau.
    # Đề xuất:
    # - Đoạn code này có thể dư thừa vì các hàm `clean_currency`, `clean_height`, `clean_weight` đã trả về float hoặc NaN.
    # - Có thể bỏ đoạn này để giảm thời gian xử lý, vì việc chuyển đổi đã được thực hiện ở bước trên.
    # - Nếu giữ lại, nên thêm logging để kiểm tra số lượng giá trị NaN:
    #   ```python
    #   logging.debug(f"Số giá trị NaN trong '{col}' sau pd.to_numeric: {df[col].isna().sum()}")
    #   ```

    data['file_uploaded'] = True
    data['features'] = df.columns.tolist()
    data['num_features'] = len(df.columns)
    data['feature_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    data['preview_data'] = df.head(5).to_dict(orient='records')
    data['data_stats'] = df.describe().to_dict()
    
    # Comment: Cập nhật dictionary `data` với thông tin từ DataFrame.
    # - `file_uploaded`: Đặt thành True để chỉ ra dữ liệu đã được tải.
    # - `features`: Danh sách các cột trong DataFrame.
    # - `num_features`: Số lượng cột.
    # - `feature_types`: Kiểu dữ liệu của từng cột.
    # - `preview_data`: 5 dòng đầu tiên của dữ liệu để hiển thị trên giao diện.
    # - `data_stats`: Thống kê mô tả (describe) của dữ liệu.
    # Đánh giá: Logic đúng, cung cấp đầy đủ thông tin cần thiết cho giao diện.
    # Đề xuất:
    # - Có thể thêm logging để ghi lại thông tin:
    #   ```python
    #   logging.debug(f"Số feature: {data['num_features']}")
    #   logging.debug(f"Kiểu dữ liệu: {data['feature_types']}")
    #   ```
    # - `df.describe()` có thể chậm nếu dữ liệu lớn, nên cân nhắc chỉ tính thống kê cho các cột số:
    #   ```python
    #   data['data_stats'] = df.select_dtypes(include=[np.number]).describe().to_dict()
    #   ```

    pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
    if os.path.exists(pca_results_file):
        try:
            with open(pca_results_file, 'r') as f:
                pca_results = json.load(f)
            data['pca_result'] = True
            data['pca_message'] = pca_results.get('pca_message', '')
            data['pca_plot'] = pca_results.get('pca_plot', None)
            data['variance_details'] = pca_results.get('variance_details', [])
        except Exception as e:
            logging.error(f"Error reading pca_results.json: {str(e)}")
            flash("Lỗi khi đọc kết quả PCA trước đó. Vui lòng chạy lại phân tích.")
    
    # Comment: Kiểm tra và đọc kết quả PCA đã lưu trước đó (`pca_results.json`).
    # - Nếu file tồn tại, đọc dữ liệu và cập nhật `data` với kết quả PCA (trạng thái, thông báo, biểu đồ, chi tiết phương sai).
    # - Xử lý lỗi nếu đọc file thất bại.
    # Đánh giá: Logic đúng, cho phép tái sử dụng kết quả PCA đã chạy trước đó, tránh chạy lại PCA nếu không cần thiết.
    # Đề xuất:
    # - Có thể thêm logging khi file không tồn tại:
    #   ```python
    #   else:
    #       logging.debug("Không tìm thấy pca_results.json")
    #   ```
    # - Nếu dữ liệu trong `pca_results.json` không còn hợp lệ (ví dụ: do thay đổi feature), nên kiểm tra tính hợp lệ trước khi sử dụng.

    selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
    if os.path.exists(selected_features_file):
        try:
            with open(selected_features_file, 'r') as f:
                data['selected_features'] = f.read().split(',')
        except Exception as e:
            logging.error(f"Error reading selected_features.txt: {str(e)}")
            flash("Lỗi khi đọc danh sách feature đã chọn. Vui lòng chọn lại feature.")
    
    # Comment: Kiểm tra và đọc danh sách feature đã chọn trước đó (`selected_features.txt`).
    # - Nếu file tồn tại, đọc danh sách feature và cập nhật vào `data`.
    # - Xử lý lỗi nếu đọc file thất bại.
    # Đánh giá: Logic đúng, tái sử dụng feature đã chọn để giữ trạng thái pipeline.
    # Đề xuất: Tương tự như trên, thêm logging khi file không tồn tại:
    #   ```python
    #   else:
    #       logging.debug("Không tìm thấy selected_features.txt")
    #   ```

    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')):
        data['proceed_to_model'] = True
    
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl')):
        data['data_processed'] = True
    
    # Comment: Kiểm tra trạng thái pipeline dựa trên sự tồn tại của các file tạm.
    # - `processed_data.pkl`: Nếu tồn tại, cho phép người dùng tiến hành bước chọn mô hình (`proceed_to_model`).
    # - `temp_data.pkl`: Nếu tồn tại, dữ liệu đã được xử lý (`data_processed`).
    # Đánh giá: Logic đúng, sử dụng file tạm để quản lý trạng thái pipeline.
    # Đề xuất: Có thể thêm logging để ghi lại trạng thái:
    #   ```python
    #   logging.debug(f"processed_data.pkl tồn tại: {data['proceed_to_model']}")
    #   logging.debug(f"temp_data.pkl tồn tại: {data['data_processed']}")
    #   ```

    if request.method == 'POST':
        if 'select_features' in request.form:
            # Xóa các file tạm trước khi xử lý feature mới
            temp_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl')
            pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
            processed_data_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')
            if os.path.exists(temp_data_file):
                os.remove(temp_data_file)
            if os.path.exists(pca_results_file):
                os.remove(pca_results_file)
            if os.path.exists(processed_data_file):
                os.remove(processed_data_file)
            if os.path.exists(selected_features_file):
                os.remove(selected_features_file)
            
            # Comment: Xử lý yêu cầu POST khi người dùng chọn feature (`select_features`).
            # - Xóa các file tạm (`temp_data.pkl`, `pca_results.json`, `processed_data.pkl`, `selected_features.txt`) để đảm bảo pipeline bắt đầu lại từ đầu.
            # Đánh giá: Logic đúng, xóa các file tạm tránh xung đột khi người dùng chọn feature mới.
            # Đề xuất:
            # - Có thể thêm logging để ghi lại hành động xóa file:
            #   ```python
            #   logging.debug("Xóa file tạm trước khi xử lý feature mới")
            #   ```
            # - Nên xử lý lỗi khi xóa file:
            #   ```python
            #   try:
            #       if os.path.exists(temp_data_file):
            #           os.remove(temp_data_file)
            #           logging.debug("Đã xóa temp_data.pkl")
            #   except Exception as e:
            #       logging.error(f"Lỗi xóa file tạm {temp_data_file}: {str(e)}")
            #   ```

            use_default = request.form.get('feature_option') == 'default'
            
            if use_default:
                selected_features = [col for col in DEFAULT_FEATURES if col in df.columns]
                if len(selected_features) < 2:
                    flash("Feature mặc định không đủ (ít nhất 2 feature). Vui lòng chọn tùy chỉnh.")
                    return render_template('process_data.html', data=data)
            else:
                selected_features = request.form.getlist('features')
                if len(selected_features) < 2:
                    flash("Vui lòng chọn ít nhất 2 feature.")
                    return render_template('process_data.html', data=data)
            
            # Comment: Xử lý lựa chọn feature từ người dùng.
            # - Nếu chọn "default", sử dụng `DEFAULT_FEATURES` và lọc các cột có trong DataFrame.
            # - Nếu chọn tùy chỉnh, lấy danh sách feature từ form (`request.form.getlist('features')`).
            # - Kiểm tra số feature tối thiểu (>= 2) để đảm bảo PCA và phân cụm có thể chạy.
            # Đánh giá: Logic đúng, xử lý tốt cả hai trường hợp (default và tùy chỉnh), đảm bảo số feature tối thiểu.
            # Đề xuất:
            # - Có thể thêm logging để ghi lại danh sách feature đã chọn:
            #   ```python
            #   logging.debug(f"Feature đã chọn: {selected_features}")
            #   ```
            # - Nếu `selected_features` không chứa cột số, nên kiểm tra sớm để tránh lỗi sau này:
            #   ```python
            #   numeric_selected = [col for col in selected_features if col in df.select_dtypes(include=[np.number]).columns]
            #   if len(numeric_selected) < 2:
            #       flash("Không đủ cột số trong feature đã chọn (ít nhất 2 cột).")
            #       return render_template('process_data.html', data=data)
            #   ```

            # Chọn dữ liệu với các feature đã chọn
            temp_df = df[selected_features].copy()
            
            # Kiểm tra và chuyển đổi các cột sang dạng số nếu có thể
            for col in temp_df.columns:
                if temp_df[col].dtype == 'object':
                    if col.lower() in ['value', 'wage', 'release clause']:
                        temp_df[col] = temp_df[col].apply(clean_currency)
                    elif col.lower() == 'height':
                        temp_df[col] = temp_df[col].apply(clean_height)
                    elif col.lower() == 'weight':
                        temp_df[col] = temp_df[col].apply(clean_weight)
            
            # Comment: Tạo DataFrame tạm `temp_df` với các feature đã chọn.
            # - Chuyển đổi các cột dạng chuỗi trong `temp_df` thành số bằng cách áp dụng các hàm `clean_currency`, `clean_height`, `clean_weight`.
            # Đánh giá: Logic đúng, nhưng có thể dư thừa vì các cột đặc biệt đã được chuyển đổi ở bước trên (`df` gốc).
            # Đề xuất:
            # - Bỏ đoạn này nếu các cột đã được chuyển đổi trước đó (ở bước xử lý `df`).
            # - Nếu giữ, nên kiểm tra lại tỷ lệ NaN sau chuyển đổi:
            #   ```python
            #   logging.debug(f"Số giá trị NaN trong '{col}' sau clean: {temp_df[col].isna().sum()}")
            #   ```

            # Chuyển đổi tất cả các cột sang kiểu số
            for col in temp_df.columns:
                try:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                except:
                    pass
            
            # Comment: Chuyển tất cả các cột trong `temp_df` sang kiểu số, với `errors='coerce'` để ép các giá trị không hợp lệ thành NaN.
            # Đánh giá: Logic đúng, nhưng tương tự như trên, có thể dư thừa vì đã chuyển đổi ở bước trước.
            # Đề xuất:
            # - Bỏ đoạn này để giảm thời gian xử lý.
            # - Nếu giữ, nên xử lý lỗi cụ thể hơn thay vì dùng `pass`:
            #   ```python
            #   except Exception as e:
            #       logging.error(f"Lỗi chuyển đổi cột '{col}' sang số: {str(e)}")
            #   ```

            # Kiểm tra số lượng cột số sau khi chuyển đổi
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                non_numeric_cols = [col for col in selected_features if col not in numeric_cols]
                flash(f"Không đủ cột số để chạy mô hình (ít nhất 2 cột). Các cột không phải số sau khi chuyển đổi: {', '.join(non_numeric_cols)}")
                data['selected_features'] = []
                data['data_processed'] = False
                return render_template('process_data.html', data=data)
            
            # Comment: Kiểm tra số lượng cột số trong `temp_df` sau khi chuyển đổi.
            # - Nếu ít hơn 2 cột số, thông báo lỗi và trả về giao diện.
            # Đánh giá: Logic đúng, đảm bảo dữ liệu có đủ cột số để chạy PCA và phân cụm.
            # Đề xuất:
            # - Có thể thêm logging để ghi lại danh sách cột số và không số:
            #   ```python
            #   logging.debug(f"Cột số: {numeric_cols}")
            #   logging.debug(f"Cột không phải số: {non_numeric_cols}")
            #   ```
            # - Nếu không có cột số nào, nên thông báo rõ ràng hơn:
            #   ```python
            #   if not numeric_cols:
            #       flash("Không có cột số nào trong feature đã chọn.")
            #   ```

            # Lưu DataFrame vào file tạm (không chuẩn hóa tại đây)
            temp_df.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl'))
            
            data['selected_features'] = selected_features
            data['data_processed'] = True
            
            with open(selected_features_file, 'w') as f:
                f.write(','.join(selected_features))
            
            return render_template('process_data.html', data=data)
        
        # Comment: Lưu DataFrame `temp_df` vào file tạm `temp_data.pkl` và cập nhật trạng thái.
        # - Lưu danh sách feature đã chọn vào `selected_features.txt`.
        # - Cập nhật `data['selected_features']` và `data['data_processed']`.
        # - Trả về giao diện `process_data.html` với dữ liệu cập nhật.
        # Đánh giá: Logic đúng, lưu dữ liệu tạm và trạng thái để sử dụng ở các bước sau (PCA, phân cụm).
        # Đề xuất:
        # - Thêm logging để ghi lại hành động:
        #   ```python
        #   logging.debug("Đã lưu temp_data.pkl và selected_features.txt")
        #   ```
        # - Xử lý lỗi khi lưu file:
        #   ```python
        #   try:
        #       temp_df.to_pickle(...)
        #   except Exception as e:
        #       logging.error(f"Lỗi lưu temp_data.pkl: {str(e)}")
        #       flash("Lỗi khi lưu dữ liệu tạm.")
        #       return render_template('process_data.html', data=data)
        #   ```

        if 'run_analysis' in request.form:
            # Xóa kết quả PCA cũ trước khi chạy PCA mới
            pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
            if os.path.exists(pca_results_file):
                os.remove(pca_results_file)
            
            selected_features = request.form.get('selected_features').split(',')
            process_method = request.form.get('process_method')
            
            data['selected_features'] = selected_features
            
            # Sử dụng DataFrame từ file tạm
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl')):
                flash("Dữ liệu không tồn tại. Vui lòng chọn lại feature.")
                data['selected_features'] = []
                data['data_processed'] = False
                return render_template('process_data.html', data=data)
            
            X = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl'))
            
            # Kiểm tra lại selected_features để đảm bảo chỉ chứa các cột có trong X
            selected_features = [col for col in selected_features if col in X.columns]
            if len(selected_features) < 2:
                flash("Không đủ cột số để chạy mô hình (ít nhất 2 cột). Vui lòng chọn lại feature.")
                data['selected_features'] = []
                data['data_processed'] = False
                return render_template('process_data.html', data=data)
            
            # Comment: Xử lý yêu cầu POST khi người dùng chạy phân tích (`run_analysis`).
            # - Xóa file `pca_results.json` để chạy lại PCA từ đầu.
            # - Lấy danh sách feature và phương pháp xử lý (`process_method`) từ form.
            # - Đọc dữ liệu từ file tạm `temp_data.pkl`.
            # - Kiểm tra số feature tối thiểu (>= 2).
            # Đánh giá: Logic đúng, xử lý yêu cầu chạy phân tích và kiểm tra dữ liệu đầu vào.
            # Đề xuất:
            # - Thêm logging để ghi lại hành động:
            #   ```python
            #   logging.debug(f"Process method: {process_method}, Selected features: {selected_features}")
            #   ```
            # - Xử lý lỗi khi đọc file `temp_data.pkl`:
            #   ```python
            #   try:
            #       X = pd.read_pickle(...)
            #   except Exception as e:
            #       logging.error(f"Lỗi đọc temp_data.pkl: {str(e)}")
            #       flash("Lỗi khi đọc dữ liệu tạm.")
            #       return render_template('process_data.html', data=data)
            #   ```

            if process_method == 'pca':
                try:
                    explained_variance = float(request.form.get('explained_variance')) / 100
                    X_pca, pca_plot, pca_message, variance_explained_pc1_pc2, variance_details = perform_pca(X, selected_features, explained_variance)
                    data['pca_result'] = X_pca is not None
                    data['pca_message'] = pca_message
                    data['pca_plot'] = pca_plot
                    data['variance_details'] = variance_details if variance_details else []
                    if X_pca is not None:
                        pd.DataFrame(X_pca).to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl'))
                        with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt'), 'w') as f:
                            f.write('True')
                        with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'explained_variance.txt'), 'w') as f:
                            f.write(str(variance_explained_pc1_pc2))
                        pca_results = {
                            'pca_message': pca_message,
                            'pca_plot': pca_plot,
                            'variance_details': variance_details
                        }
                        with open(pca_results_file, 'w') as f:
                            json.dump(pca_results, f)
                        data['proceed_to_model'] = True
                        return render_template('process_data.html', data=data)
                    else:
                        flash(pca_message)
                        data['selected_features'] = []
                        data['data_processed'] = False
                        return render_template('process_data.html', data=data)
                except ValueError:
                    flash("Tỷ lệ phương sai không hợp lệ (1-100).")
                    data['selected_features'] = []
                    data['data_processed'] = False
                    return render_template('process_data.html', data=data)
            
            # Comment: Xử lý khi người dùng chọn phương pháp `pca`.
            # - Lấy tỷ lệ phương sai giải thích (`explained_variance`) từ form và chuyển thành tỷ lệ (0-1).
            # - Gọi hàm `perform_pca` để chạy PCA.
            # - Cập nhật kết quả PCA vào `data` và lưu vào file tạm (`processed_data.pkl`, `pca_results.json`, v.v.).
            # - Xử lý lỗi nếu tỷ lệ phương sai không hợp lệ.
            # Đánh giá: Logic đúng, xử lý PCA và lưu kết quả hợp lý.
            # Đề xuất:
            # - Thêm kiểm tra giá trị `explained_variance` trước khi chuyển đổi:
            #   ```python
            #   if not (0 < explained_variance <= 100):
            #       flash("Tỷ lệ phương sai phải nằm trong khoảng 1-100.")
            #       return render_template('process_data.html', data=data)
            #   ```
            # - Thêm logging để ghi lại kết quả PCA:
            #   ```python
            #   logging.debug(f"PCA completed: {pca_message}")
            #   ```

            else:
                X = X.select_dtypes(include=[np.number]).dropna()
                if X.empty or len(X.columns) < 2:
                    numeric_cols = X.columns.tolist()
                    non_numeric_cols = [col for col in selected_features if col not in numeric_cols]
                    flash(f"Không đủ cột số để chạy mô hình (ít nhất 2 cột). Các cột không phải số: {', '.join(non_numeric_cols)}")
                    data['selected_features'] = []
                    data['data_processed'] = False
                    return render_template('process_data.html', data=data)
                
                X.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl'))
                with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt'), 'w') as f:
                    f.write('False')
                data['proceed_to_model'] = True
                return redirect(url_for('select_model'))
    
    # Comment: Xử lý khi người dùng không chọn PCA (`process_method != 'pca'`).
    # - Lọc các cột số và loại bỏ hàng chứa NaN.
    # - Kiểm tra số cột số tối thiểu (>= 2).
    # - Lưu dữ liệu vào `processed_data.pkl`, đặt `use_pca=False`, và chuyển hướng đến bước chọn mô hình.
    # Đánh giá: Logic đúng, cho phép người dùng bỏ qua PCA và tiến hành phân cụm trực tiếp.
    # Đề xuất:
    # - Thêm logging để ghi lại trạng thái:
    #   ```python
    #   logging.debug("Không sử dụng PCA, chuyển thẳng đến bước chọn mô hình")
    #   ```
    # - Có thể thêm kiểm tra số lượng hàng sau khi loại bỏ NaN:
    #   ```python
    #   if len(X) < 2:
    #       flash("Dữ liệu sau khi loại bỏ NaN không đủ (ít nhất 2 hàng).")
    #       return render_template('process_data.html', data=data)
    #   ```

    return render_template('process_data.html', data=data)

# Comment: Trả về giao diện mặc định nếu không có yêu cầu POST.
# Đánh giá: Logic đúng, hiển thị giao diện với dữ liệu hiện tại.
# Đề xuất: Có thể thêm logging:
#   ```python
#   logging.debug("Trả về giao diện process_data.html (GET)")
#   ```

def download_pca():
    pca_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_result.csv')
    if os.path.exists(pca_file):
        return send_file(pca_file, as_attachment=True)
    else:
        flash("Chưa có kết quả PCA.")
        return redirect(url_for('process_data'))

# Comment: Hàm download_pca cho phép người dùng tải kết quả PCA (`pca_result.csv`).
# - Nếu file tồn tại, gửi file về client dưới dạng tải xuống.
# - Nếu không, thông báo lỗi và chuyển hướng về trang `process_data`.
# Đánh giá: Logic đúng, cung cấp tính năng tải kết quả PCA.
# Đề xuất:
# - Thêm logging để ghi lại hành động:
#   ```python
#   logging.debug("Tải file pca_result.csv")
#   ```
# - Xử lý lỗi khi gửi file:
#   ```python
#   try:
#       return send_file(pca_file, as_attachment=True)
#   except Exception as e:
#       logging.error(f"Lỗi tải file pca_result.csv: {str(e)}")
#       flash("Lỗi khi tải kết quả PCA.")
#       return redirect(url_for('process_data'))
#   ```