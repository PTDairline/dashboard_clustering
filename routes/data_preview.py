from flask import render_template, flash, redirect, url_for, request, current_app
import os
import pandas as pd

# Comment: Import các thư viện cần thiết.
# - Flask: Dùng để xử lý request, render giao diện, và quản lý ứng dụng web.
# - os, pandas: Dùng cho xử lý file và dữ liệu.

def data_preview():
    # Comment: Khởi tạo dictionary `data` để lưu trạng thái và dữ liệu cho giao diện.
    # - `features`: Danh sách các cột trong dữ liệu.
    # - `num_features`: Số lượng cột.
    # - `feature_types`: Kiểu dữ liệu của từng cột.
    # - `preview_data`: 5 dòng đầu tiên của dữ liệu.
    # - `data_stats`: Thống kê mô tả của dữ liệu.
    # - `file_uploaded`: Trạng thái file đã được tải hay chưa.
    # - `proceed_to_process`: Trạng thái cho phép tiến hành bước xử lý dữ liệu tiếp theo.
    data = {
        'features': [],
        'num_features': 0,
        'feature_types': {},
        'preview_data': None,
        'data_stats': None,
        'file_uploaded': False,
        'proceed_to_process': False
    }
    
    # Comment: Kiểm tra xem file dữ liệu (`data.pkl`) đã được tải lên chưa.
    # - Nếu có, đọc dữ liệu và cập nhật thông tin vào `data`.
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
        # Comment: Đọc dữ liệu từ file `data.pkl` vào DataFrame `df`.
        df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
        
        # Comment: Cập nhật trạng thái và thông tin dữ liệu.
        # - Đặt `file_uploaded` và `proceed_to_process` thành True.
        # - Lưu danh sách cột, số lượng cột, kiểu dữ liệu, dữ liệu preview, và thống kê mô tả.
        data['file_uploaded'] = True
        data['features'] = df.columns.tolist()
        data['num_features'] = len(df.columns)
        data['feature_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        data['preview_data'] = df.head(5).to_dict(orient='records')
        data['data_stats'] = df.describe().to_dict()
        data['proceed_to_process'] = True  # Sửa cú pháp từ ['proceed_to_process': True] thành ['proceed_to_process'] = True
    
    # Comment: Trả về giao diện `data_preview.html` với thông tin dữ liệu.
    # - Nếu file không tồn tại, trả về giao diện với `data` mặc định (không có dữ liệu).
    return render_template('data_preview.html', data=data)