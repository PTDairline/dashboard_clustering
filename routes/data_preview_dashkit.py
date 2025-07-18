from flask import render_template, flash, redirect, url_for, request, current_app
import os
import pandas as pd

def data_preview_dashkit():
    # Khởi tạo dictionary `data` để lưu trạng thái và dữ liệu cho giao diện.
    data = {
        'features': [],
        'num_features': 0,
        'feature_types': {},
        'preview_data': None,
        'data_stats': None,
        'file_uploaded': False,
        'proceed_to_process': False,
        'numerical_features': [],
        'categorical_features': [],
        'numerical_count': 0,
        'categorical_count': 0,
        'datetime_count': 0,
        'other_count': 0,
        'missing_values': 0
    }
    
    # Kiểm tra xem file dữ liệu đã được tải lên chưa
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
        try:
            # Đọc dữ liệu từ file `data.pkl` vào DataFrame `df`
            df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
            
            # Cập nhật thông tin vào `data`
            data['features'] = df.columns.tolist()
            data['num_features'] = len(data['features'])
            data['feature_types'] = {col: str(df[col].dtype) for col in df.columns}
            data['preview_data'] = df.head().to_dict('records')
            data['file_uploaded'] = True
            data['num_rows'] = df.shape[0]
            
            # Tính số giá trị thiếu
            data['missing_values'] = df.isna().sum().sum()
            
            # Phân loại các cột theo loại dữ liệu
            numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            data['numerical_features'] = numerical_columns
            data['categorical_features'] = categorical_columns
            data['numerical_count'] = len(numerical_columns)
            data['categorical_count'] = len(categorical_columns)
            data['datetime_count'] = len(datetime_columns)
            data['other_count'] = df.shape[1] - len(numerical_columns) - len(categorical_columns) - len(datetime_columns)
            
            # Tính toán thống kê mô tả chỉ cho các cột số
            try:
                data['data_stats'] = df[numerical_columns].describe().to_dict()
            except Exception as e:
                flash(f"Lỗi khi tính thống kê: {str(e)}")
                data['data_stats'] = None
            
            # Kiểm tra xem có thể tiến hành bước xử lý dữ liệu tiếp theo không
            data['proceed_to_process'] = True
            
        except Exception as e:
            flash(f"Lỗi khi tải dữ liệu: {str(e)}")
            return render_template('data_preview_dashkit.html', data=data)
    
    else:
        flash("Không tìm thấy dữ liệu. Vui lòng tải file lên trước.")
        return render_template('data_preview_dashkit.html', data=data)
    
    # Render với template Dashkit
    return render_template('data_preview_dashkit.html', data=data)