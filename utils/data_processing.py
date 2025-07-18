import pandas as pd
import numpy as np
import re


# Comment: Import các thư viện cần thiết.
# - pandas, numpy: Dùng cho xử lý dữ liệu.
# - re: Dùng cho chuyển đổi tiền tệ.

def parse_money(value):
    # Chuyển đổi giá trị tiền tệ kiểu '€110.5M', '€1.2B', '€500K' thành số thực
    if pd.isna(value):
        return np.nan
    
    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    
    try:
        # Xóa ký hiệu tiền tệ, dấu phẩy và khoảng trắng
        clean_value = value.replace('€', '').replace(',', '').strip()
        
        # Xử lý hậu tố M (triệu), K (nghìn), B (tỷ)
        if clean_value.endswith('M'):
            return float(clean_value[:-1]) * 1e6
        elif clean_value.endswith('K'):
            return float(clean_value[:-1]) * 1e3
        elif clean_value.endswith('B'):
            return float(clean_value[:-1]) * 1e9
        else:
            # Thử chuyển đổi trực tiếp nếu không có hậu tố
            return float(clean_value)
    except Exception:
        return np.nan

def convert_money_columns(df):
    # Danh sách cột tiền tệ phổ biến - chỉ chuyển đổi các cột này
    money_columns = ['Value', 'Wage', 'Release Clause']
    
    # Chỉ chuyển đổi các cột tiền tệ đã biết
    for col in money_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_money)
    
    return df

def allowed_file(filename):
    # Comment: Hàm kiểm tra định dạng file có được phép tải lên không.
    # - Chỉ cho phép các file có đuôi `.csv` hoặc `.xlsx`.
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def read_data(file_path):
    # Comment: Hàm đọc dữ liệu từ file và trả về DataFrame.
    # - Hỗ trợ hai định dạng: `.csv` và `.xlsx`.
    # - Trả về DataFrame và thông báo lỗi (nếu có).
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        df = convert_money_columns(df)
        return df, None
    except Exception as e:
        return None, str(e)

def handle_null_values(df):
    # Comment: Hàm xử lý giá trị null trong DataFrame.
    # - Xóa các cột có tỷ lệ null > 50%.
    # - Điền giá trị null: dùng trung bình cho cột số, mode cho cột không phải số.
    dropped_columns = []
    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        if null_ratio > 0.5:
            # Comment: Nếu tỷ lệ null > 50%, xóa cột và thêm vào danh sách `dropped_columns`.
            dropped_columns.append(col)
            df = df.drop(columns=[col])
        else:
            # Comment: Điền giá trị null.
            # - Cột số: Dùng trung bình.
            # - Cột không phải số: Dùng mode (giá trị xuất hiện nhiều nhất).
            if np.issubdtype(df[col].dtype, np.number):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
                df[col].fillna(mode_value, inplace=True)
    return df, dropped_columns