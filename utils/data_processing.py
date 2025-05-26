import pandas as pd
import numpy as np

# Comment: Import các thư viện cần thiết.
# - pandas, numpy: Dùng cho xử lý dữ liệu.

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