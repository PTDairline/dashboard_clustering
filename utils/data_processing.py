import pandas as pd
import numpy as np
import re
import logging

def parse_money(value):
    """Chuyển đổi giá trị tiền tệ kiểu '€110.5M', '€1.2B', '€500K' thành số thực."""
    if pd.isna(value):
        return np.nan
    
    if not isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    
    try:
        clean_value = value.replace('€', '').replace(',', '').strip()
        if clean_value.endswith('M'):
            return float(clean_value[:-1]) * 1e6
        elif clean_value.endswith('K'):
            return float(clean_value[:-1]) * 1e3
        elif clean_value.endswith('B'):
            return float(clean_value[:-1]) * 1e9
        else:
            return float(clean_value)
    except Exception:
        logging.error(f"Cannot convert money value '{value}' to float")
        return np.nan

def convert_height_to_cm(value):
    """Chuyển đổi chiều cao dạng feet-inches (ví dụ '5'7') sang cm."""
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    s = str(value).strip()
    match = re.match(r"^(\d+)'(\d+)?$", s)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2)) if match.group(2) else 0
        return round((feet * 12 + inches) * 2.54, 1)
    try:
        return float(s)
    except Exception:
        logging.error(f"Cannot convert height '{s}' to cm")
        return np.nan

def convert_weight_to_kg(value):
    """Chuyển đổi cân nặng dạng số hoặc có đơn vị lbs sang kg."""
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    s = str(value).strip()
    match = re.match(r"^(\d+)(?:\s?lbs)?$", s, re.IGNORECASE)
    if match:
        lbs = int(match.group(1))
        return round(lbs * 0.453592, 1)
    numbers = re.findall(r'\d+', s)
    if numbers:
        lbs = int(numbers[0])
        logging.warning(f"Extracted first number '{lbs}' from invalid weight '{s}'")
        return round(lbs * 0.453592, 1)
    try:
        return float(s)
    except Exception:
        logging.error(f"Cannot convert weight '{s}' to kg")
        return np.nan

def allowed_file(filename):
    """Kiểm tra định dạng file có được phép tải lên không."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def read_data(file_path):
    """Đọc dữ liệu từ file và trả về DataFrame."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        
        # Chuyển đổi cột tiền tệ
        money_columns = ['Value', 'Wage', 'Release Clause']
        for col in money_columns:
            if col in df.columns:
                df[col] = df[col].apply(parse_money)
        
        # Chuyển đổi cột Height và Weight
        if 'Height' in df.columns:
            df['Height'] = df['Height'].apply(convert_height_to_cm)
        if 'Weight' in df.columns:
            df['Weight'] = df['Weight'].apply(convert_weight_to_kg)
        
        # Loại bỏ các hàng có NaN trong Height hoặc Weight
        original_len = len(df)
        if 'Height' in df.columns or 'Weight' in df.columns:
            df = df.dropna(subset=['Height', 'Weight'], how='any')
            dropped_rows = original_len - len(df)
            if dropped_rows > 0:
                logging.warning(f"Dropped {dropped_rows} rows due to NaN in Height or Weight")
        
        return df, None
    except Exception as e:
        logging.error(f"Error reading data: {str(e)}")
        return None, str(e)

def handle_null_values(df):
    """Xử lý giá trị null trong DataFrame."""
    dropped_columns = []
    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        if null_ratio > 0.5:
            dropped_columns.append(col)
            df = df.drop(columns=[col])
        else:
            if np.issubdtype(df[col].dtype, np.number):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
                df[col] = df[col].fillna(mode_value)
    return df, dropped_columns