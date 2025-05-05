import pandas as pd
import numpy as np

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def read_data(file_path):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

def handle_null_values(df):
    dropped_columns = []
    for col in df.columns:
        null_ratio = df[col].isnull().mean()
        if null_ratio > 0.5:
            dropped_columns.append(col)
            df = df.drop(columns=[col])
        else:
            if np.issubdtype(df[col].dtype, np.number):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else np.nan
                df[col].fillna(mode_value, inplace=True)
    return df, dropped_columns