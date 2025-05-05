from flask import render_template, request, redirect, url_for, flash, send_file, current_app
import os
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import StandardScaler
from utils.pca import perform_pca
from utils.data_processing import handle_null_values

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
    
    if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
        flash("Vui lòng tải file dữ liệu.")
        return redirect(url_for('index'))
    
    df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
    
    # Kiểm tra và chuyển đổi các cột đặc biệt thành dạng số nếu chúng tồn tại
    required_columns = ['Value', 'Wage', 'Release Clause', 'Height', 'Weight']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        flash(f"Bộ dữ liệu thiếu các cột: {', '.join(missing_columns)}. Một số tính năng có thể không hoạt động chính xác.")
    
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
    
    # Kiểm tra và chuyển các cột sang kiểu số nếu có thể
    for col in df.columns:
        if col in ['Value', 'Wage', 'Release Clause', 'Height', 'Weight']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    data['file_uploaded'] = True
    data['features'] = df.columns.tolist()
    data['num_features'] = len(df.columns)
    data['feature_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    data['preview_data'] = df.head(5).to_dict(orient='records')
    data['data_stats'] = df.describe().to_dict()
    
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
    
    selected_features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
    if os.path.exists(selected_features_file):
        try:
            with open(selected_features_file, 'r') as f:
                data['selected_features'] = f.read().split(',')
        except Exception as e:
            logging.error(f"Error reading selected_features.txt: {str(e)}")
            flash("Lỗi khi đọc danh sách feature đã chọn. Vui lòng chọn lại feature.")
    
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'processed_data.pkl')):
        data['proceed_to_model'] = True
    
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl')):
        data['data_processed'] = True
    
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
            
            # Chuyển đổi tất cả các cột sang kiểu số
            for col in temp_df.columns:
                try:
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                except:
                    pass
            
            # Kiểm tra số lượng cột số sau khi chuyển đổi
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                non_numeric_cols = [col for col in selected_features if col not in numeric_cols]
                flash(f"Không đủ cột số để chạy mô hình (ít nhất 2 cột). Các cột không phải số sau khi chuyển đổi: {', '.join(non_numeric_cols)}")
                data['selected_features'] = []
                data['data_processed'] = False
                return render_template('process_data.html', data=data)
            
            # Chuẩn hóa dữ liệu bằng StandardScaler
            scaler = StandardScaler()
            temp_df[numeric_cols] = scaler.fit_transform(temp_df[numeric_cols])
            
            # Lưu DataFrame đã chuẩn hóa vào file tạm
            temp_df.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl'))
            
            data['selected_features'] = selected_features
            data['data_processed'] = True
            
            with open(selected_features_file, 'w') as f:
                f.write(','.join(selected_features))
            
            return render_template('process_data.html', data=data)
        
        if 'run_analysis' in request.form:
            # Xóa kết quả PCA cũ trước khi chạy PCA mới
            pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
            if os.path.exists(pca_results_file):
                os.remove(pca_results_file)
            
            selected_features = request.form.get('selected_features').split(',')
            process_method = request.form.get('process_method')
            
            data['selected_features'] = selected_features
            
            # Sử dụng DataFrame đã chuẩn hóa từ file tạm
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'temp_data.pkl')):
                flash("Dữ liệu đã chuẩn hóa không tồn tại. Vui lòng chọn lại feature.")
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
    
    return render_template('process_data.html', data=data)

def download_pca():
    pca_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_result.csv')
    if os.path.exists(pca_file):
        return send_file(pca_file, as_attachment=True)
    else:
        flash("Chưa có kết quả PCA.")
        return redirect(url_for('process_data'))