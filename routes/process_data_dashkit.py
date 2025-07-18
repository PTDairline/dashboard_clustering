import os
import pandas as pd
import numpy as np
from flask import render_template, request, flash, redirect, url_for, current_app, send_file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import plotly.utils
import json
import re

# Thêm imports cho matplotlib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from utils.data_processing import convert_money_columns

# Danh sách feature mặc định
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

def convert_currency_to_float(value):
    """Chuyển đổi chuỗi tiền tệ (e.g., '€110.5M', '€1.5K', '€0') thành số float."""
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    try:
        # Xóa ký hiệu € và khoảng trắng
        value = str(value).replace('€', '').strip()
        # Xử lý hậu tố M (triệu), K (nghìn), hoặc không có hậu tố
        if value.endswith('M'):
            return float(value[:-1]) * 1_000_000
        elif value.endswith('K'):
            return float(value[:-1]) * 1_000
        else:
            # Thử chuyển đổi trực tiếp nếu không có hậu tố
            return float(value)
    except (ValueError, TypeError):
        return np.nan

def is_convertible_to_numeric(series):
    """Kiểm tra xem một cột có thể chuyển đổi hoàn toàn thành số hay không."""
    try:
        series.apply(convert_currency_to_float).astype(float)
        return True
    except (ValueError, TypeError):
        return False

def convert_height_to_cm(value):
    """Chuyển đổi chiều cao dạng feet-inches (ví dụ "5'7") sang cm."""
    if pd.isna(value):
        return np.nan
    
    s = str(value).strip()
    
    # Xử lý các định dạng feet-inches khác nhau
    # Pattern 1: 5'7, 5'7", 5ft7, 5ft 7in, 5' 7"
    patterns = [
        r"^(\d+)'(\d+)(?:\")?$",                   # 5'7 hoặc 5'7"
        r"^(\d+)\s*(?:ft|feet)\s*(\d+)(?:in|\")?$", # 5ft7 hoặc 5ft 7in
        r"^(\d+)'\s*(\d+)(?:\")?$",                # 5' 7 hoặc 5' 7"
        r"^(\d+)\s*-\s*(\d+)$"                     # 5-7
    ]
    
    for pattern in patterns:
        match = re.match(pattern, s, re.IGNORECASE)
        if match:
            try:
                feet = int(match.group(1))
                inches = int(match.group(2))
                return round((feet * 12 + inches) * 2.54, 1)
            except (ValueError, IndexError):
                continue
    
    # Nếu là số đơn thuần (có thể là cm)
    try:
        return float(s)
    except Exception:
        return np.nan

def convert_weight_to_kg(value):
    """Chuyển đổi cân nặng dạng số hoặc có đơn vị lbs sang kg."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    match = re.match(r"^(\d+)(?:\s?lbs)?$", s, re.IGNORECASE)
    if match:
        lbs = int(match.group(1))
        return round(lbs * 0.453592, 1)
    try:
        return float(s)
    except Exception:
        return np.nan

def convert_height_weight_columns(df):
    # Chuyển đổi cột Height và Weight nếu có
    if 'Height' in df.columns:
        df['Height'] = df['Height'].apply(convert_height_to_cm)
    if 'Weight' in df.columns:
        df['Weight'] = df['Weight'].apply(convert_weight_to_kg)
    return df

def process_data_dashkit():
    # Khởi tạo data mặc định để tránh lỗi undefined
    data = {
        'num_rows': 0,
        'num_features': 0,
        'numerical_features': [],
        'numeric_columns': [],
        'non_numeric_columns': [],
        'selected_features': [],
        'pca_result': None,
        'use_pca': True,
        'proceed_to_model': False,
        'default_features': DEFAULT_FEATURES
    }
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'select_features':
            # Kiểm tra dữ liệu
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
                flash('Không có dữ liệu để xử lý. Vui lòng tải lên dữ liệu trước.')
                return redirect(url_for('process_data_dashkit'))
            
            try:
                df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
                df = convert_money_columns(df)
                df = convert_height_weight_columns(df)
                
                feature_option = request.form.get('feature_option')
                use_pca = request.form.get('use_pca') == 'yes'
                
                if feature_option == 'custom':
                    selected_features = request.form.getlist('features')
                    if not selected_features:
                        flash('Vui lòng chọn ít nhất một feature.')
                        return redirect(url_for('process_data_dashkit'))
                else:
                    # Sử dụng feature mặc định có trong dữ liệu
                    selected_features = [f for f in DEFAULT_FEATURES if f in df.columns]
                
                # Cập nhật danh sách cột số và không phải số
                numeric_columns = [col for col in df.columns if is_convertible_to_numeric(df[col])]
                non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
                
                # Kiểm tra PCA với cột số
                if use_pca:
                    invalid_features = [f for f in selected_features if f not in numeric_columns]
                    if invalid_features:
                        flash(f'PCA chỉ hỗ trợ các cột số. Các cột không hợp lệ: {", ".join(invalid_features)}')
                        return redirect(url_for('process_data_dashkit'))
                
                # Lưu selected features
                with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'), 'w') as f:
                    f.write(','.join(selected_features))
                
                # Lưu lựa chọn PCA
                with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt'), 'w') as f:
                    f.write('yes' if use_pca else 'no')
                
                flash(f'Đã chọn {len(selected_features)} features.')
                
                # Nếu không dùng PCA, chuẩn bị dữ liệu và chuyển sang bước chọn mô hình
                if not use_pca:
                    try:
                        selected_data = df[selected_features].copy()
                        for col in selected_features:
                            if col in numeric_columns:
                                selected_data[col] = selected_data[col].fillna(selected_data[col].mean())
                            else:
                                selected_data[col] = selected_data[col].fillna(selected_data[col].mode()[0])
                        
                        selected_data.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_data.pkl'))
                        
                        pca_results = {
                            'n_components': len(selected_features),
                            'explained_variance_ratio': 1.0,
                            'original_features': selected_features,
                            'no_pca': True
                        }
                        with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json'), 'w') as f:
                            json.dump(pca_results, f)
                        
                        # Xóa cache BCVI
                        bcvi_cache_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_cache.pkl')
                        bcvi_flag_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'bcvi_calculated.flag')
                        if os.path.exists(bcvi_cache_file):
                            os.remove(bcvi_cache_file)
                        if os.path.exists(bcvi_flag_file):
                            os.remove(bcvi_flag_file)
                            
                        flash('Đã bỏ qua PCA và dùng trực tiếp các features đã chọn.')
                        return redirect(url_for('select_model'))
                    except Exception as e:
                        flash(f'Lỗi khi chuẩn bị dữ liệu không dùng PCA: {str(e)}')
                        return redirect(url_for('process_data_dashkit'))
                
                return redirect(url_for('process_data_dashkit'))
                
            except Exception as e:
                flash(f'Lỗi khi xử lý dữ liệu: {str(e)}')
                return redirect(url_for('process_data_dashkit'))
        
        elif action == 'perform_pca':
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
                flash('Vui lòng tải dữ liệu trước khi thực hiện PCA.')
                return redirect(url_for('process_data_dashkit'))
            
            if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')):
                flash('Vui lòng chọn features trước khi thực hiện PCA.')
                return redirect(url_for('process_data_dashkit'))
            
            try:
                df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
                df = convert_money_columns(df)
                df = convert_height_weight_columns(df)
                
                with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt'), 'r') as f:
                    content = f.read().strip()
                    selected_features = content.split(',') if content else []
                
                if not selected_features:
                    flash('Không có features nào được chọn.')
                    return redirect(url_for('process_data_dashkit'))
                
                # Kiểm tra lại các cột có thể chuyển đổi thành số
                numeric_columns = [col for col in selected_features if is_convertible_to_numeric(df[col])]
                invalid_features = [f for f in selected_features if f not in numeric_columns]
                if invalid_features:
                    flash(f'PCA chỉ hỗ trợ các cột số. Các cột không hợp lệ: {", ".join(invalid_features)}')
                    return redirect(url_for('process_data_dashkit'))
                
                explained_variance_ratio = float(request.form.get('explained_variance_ratio', 90)) / 100
                
                X = df[numeric_columns]
                X = X.fillna(X.mean(numeric_only=True))
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                pca = PCA()
                X_pca = pca.fit_transform(X_scaled)
                
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= explained_variance_ratio) + 1
                
                pca_final = PCA(n_components=n_components)
                X_pca_final = pca_final.fit_transform(X_scaled)
                
                scree_plot_json = create_scree_plot(pca.explained_variance_ratio_)
                pca_2d_plot_json = create_pca_2d_plot(X_pca_final)
                pca_variance_plot = create_pca_variance_plot_classic(
                    pca.explained_variance_ratio_[:min(20, len(pca.explained_variance_ratio_))], 
                    cumsum[:min(20, len(cumsum))], 
                    n_components
                )
                variance_details = create_variance_details_table(
                    pca_final.explained_variance_ratio_, 
                    cumsum[:n_components], 
                    n_components
                )
                
                loadings = pca_final.components_
                components_matrix = loadings.copy()
                
                feature_importance = np.mean(np.abs(loadings), axis=0)
                
                full_pca_loadings = {}
                for i in range(n_components):
                    pc_loadings = []
                    for j, feature in enumerate(numeric_columns):
                        loading_value = float(loadings[i, j])
                        abs_loading = float(abs(loading_value))
                        pc_loadings.append({
                            'feature': feature,
                            'loading': loading_value,
                            'abs_loading': abs_loading,
                            'direction': "+" if loading_value > 0 else "-",
                            'contribution': f"{'+' if loading_value > 0 else '-'}{feature} ({abs_loading:.3f})"
                        })
                    pc_loadings.sort(key=lambda x: x['abs_loading'], reverse=True)
                    full_pca_loadings[f'PC{i+1}'] = pc_loadings
                
                num_top_features = min(5, len(numeric_columns))
                top_features_by_component = {}
                for i in range(n_components):
                    top_features_info = full_pca_loadings[f'PC{i+1}'][:num_top_features]
                    top_features_by_component[f'PC{i+1}'] = top_features_info
                    
                feature_importance_dict = {}
                for i, feature in enumerate(numeric_columns):
                    feature_importance_dict[feature] = float(feature_importance[i])
                
                pca_results = {
                    'n_components': int(n_components),
                    'explained_variance_ratio': float(cumsum[n_components-1]),
                    'target_variance': float(explained_variance_ratio),
                    'original_features': numeric_columns,
                    'individual_explained_variance': [float(x) for x in pca_final.explained_variance_ratio_],
                    'cumulative_explained_variance': [float(x) for x in cumsum[:n_components]],
                    'pca_message': f'PCA đã hoàn thành! Đã giảm từ {len(numeric_columns)} features xuống {n_components} thành phần chính, giữ lại {float(cumsum[n_components-1]*100):.1f}% phương sai.',
                    'variance_details': variance_details,
                    'top_features_by_component': top_features_by_component,
                    'full_pca_loadings': full_pca_loadings,
                    'feature_importance': feature_importance_dict,
                    'components_matrix': components_matrix.tolist(),
                    'scree_plot': scree_plot_json,
                    'pca_2d_plot': pca_2d_plot_json,
                    'pca_plot': pca_variance_plot
                }
                
                components_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_components.npy')
                with open(components_file, 'wb') as f:
                    np.save(f, components_matrix)
                
                pca_df = pd.DataFrame(X_pca_final, columns=[f'PC{i+1}' for i in range(n_components)])
                pca_df.to_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_data.pkl'))
                
                pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
                with open(pca_results_file, 'w', encoding='utf-8') as f:
                    json.dump(pca_results, f, ensure_ascii=False, indent=2)
                
                flash(f'PCA hoàn thành! Đã giảm từ {len(numeric_columns)} features xuống {n_components} thành phần chính.')
                return redirect(url_for('process_data_dashkit'))
                
            except Exception as e:
                flash(f'Lỗi khi thực hiện PCA: {str(e)}')
                return redirect(url_for('process_data_dashkit'))
    
    # GET request - hiển thị trang
    try:
        if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
            flash('Không có dữ liệu để xử lý. Vui lòng tải lên dữ liệu trước.')
            return render_template('process_data_dashkit.html', data=data)
        
        df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
        df = convert_money_columns(df)
        df = convert_height_weight_columns(df)
        
        all_features = df.columns.tolist()
        exclude_columns = ['ID', 'Jersey Number']
        numerical_features = [col for col in all_features if col not in exclude_columns]
        
        numeric_columns = [col for col in df.columns if is_convertible_to_numeric(df[col])]
        non_numeric_columns = [col for col in df.columns if col not in numeric_columns]
        
        use_pca = True
        use_pca_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'use_pca.txt')
        if os.path.exists(use_pca_file):
            with open(use_pca_file, 'r') as f:
                use_pca = f.read().strip() == 'yes'
        
        selected_features = []
        features_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'selected_features.txt')
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                content = f.read().strip()
                selected_features = content.split(',') if content else []
        
        pca_result = None
        pca_results_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_results.json')
        if os.path.exists(pca_results_file):
            try:
                with open(pca_results_file, 'r', encoding='utf-8') as f:
                    pca_result = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading PCA results: {e}")
                os.remove(pca_results_file)
        
        data = {
            'num_rows': len(df),
            'num_features': len(df.columns),
            'numerical_features': numerical_features,
            'numeric_columns': numeric_columns,
            'non_numeric_columns': non_numeric_columns,
            'selected_features': selected_features,
            'pca_result': pca_result,
            'use_pca': use_pca,
            'proceed_to_model': bool(pca_result) or (selected_features and not use_pca),
            'default_features': DEFAULT_FEATURES
        }
        
        return render_template('process_data_dashkit.html', data=data)
        
    except Exception as e:
        flash(f'Lỗi khi tải dữ liệu: {str(e)}')
        return render_template('process_data_dashkit.html', data=data)

def create_scree_plot(explained_variance_ratio):
    try:
        components = list(range(1, min(len(explained_variance_ratio) + 1, 21)))
        variance_values = [float(x) for x in explained_variance_ratio[:20]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=components,
            y=variance_values,
            mode='lines+markers',
            name='Explained Variance Ratio',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4')
        ))
        fig.update_layout(
            title={'text': 'Scree Plot - Explained Variance by Component', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        return fig.to_json()
    except Exception as e:
        print(f"Error creating scree plot: {e}")
        return None

def create_pca_2d_plot(X_pca):
    try:
        if X_pca.shape[1] < 2:
            return None
        max_points = 1000
        if X_pca.shape[0] > max_points:
            indices = np.random.choice(X_pca.shape[0], max_points, replace=False)
            X_pca_sample = X_pca[indices]
        else:
            X_pca_sample = X_pca
        pc1 = [float(x) for x in X_pca_sample[:, 0]]
        pc2 = [float(x) for x in X_pca_sample[:, 1]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pc1,
            y=pc2,
            mode='markers',
            name='Data Points',
            marker=dict(
                size=8,
                color=pc1,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="PC1 Value"),
                line=dict(width=0.5, color='white')
            )
        ))
        fig.update_layout(
            title={'text': 'PCA 2D Visualization', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title='First Principal Component (PC1)',
            yaxis_title='Second Principal Component (PC2)',
            height=500,
            showlegend=False,
            template='plotly_white'
        )
        return fig.to_json()
    except Exception as e:
        print(f"Error creating PCA 2D plot: {e}")
        return None

def create_pca_variance_plot_classic(explained_variance_ratio, cumulative_variance, n_components):
    try:
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        components = range(1, len(explained_variance_ratio) + 1)
        ax1.bar(components, explained_variance_ratio, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Thành phần chính')
        ax1.set_ylabel('Tỷ lệ phương sai giải thích')
        ax1.set_title('Tỷ lệ phương sai của từng thành phần')
        ax1.grid(True, alpha=0.3)
        if n_components > 0:
            ax1.bar(components[:n_components], explained_variance_ratio[:n_components], 
                   alpha=0.9, color='orange', label=f'{n_components} thành phần được chọn')
            ax1.legend()
        ax2.plot(components, cumulative_variance, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% phương sai')
        ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% phương sai')
        if n_components > 0:
            ax2.axvline(x=n_components, color='green', linestyle='--', alpha=0.7, 
                       label=f'Số thành phần chọn: {n_components}')
            ax2.plot(n_components, cumulative_variance[n_components-1], 'ro', markersize=10)
        ax2.set_xlabel('Số thành phần chính')
        ax2.set_ylabel('Phương sai tích lũy')
        ax2.set_title('Phương sai tích lũy theo số thành phần')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.05)
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error creating variance plot: {e}")
        return None

def create_variance_details_table(explained_variance_ratio, cumulative_variance, n_components):
    try:
        variance_details = []
        for i in range(n_components):
            variance_details.append({
                'component': f'PC{i+1}',
                'variance_ratio': float(explained_variance_ratio[i] * 100),
                'cumulative_variance': float(cumulative_variance[i])
            })
        return variance_details
    except Exception as e:
        print(f"Error creating variance details: {e}")        
        return []

def download_pca_dashkit():
    pca_file = os.path.join(current_app.config['UPLOAD_FOLDER'], 'pca_result.csv')
    if os.path.exists(pca_file):
        return send_file(pca_file, as_attachment=True)
    else:
        flash("Chưa có kết quả PCA.")
        return redirect(url_for('process_data_dashkit'))