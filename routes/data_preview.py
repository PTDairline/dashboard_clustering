from flask import render_template, flash, redirect, url_for, request, current_app
import os
import pandas as pd

def data_preview():
    data = {
        'features': [],
        'num_features': 0,
        'feature_types': {},
        'preview_data': None,
        'data_stats': None,
        'file_uploaded': False,
        'proceed_to_process': False
    }
    
    if os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl')):
        df = pd.read_pickle(os.path.join(current_app.config['UPLOAD_FOLDER'], 'data.pkl'))
        data['file_uploaded'] = True
        data['features'] = df.columns.tolist()
        data['num_features'] = len(df.columns)
        data['feature_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        data['preview_data'] = df.head(5).to_dict(orient='records')
        data['data_stats'] = df.describe().to_dict()
        data['proceed_to_process'] = True  # Sửa cú pháp từ ['proceed_to_process': True] thành ['proceed_to_process'] = True
    
    return render_template('data_preview.html', data=data)