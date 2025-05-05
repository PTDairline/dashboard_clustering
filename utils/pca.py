import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import base64
import logging

def perform_pca(df, selected_features, explained_variance_ratio):
    try:
        logging.debug(f"Selected features for PCA: {selected_features}")
        numeric_cols = [col for col in selected_features if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
        logging.debug(f"Numeric columns: {numeric_cols}")
        
        if len(numeric_cols) < 2:
            return None, None, "Cần ít nhất 2 cột số để chạy PCA.", None, None
        
        X = df[numeric_cols].copy()
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].mean(), inplace=True)
        
        if X.empty or X.shape[0] < 2:
            return None, None, "Dữ liệu không đủ để chạy PCA.", None, None
        
        X = (X - X.mean()) / X.std()
        pca = PCA()
        pca.fit(X)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        explained_variance_ratios = pca.explained_variance_ratio_
        n_components = np.argmax(cumulative_variance >= explained_variance_ratio) + 1
        if n_components < 1:
            n_components = 1
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        variance_explained_pc1_pc2 = sum(pca.explained_variance_ratio_[:2]) * 100 if n_components >= 2 else pca.explained_variance_ratio_[0] * 100
        
        variance_details = [
            {'component': f'PC{i+1}', 'variance_ratio': ratio * 100, 'cumulative_variance': cumulative}
            for i, (ratio, cumulative) in enumerate(zip(explained_variance_ratios, cumulative_variance))
        ]
        
        pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        pca_file = os.path.join('uploads', 'pca_result.csv')
        pca_df.to_csv(pca_file, index=False)
        
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        plt.axhline(y=explained_variance_ratio, color='r', linestyle='--')
        plt.xlabel('Số thành phần chính')
        plt.ylabel('Tỷ lệ phương sai tích lũy')
        plt.title('Phương sai tích lũy PCA')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close('all')
        logging.debug(f"PCA completed: {n_components} components")
        
        return (X_pca, plot_data, 
                f"PCA giữ {n_components} thành phần, giải thích {cumulative_variance[n_components-1]*100:.2f}% phương sai.", 
                variance_explained_pc1_pc2, variance_details)
    except Exception as e:
        logging.error(f"PCA error: {str(e)}")
        return None, None, f"Lỗi PCA: {str(e)}", None, None