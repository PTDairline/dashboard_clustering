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
        
        # Comment: Tính tỷ lệ phương sai giải thích của hai thành phần chính đầu tiên (PC1 và PC2).
        # - Nếu `n_components >= 2`, lấy tổng tỷ lệ phương sai của PC1 và PC2.
        # - Nếu `n_components = 1`, chỉ lấy tỷ lệ của PC1.
        # Đánh giá: Logic đúng, cung cấp thông tin về mức độ giải thích của hai thành phần chính đầu tiên.
        # Đề xuất:
        # - Có thể thêm logging để ghi lại giá trị:
        #   ```python
        #   logging.debug(f"Tỷ lệ phương sai giải thích bởi PC1+PC2: {variance_explained_pc1_pc2:.2f}%")
        #   ```
        # - Có thể tính thêm tỷ lệ phương sai tích lũy để hiển thị trên giao diện:
        #   ```python
        #   cumulative_explained = sum(pca.explained_variance_ratio_) * 100
        #   ```

        variance_details = [
            {'component': f'PC{i+1}', 'variance_ratio': ratio * 100, 'cumulative_variance': cumulative}
            for i, (ratio, cumulative) in enumerate(zip(explained_variance_ratios, cumulative_variance))
        ]
        
        # Comment: Tạo danh sách chi tiết phương sai cho từng thành phần chính.
        # - Mỗi phần tử là một dict chứa tên thành phần (`PC{i+1}`), tỷ lệ phương sai (`variance_ratio`), và tỷ lệ phương sai tích lũy (`cumulative_variance`).
        # Đánh giá: Logic đúng, cung cấp thông tin chi tiết để hiển thị trên giao diện (ví dụ: bảng phương sai).
        # Đề xuất:
        # - Có thể thêm logging để ghi lại thông tin:
        #   ```python
        #   logging.debug(f"Chi tiết phương sai: {variance_details}")
        #   ```
        # - Có thể kiểm tra độ dài của `variance_details` để đảm bảo không rỗng:
        #   ```python
        #   if not variance_details:
        #       logging.warning("Không có chi tiết phương sai, PCA có thể thất bại")
        #   ```

        pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        pca_file = os.path.join('uploads', 'pca_result.csv')
        pca_df.to_csv(pca_file, index=False)
        
        # Comment: Lưu kết quả PCA (`X_pca`) vào DataFrame và lưu thành file CSV (`pca_result.csv`).
        # - Tạo DataFrame `pca_df` với các cột dạng `PC1`, `PC2`, ...
        # - Lưu vào file `pca_result.csv` trong thư mục `uploads`.
        # Đánh giá: Logic đúng, cho phép người dùng tải kết quả PCA dưới dạng CSV.
        # Đề xuất:
        # - Thêm logging để ghi lại hành động:
        #   ```python
        #   logging.debug(f"Đã lưu kết quả PCA vào {pca_file}")
        #   ```
        # - Xử lý lỗi khi lưu file:
        #   ```python
        #   try:
        #       pca_df.to_csv(pca_file, index=False)
        #   except Exception as e:
        #       logging.error(f"Lỗi lưu pca_result.csv: {str(e)}")
        #       return None, None, f"Lỗi lưu kết quả PCA: {str(e)}", None, None
        #   ```
        # - Đường dẫn `'uploads'` có thể cần kiểm tra thư mục tồn tại:
        #   ```python
        #   os.makedirs('uploads', exist_ok=True)
        #   ```

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
        
        # Comment: Vẽ biểu đồ phương sai tích lũy của PCA.
        # - Tạo biểu đồ đường với trục x là số thành phần chính, trục y là tỷ lệ phương sai tích lũy.
        # - Vẽ đường ngang tại `explained_variance_ratio` để người dùng thấy ngưỡng.
        # - Lưu biểu đồ dưới dạng PNG, mã hóa base64 để hiển thị trên giao diện.
        # - Đóng biểu đồ để giải phóng bộ nhớ.
        # Đánh giá: Logic đúng, tạo biểu đồ trực quan hóa kết quả PCA hợp lý.
        # Đề xuất:
        # - Có thể thêm lưới (grid) để dễ nhìn hơn:
        #   ```python
        #   plt.grid(True)
        #   ```
        # - Có thể thêm logging thời gian tạo biểu đồ:
        #   ```python
        #   import time
        #   start_time = time.time()
        #   plt.savefig(...)
        #   logging.debug(f"Thời gian tạo biểu đồ PCA: {time.time() - start_time:.2f} giây")
        #   ```
        # - Xử lý lỗi khi tạo biểu đồ:
        #   ```python
        #   try:
        #       plt.savefig(buf, format='png', dpi=100)
        #   except Exception as e:
        #       logging.error(f"Lỗi tạo biểu đồ PCA: {str(e)}")
        #       return None, None, "Lỗi tạo biểu đồ PCA.", None, None
        #   ```

        return (X_pca, plot_data, 
                f"PCA giữ {n_components} thành phần, giải thích {cumulative_variance[n_components-1]*100:.2f}% phương sai.", 
                variance_explained_pc1_pc2, variance_details)
    
    # Comment: Trả về kết quả PCA.
    # - `X_pca`: Ma trận dữ liệu sau PCA.
    # - `plot_data`: Biểu đồ phương sai tích lũy (base64).
    # - Thông báo kết quả PCA (số thành phần, tỷ lệ phương sai).
    # - `variance_explained_pc1_pc2`: Tỷ lệ phương sai của PC1+PC2.
    # - `variance_details`: Chi tiết phương sai của từng thành phần.
    # Đánh giá: Logic đúng, trả về đầy đủ thông tin cần thiết để hiển thị trên giao diện và sử dụng ở các bước sau.
    # Đề xuất:
    # - Có thể thêm logging để ghi lại kích thước của `X_pca`:
    #   ```python
    #   logging.debug(f"Kích thước X_pca: {X_pca.shape}")
    #   ```

    except Exception as e:
        logging.error(f"PCA error: {str(e)}")
        return None, None, f"Lỗi PCA: {str(e)}", None, None

# Comment: Xử lý lỗi nếu PCA thất bại.
# - Ghi log lỗi và trả về kết quả lỗi với thông báo.
# Đánh giá: Logic đúng, xử lý lỗi tổng quát để tránh crash pipeline.
# Đề xuất:
# - Có thể cung cấp thông báo lỗi chi tiết hơn cho người dùng:
#   ```python
#   if "memory" in str(e).lower():
#       return None, None, "Lỗi PCA: Bộ nhớ không đủ để chạy PCA.", None, None
#   ```