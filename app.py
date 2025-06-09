import sys
import os
import logging

# Thêm thư mục gốc vào sys.path để Python nhận diện package
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask

app = Flask(__name__)
app.secret_key = "super_secret_key"
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Import routes (phiên bản gốc)
from routes.index import index
from routes.data_preview import data_preview
from routes.process_data import process_data, download_pca
from routes.select_model import select_model
from routes.bcvi import bcvi, download_bcvi

# Import Dashkit routes (phiên bản mới)
# Import Dashkit routes (phiên bản mới)
from routes.index_dashkit import index_dashkit
from routes.data_preview_dashkit import data_preview_dashkit
from routes.process_data_dashkit import process_data_dashkit
from routes.select_model_dashkit import select_model_dashkit
from routes.clustering_metrics_dashkit import clustering_metrics_dashkit
from routes.bcvi_dashkit_fixed import bcvi_dashkit, download_bcvi_dashkit
from routes.cluster_analysis_dashkit import cluster_analysis_dashkit  # Đảm bảo import này đúng
# Đăng ký các route gốc (để tương thích ngược)
app.add_url_rule('/old_index', 'old_index', index, methods=['GET', 'POST'])
app.add_url_rule('/old_data_preview', 'old_data_preview', data_preview)
app.add_url_rule('/old_process_data', 'old_process_data', process_data, methods=['GET', 'POST'])
app.add_url_rule('/old_select_model', 'old_select_model', select_model, methods=['GET', 'POST'])
app.add_url_rule('/old_bcvi', 'old_bcvi', bcvi, methods=['GET', 'POST'])
app.add_url_rule('/old_download_bcvi', 'old_download_bcvi', download_bcvi)
app.add_url_rule('/download_pca', 'download_pca', download_pca)

# Đăng ký các route Dashkit (phiên bản mới - làm mặc định)
app.add_url_rule('/', 'dashkit_index', index_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/data_preview', 'dashkit_data_preview', data_preview_dashkit)
app.add_url_rule('/process_data_dashkit', 'process_data_dashkit', process_data_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/select_model', 'select_model', select_model_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/clustering_metrics', 'clustering_metrics', clustering_metrics_dashkit)
app.add_url_rule('/bcvi', 'bcvi', bcvi_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/download_bcvi', 'download_bcvi_dashkit', download_bcvi_dashkit)

# THÊM ROUTE MỚI CHO PHÂN TÍCH ĐẶC TRƯNG CỤM
app.add_url_rule('/cluster_analysis', 'cluster_analysis_dashkit', cluster_analysis_dashkit, methods=['GET', 'POST'])
# Thêm vào cuối file app.py trước if __name__ == '__main__':

if __name__ == '__main__':
    app.run(debug=True)