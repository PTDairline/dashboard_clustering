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

# Import routes
from routes.index import index
from routes.data_preview import data_preview
from routes.process_data import process_data, download_pca
from routes.select_model import select_model
from routes.bcvi import bcvi, download_bcvi

# Import Dashkit routes
from routes.index_dashkit import index_dashkit
from routes.data_preview_dashkit import data_preview_dashkit
from routes.process_data_dashkit import process_data_dashkit
from routes.select_model_dashkit import select_model_dashkit
from routes.clustering_metrics_dashkit import clustering_metrics_dashkit  # THÊM DÒNG NÀY
from routes.bcvi_dashkit_fixed import bcvi_dashkit, download_bcvi_dashkit

# Đăng ký các route Dashkit (phiên bản mới)
app.add_url_rule('/', 'dashkit_index', index_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/data_preview', 'dashkit_data_preview', data_preview_dashkit)
app.add_url_rule('/process_data_dashkit', 'process_data_dashkit', process_data_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/select_model', 'select_model', select_model_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/clustering_metrics', 'clustering_metrics', clustering_metrics_dashkit)
app.add_url_rule('/bcvi', 'bcvi', bcvi_dashkit, methods=['GET', 'POST'])
app.add_url_rule('/download_bcvi', 'download_bcvi_dashkit', download_bcvi_dashkit)
if __name__ == '__main__':
    app.run(debug=True)