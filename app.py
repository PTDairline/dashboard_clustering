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
from routes.bcvi import bcvi

# Đăng ký các route với ứng dụng Flask
app.route('/', methods=['GET', 'POST'])(index)
app.route('/data_preview')(data_preview)
app.route('/process_data', methods=['GET', 'POST'])(process_data)
app.route('/download_pca')(download_pca)
app.route('/select_model', methods=['GET', 'POST'])(select_model)
app.route('/bcvi', methods=['GET', 'POST'])(bcvi)  

if __name__ == '__main__':
    app.run(debug=True)