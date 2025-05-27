# Dashboard với Dashkit

Đây là hướng dẫn tích hợp Dashkit vào dashboard của bạn.

## Giới thiệu

Dashkit là một template/framework Bootstrap hiện đại cho trang quản trị, giúp tạo giao diện đẹp và chuyên nghiệp. Dự án này đã được tích hợp Dashkit vào dashboard phân cụm dữ liệu hiện có.

## Cấu trúc thư mục

```
static/
  dashkit/
    css/
      dashkit.css      # CSS cơ bản của Dashkit
      custom.css       # CSS tùy chỉnh
      icons.css        # Các icon SVG
    js/
      dashkit.js       # JavaScript của Dashkit
    fonts/             # Fonts cho Dashkit
    img/               # Hình ảnh cho Dashkit
templates/
  base_dashkit.html    # Template cơ bản
  index_dashkit.html   # Trang chủ với Dashkit
  data_preview_dashkit.html  # Trang xem dữ liệu với Dashkit
```

## Cách sử dụng

1. **Kế thừa từ template cơ bản**: Khi muốn tạo một trang mới, hãy kế thừa từ `base_dashkit.html`:

```html
{% extends "base_dashkit.html" %}

{% block title %}Tên trang{% endblock %}

{% block header %}Tiêu đề trang{% endblock %}

{% block content %}
  <!-- Nội dung trang -->
{% endblock %}
```

2. **Thêm CSS tùy chỉnh**: Sử dụng block `extra_css` để thêm CSS riêng cho trang:

```html
{% block extra_css %}
<style>
  /* CSS cho trang này */
</style>
{% endblock %}
```

3. **Thêm JavaScript tùy chỉnh**: Sử dụng block `extra_js` để thêm JavaScript riêng cho trang:

```html
{% block extra_js %}
<script>
  // JavaScript cho trang này
</script>
{% endblock %}
```

## Components

### Cards

```html
<div class="card">
  <div class="card-header">
    <h5 class="card-title">
      <i class="fas fa-chart-line me-2 text-primary"></i>Tiêu đề
    </h5>
  </div>
  <div class="card-body">
    <!-- Nội dung card -->
  </div>
</div>
```

### Buttons

```html
<button class="btn btn-primary">Button chính</button>
<button class="btn btn-secondary">Button phụ</button>
<button class="btn btn-success">Button thành công</button>
<button class="btn btn-danger">Button nguy hiểm</button>
<button class="btn btn-warning">Button cảnh báo</button>
<button class="btn btn-info">Button thông tin</button>
```

### Tables

```html
<table class="table table-striped table-hover">
  <thead class="table-light">
    <tr>
      <th>Cột 1</th>
      <th>Cột 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dữ liệu 1</td>
      <td>Dữ liệu 2</td>
    </tr>
  </tbody>
</table>
```

### Alerts

```html
<div class="alert alert-success">Thông báo thành công</div>
<div class="alert alert-danger">Thông báo lỗi</div>
<div class="alert alert-warning">Thông báo cảnh báo</div>
<div class="alert alert-info">Thông báo thông tin</div>
```

## Các trang đã tích hợp

1. **Trang chủ**: `/` - Sử dụng template `index_dashkit.html`
2. **Xem dữ liệu**: `/data_preview` - Sử dụng template `data_preview_dashkit.html`

## Chuyển đổi từ version cũ

Các trang cũ vẫn được giữ lại và có thể truy cập từ:

1. **Trang chủ (cũ)**: `/classic`
2. **Xem dữ liệu (cũ)**: `/classic/data_preview`

## Icons

Dashkit sử dụng Font Awesome cho các icon. Một số icon thông dụng:

- `<i class="fas fa-home"></i>` - Icon trang chủ
- `<i class="fas fa-chart-bar"></i>` - Icon biểu đồ
- `<i class="fas fa-table"></i>` - Icon bảng
- `<i class="fas fa-cogs"></i>` - Icon cài đặt
- `<i class="fas fa-user"></i>` - Icon người dùng
- `<i class="fas fa-file"></i>` - Icon file
