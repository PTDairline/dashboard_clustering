/**
 * bcvi_dashboard.js - Xử lý biểu đồ và giao diện cho trang BCVI Dashboard
 */

// Khởi tạo theo dõi chart instances
if (!window.chartInstances) {
    window.chartInstances = {};
}

// Hàm setup lazy loading
function setupLazyLoading() {
    console.log("[BCVI Dashboard] Setting up lazy loading");
    
    // Lazy loading cho tables
    const lazyTables = document.querySelectorAll('.lazy-table[data-lazy="true"]');
    const lazyCharts = document.querySelectorAll('.lazy-chart[data-lazy="true"]');
    
    console.log(`[BCVI Dashboard] Found ${lazyTables.length} lazy tables and ${lazyCharts.length} lazy charts`);
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                
                if (element.classList.contains('lazy-table')) {
                    // Load table
                    const placeholder = element.querySelector('.loading-placeholder');
                    const table = element.querySelector('[data-content="cluster-size-table"]');
                    
                    setTimeout(() => {
                        placeholder.style.display = 'none';
                        table.classList.remove('d-none');
                        element.removeAttribute('data-lazy');
                    }, 100);
                    
                } else if (element.classList.contains('lazy-chart')) {
                    // Load chart
                    const placeholder = element.querySelector('.loading-placeholder');
                    const canvas = element.querySelector('canvas');
                    
                    setTimeout(() => {
                        placeholder.style.display = 'none';
                        canvas.classList.remove('d-none');
                        element.removeAttribute('data-lazy');
                    }, 200);
                }
                
                observer.unobserve(element);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '50px'
    });
    
    lazyTables.forEach(table => observer.observe(table));
    lazyCharts.forEach(chart => observer.observe(chart));
}

// Cập nhật các chỉ báo độ ưu tiên cho các trường hợp alpha
function updateAlphaIndicators() {
    document.querySelectorAll('input[name^="alpha_"]').forEach(input => {
        const value = parseFloat(input.value) || 0;
        const k = input.getAttribute('data-k');
        const indicator = document.querySelector(`.priority-indicator[data-k="${k}"]`);
        
        if (indicator) {
            let text = '';
            let className = 'badge ';
            
            if (value < 1) {
                text = 'Thấp';
                className += 'bg-info';
            } else if (value === 1) {
                text = 'Chuẩn';
                className += 'bg-secondary';
            } else if (value <= 10) {
                text = 'Cao';
                className += 'bg-warning';
            } else if (value <= 100) {
                text = 'Rất cao';
                className += 'bg-danger';
            } else {
                text = 'Cực cao';
                className += 'bg-dark';
            }
            
            indicator.textContent = text;
            indicator.className = className;
        }
    });
}

// Cập nhật tổng Alpha
function updateAlphaSum() {
    let sum = 0;
    document.querySelectorAll('input[name^="alpha_"]').forEach(input => {
        sum += parseFloat(input.value) || 0;
    });
    
    const sumElement = document.getElementById('alphaSum');
    if (sumElement) {
        sumElement.textContent = sum.toFixed(2);
    }
}

// Thiết lập preset alpha
function setAlphaPreset(preset) {
    const inputs = document.querySelectorAll('input[name^="alpha_"]');
    const count = inputs.length;
    
    inputs.forEach((input, index) => {
        let value = 1.0;

        switch(preset) {
            case 'uniform':
                value = 1.00;
                break;
            case 'ascending':
                value = 0.50 + (index * 9.50 / (count - 1)); // 0.5 → 10
                break;
            case 'descending':
                value = 10.00 - (index * 9.50 / (count - 1)); // 10 → 0.5
                break;
            case 'peak':
                const mid = Math.floor(count / 2);
                const distance = Math.abs(index - mid);
                value = 25.00 - (distance * 5); // Max 25 at center
                break;
            case 'extreme':
                value = 100.00; // Tất cả đều 100
                break;
        }
        
        input.value = value.toFixed(2);
    });
    
    updateAlphaIndicators();
    updateAlphaSum();
}

// Thiết lập alpha tùy chỉnh cho tất cả K
function setCustomAlpha() {
    const customValue = document.getElementById('customAlpha').value;
    if (customValue && !isNaN(customValue) && parseFloat(customValue) > 0) {
        document.querySelectorAll('input[name^="alpha_"]').forEach(input => {
            input.value = parseFloat(customValue).toFixed(2);
        });
        
        updateAlphaIndicators();
        updateAlphaSum();
        
        // Clear custom input
        document.getElementById('customAlpha').value = '';
    } else {
        alert('Vui lòng nhập giá trị alpha hợp lệ (> 0)');
    }
}

// Thiết lập alpha tối ưu cho k cụ thể
function setAlphaForOptimalK(optimalK) {
    document.querySelectorAll('input[name^="alpha_"]').forEach(input => {
        const k = parseInt(input.getAttribute('data-k'));
        
        if (k === optimalK) {
            input.value = '25.00';  // Ưu tiên rất cao cho k tối ưu
        } else {
            input.value = '0.50';  // Ưu tiên thấp cho các k khác
        }
    });
    
    updateAlphaIndicators();
    updateAlphaSum();
}

// Hiển thị loading indicator trên form submit
function setupFormSubmit() {
    const bcviForm = document.getElementById('bcviForm');
    if (bcviForm) {
        bcviForm.addEventListener('submit', function() {
            console.log('Form submitted!');
            
            // Show loading overlay
            const loadingOverlay = document.createElement('div');
            loadingOverlay.id = 'bcvi-loading-overlay'; // Add ID for easy removal
            loadingOverlay.style.position = 'fixed';
            loadingOverlay.style.top = '0';
            loadingOverlay.style.left = '0';
            loadingOverlay.style.width = '100%';
            loadingOverlay.style.height = '100%';
            loadingOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
            loadingOverlay.style.zIndex = '9999';
            loadingOverlay.style.display = 'flex';
            loadingOverlay.style.justifyContent = 'center';
            loadingOverlay.style.alignItems = 'center';
            
            const spinner = document.createElement('div');
            spinner.innerHTML = `
                <div class="spinner-border text-light" role="status">
                    <span class="visually-hidden">Đang tính toán...</span>
                </div>
                <div class="text-light mt-3">Đang tính BCVI...</div>
            `;
            loadingOverlay.appendChild(spinner);
            
            document.body.appendChild(loadingOverlay);
        });
    }
}

// Khởi tạo các biểu đồ so sánh CVI và BCVI
function initializeCharts() {
    console.log('[BCVI Dashboard] Initializing charts');
    
    // Kiểm tra xem có hàm vẽ biểu đồ không
    if (typeof createCVIComparisonChart !== 'function') {
        console.error('[BCVI Dashboard] Chart drawing functions not available!');
        document.querySelectorAll('.chart-container').forEach(container => {
            container.innerHTML = '<div class="alert alert-danger">Error: Chart drawing functions not available</div>';
        });
        return false;
    }
    
    // Tìm tất cả canvas để debug
    const canvasIds = Array.from(document.querySelectorAll('canvas[id]')).map(c => c.id);
    console.log('[BCVI Dashboard] Found canvas elements:', canvasIds);
    
    // Kiểm tra canvas biểu đồ BCVI
    const bcviCanvases = Array.from(document.querySelectorAll('canvas[id$="-bcvi-comparison-chart"]'));
    console.log('[BCVI Dashboard] Found BCVI charts:', bcviCanvases.map(c => c.id));
    
    // Kiểm tra canvas biểu đồ CVI
    const cviCanvases = Array.from(document.querySelectorAll('canvas[id$="-cvi-comparison-chart"]'));
    console.log('[BCVI Dashboard] Found CVI charts:', cviCanvases.map(c => c.id));
    
    // Nếu không có canvas nào, báo lỗi
    if (canvasIds.length === 0) {
        console.error('[BCVI Dashboard] No chart canvases found!');
        return false;
    }
    
    return true;
}

// Xử lý khi trang đã load
document.addEventListener('DOMContentLoaded', function() {
    console.log('[BCVI Dashboard] Page loaded');
    
    // Remove any loading overlay that might still be present from previous submission
    const existingOverlay = document.getElementById('bcvi-loading-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    // Xử lý form submit
    setupFormSubmit();
    
    // Hiệu ứng hover cho BCVI cards
    document.querySelectorAll('.bcvi-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 8px 20px rgba(0,0,0,0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
        });
    });

    // Cuộn mượt cho tables dài
    document.querySelectorAll('.bcvi-table').forEach(table => {
        table.style.scrollBehavior = 'smooth';
    });
    
    // Xử lý thay đổi Alpha input
    document.querySelectorAll('input[name^="alpha_"]').forEach(input => {
        input.addEventListener('input', function() {
            updateAlphaIndicators();
            updateAlphaSum();
        });
        
        input.addEventListener('focus', function() {
            this.closest('tr').style.backgroundColor = '#f8f9fa';
        });
        
        input.addEventListener('blur', function() {
            this.closest('tr').style.backgroundColor = '';
        });
    });
    
    // Hiệu ứng hover cho Next Steps card
    const nextStepsCard = document.querySelector('.next-steps-card');
    if (nextStepsCard) {
        nextStepsCard.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px)';
            this.style.boxShadow = '0 15px 35px rgba(0,0,0,0.2)';
        });
        
        nextStepsCard.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
        });
    }
    
    // Khởi tạo indicators
    updateAlphaIndicators();
    updateAlphaSum();
    
    // Setup lazy loading
    setupLazyLoading();
    
    // Khởi tạo biểu đồ sau khi trang đã load hoàn toàn
    setTimeout(function() {
        // Kiểm tra xem có các hàm vẽ biểu đồ không và các canvas đã sẵn sàng chưa
        initializeCharts();
    }, 500);
});
