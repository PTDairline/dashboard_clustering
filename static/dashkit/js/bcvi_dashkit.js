/**
 * BCVI Dashboard JavaScript
 * Handles interactive functionality for the BCVI Dashboard
 */

document.addEventListener('DOMContentLoaded', function() {
    // Remove any loading overlay that might still be present from previous submission
    const existingOverlay = document.getElementById('bcvi-loading-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    // Hiệu ứng hover cho BCVI cards
    const bcviCards = document.querySelectorAll('.bcvi-card');
    bcviCards.forEach(card => {
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
    const bcviTables = document.querySelectorAll('.bcvi-table');
    bcviTables.forEach(table => {
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
      
    // Form submission handler with debugging
    const bcviForm = document.getElementById('bcviForm');
    if (bcviForm) {
        bcviForm.addEventListener('submit', function(e) {
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
            spinner.innerHTML = '<div class="spinner-border text-light" role="status"><span class="visually-hidden">Đang tính toán...</span></div><div class="text-light mt-3">Đang tính BCVI...</div>';
            loadingOverlay.appendChild(spinner);
            
            document.body.appendChild(loadingOverlay);
        });
    }
    
    // Khởi tạo indicators
    updateAlphaIndicators();
    updateAlphaSum();
});

// Cập nhật các chỉ báo độ ưu tiên
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
