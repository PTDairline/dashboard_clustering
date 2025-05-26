// Tối ưu hóa: Thêm Intersection Observer cho lazy loading
function setupLazyLoading() {
    // Lazy loading cho tables
    const lazyTables = document.querySelectorAll('.lazy-table[data-lazy="true"]');
    const lazyCharts = document.querySelectorAll('.lazy-chart[data-lazy="true"]');
    
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
                        
                        // Trigger chart creation
                        const model = canvas.getAttribute('data-model');
                        const sizesJson = canvas.getAttribute('data-sizes');
                        
                        if (model && sizesJson) {
                            try {
                                const sizes = JSON.parse(sizesJson);
                                createClusterSizeChart(model, sizes);
                            } catch (e) {
                                console.error('Lỗi khi parse dữ liệu biểu đồ:', e);
                            }
                        }
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

// Khi trang đã load xong - với lazy loading và tối ưu hóa hiệu suất
document.addEventListener('DOMContentLoaded', function() {
    // Setup lazy loading
    setupLazyLoading();
    
    // Thêm loading indicator
    showLoadingIndicator();
    
    // Tối ưu hóa: Sử dụng requestAnimationFrame thay vì setTimeout
    requestAnimationFrame(() => {
        // Tìm các canvas không có lazy loading (visible ngay)
        const immediateCanvases = document.querySelectorAll('canvas[id^="clusterSizeChart"]:not(.d-none)');
        
        console.log('Found', immediateCanvases.length, 'immediate cluster size charts');
        
        if (immediateCanvases.length === 0) {
            hideLoadingIndicator();
            return;
        }
        
        // Tối ưu hóa: Batch processing để tránh lag
        const batchSize = 2; // Xử lý 2 chart cùng lúc
        let currentIndex = 0;
        
        function processBatch() {
            const endIndex = Math.min(currentIndex + batchSize, immediateCanvases.length);
            
            for (let i = currentIndex; i < endIndex; i++) {
                const canvas = immediateCanvases[i];
                const model = canvas.getAttribute('data-model');
                const sizesJson = canvas.getAttribute('data-sizes');
                
                console.log('Processing immediate canvas for model', model);
                
                if (model && sizesJson) {
                    try {
                        const sizes = JSON.parse(sizesJson);
                        createClusterSizeChart(model, sizes);
                    } catch (e) {
                        console.error('Lỗi khi parse dữ liệu biểu đồ:', e);
                    }
                }
            }
            
            currentIndex = endIndex;
            
            if (currentIndex < immediateCanvases.length) {
                requestAnimationFrame(processBatch);
            } else {
                hideLoadingIndicator();
            }
        }
        
        // Bắt đầu batch processing
        processBatch();
        
        // Fallback: ẩn loading sau 3 giây
        setTimeout(hideLoadingIndicator, 3000);
    });
});

// Hiển thị loading indicator
function showLoadingIndicator() {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'bcvi-loading';
    loadingDiv.innerHTML = `
        <div style="text-align: center; padding: 20px;">
            <div class="spinner-border text-primary" role="status">
                <span class="sr-only">Đang tải biểu đồ...</span>
            </div>
            <p class="mt-2">Đang tải biểu đồ BCVI...</p>
        </div>
    `;
    loadingDiv.style.cssText = `
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        z-index: 9999;
    `;
    document.body.appendChild(loadingDiv);
}

// Ẩn loading indicator
function hideLoadingIndicator() {
    const loadingDiv = document.getElementById('bcvi-loading');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Hàm hiển thị biểu đồ kích thước cụm
function createClusterSizeChart(modelName, clusterSizes) {
    // Tìm canvas element theo ID
    const canvas = document.getElementById('clusterSizeChart' + modelName);
    if (!canvas) {
        console.error('Không tìm thấy canvas cho mô hình', modelName);
        return;
    }
    
    const ctx = canvas.getContext('2d');
    
    // Kiểm tra xem clusterSizes có dữ liệu không
    if (!clusterSizes || Object.keys(clusterSizes).length === 0) {
        console.error('Không có dữ liệu kích thước cụm cho mô hình', modelName);
        // Vẽ biểu đồ thông báo lỗi
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Không có dữ liệu'],
                datasets: [{
                    data: [0],
                    backgroundColor: ['#d9534f']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                }
            }
        });
        return;
    }
    
    // Chuẩn bị dữ liệu
    const labels = Object.keys(clusterSizes).map(key => `Cụm ${key}`);
    const sizes = Object.values(clusterSizes);
    
    // Tạo biểu đồ
    const chart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: sizes,
                backgroundColor: [
                    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                    '#FF9F40', '#8BC34A', '#607D8B', '#E91E63', '#673AB7'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                }
            }
        }
    });
}