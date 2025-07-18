/**
 * bcvi_chart_comparison.js
 * Script để hiển thị biểu đồ so sánh các chỉ số BCVI và CVI
 */

// Biến toàn cục để lưu trữ các biểu đồ đã tạo
let bcviComparisonCharts = {};
let cviComparisonCharts = {};

/**
 * Tạo biểu đồ so sánh BCVI cho một mô hình
 * @param {string} modelName - Tên của mô hình
 * @param {object} bcviData - Dữ liệu BCVI cho mô hình
 */
function createBCVIComparisonChart(modelName, bcviData) {
    // Kiểm tra xem có dữ liệu BCVI cho mô hình hay không
    if (!bcviData || Object.keys(bcviData).length === 0) {
        console.error(`Không có dữ liệu BCVI cho mô hình ${modelName}`);
        return;
    }

    // ID của canvas để vẽ biểu đồ
    const canvasId = `${modelName}-bcvi-comparison-chart`;
    const chartCanvas = document.getElementById(canvasId);
    
    if (!chartCanvas) {
        console.error(`Không tìm thấy canvas với ID ${canvasId}`);
        return;
    }

    // Kiểm tra và hủy biểu đồ cũ nếu có
    if (bcviComparisonCharts[modelName]) {
        bcviComparisonCharts[modelName].destroy();
    }

    // Chuẩn bị dữ liệu cho biểu đồ
    const datasets = [];
    const kValues = new Set();
    const cviTypes = Object.keys(bcviData);

    // Màu sắc cho từng loại CVI
    const colors = {
        'silhouette': 'rgb(75, 192, 192)',
        'calinski_harabasz': 'rgb(255, 99, 132)',
        'starczewski': 'rgb(255, 205, 86)',
        'wiroonsri': 'rgb(54, 162, 235)'
    };

    // Tạo datasets cho biểu đồ
    cviTypes.forEach(cviType => {
        if (bcviData[cviType] && bcviData[cviType].length > 0) {
            // Dữ liệu BCVI cho loại CVI này
            const data = bcviData[cviType].map(item => ({
                x: item.k,
                y: item.bcvi,
                alpha: item.alpha,
                cvi: item.cvi
            }));

            // Thêm tất cả các giá trị k vào tập hợp
            data.forEach(item => kValues.add(item.x));

            // Thêm dataset cho loại CVI này
            datasets.push({
                label: `BCVI (${cviType.toUpperCase()})`,
                data: data,
                borderColor: colors[cviType] || `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`,
                backgroundColor: colors[cviType] ? colors[cviType].replace('rgb', 'rgba').replace(')', ', 0.2)') : `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 0.2)`,
                borderWidth: 2,
                tension: 0.2,
                pointStyle: 'circle',
                pointRadius: 5,
                pointHoverRadius: 8
            });
        }
    });

    // Tạo biểu đồ
    bcviComparisonCharts[modelName] = new Chart(chartCanvas, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `So sánh chỉ số BCVI giữa các phương pháp cho ${modelName}`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label || '';
                            const value = context.raw.y.toFixed(3);
                            const alpha = context.raw.alpha.toFixed(2);
                            const cvi = context.raw.cvi.toFixed(3);
                            return [
                                `${label}: ${value}`,
                                `Alpha: ${alpha}`,
                                `CVI gốc: ${cvi}`
                            ];
                        }
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 12
                        },
                        usePointStyle: true
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Số cụm (k)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            if (Number.isInteger(value) && value >= Math.min(...kValues) && value <= Math.max(...kValues)) {
                                return value;
                            }
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Giá trị BCVI',
                        font: {
                            weight: 'bold'
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                intersect: false
            }
        }
    });
}

/**
 * Tạo biểu đồ so sánh CVI gốc cho một mô hình
 * @param {string} modelName - Tên của mô hình
 * @param {object} bcviData - Dữ liệu BCVI cho mô hình (chứa cả CVI gốc)
 */
function createCVIComparisonChart(modelName, bcviData) {
    // Kiểm tra xem có dữ liệu BCVI cho mô hình hay không (dữ liệu này có chứa CVI gốc)
    if (!bcviData || Object.keys(bcviData).length === 0) {
        console.error(`Không có dữ liệu CVI cho mô hình ${modelName}`);
        return;
    }

    // ID của canvas để vẽ biểu đồ CVI
    const canvasId = `${modelName}-cvi-comparison-chart`;
    const chartCanvas = document.getElementById(canvasId);
    
    if (!chartCanvas) {
        console.error(`Không tìm thấy canvas với ID ${canvasId}`);
        return;
    }

    // Kiểm tra và hủy biểu đồ cũ nếu có
    if (cviComparisonCharts[modelName]) {
        cviComparisonCharts[modelName].destroy();
    }

    // Chuẩn bị dữ liệu cho biểu đồ
    const datasets = [];
    const kValues = new Set();
    const cviTypes = Object.keys(bcviData);

    // Màu sắc cho từng loại CVI - khác với màu BCVI để dễ phân biệt
    const colors = {
        'silhouette': 'rgb(25, 142, 142)',
        'calinski_harabasz': 'rgb(205, 49, 82)',
        'starczewski': 'rgb(205, 155, 36)',
        'wiroonsri': 'rgb(24, 112, 185)'
    };

    // Tạo datasets cho biểu đồ CVI
    cviTypes.forEach(cviType => {
        if (bcviData[cviType] && bcviData[cviType].length > 0) {
            // Dữ liệu CVI gốc cho loại CVI này
            const data = bcviData[cviType].map(item => ({
                x: item.k,
                y: item.cvi,  // Sử dụng giá trị CVI gốc thay vì BCVI
                alpha: item.alpha,
                bcvi: item.bcvi
            }));

            // Thêm tất cả các giá trị k vào tập hợp
            data.forEach(item => kValues.add(item.x));

            // Thêm dataset cho loại CVI này
            datasets.push({
                label: `${cviType.toUpperCase()}`,
                data: data,
                borderColor: colors[cviType] || `rgb(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255})`,
                backgroundColor: colors[cviType] ? colors[cviType].replace('rgb', 'rgba').replace(')', ', 0.2)') : `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 0.2)`,
                borderWidth: 2,
                tension: 0.2,
                pointStyle: 'diamond',
                pointRadius: 5,
                pointHoverRadius: 8,
                borderDash: cviType === 'silhouette' ? [] : [5, 5]  // Silhouette dùng nét liền, các CVI khác dùng nét đứt
            });
        }
    });

    // Tạo biểu đồ
    cviComparisonCharts[modelName] = new Chart(chartCanvas, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `So sánh các chỉ số CVI gốc cho ${modelName}`,
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            const label = context.dataset.label || '';
                            const value = context.raw.y.toFixed(3);
                            const alpha = context.raw.alpha.toFixed(2);
                            const bcvi = context.raw.bcvi.toFixed(3);
                            return [
                                `${label}: ${value}`,
                                `Alpha: ${alpha}`,
                                `BCVI: ${bcvi}`
                            ];
                        }
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        font: {
                            size: 12
                        },
                        usePointStyle: true
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Số cụm (k)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        stepSize: 1,
                        callback: function(value) {
                            if (Number.isInteger(value) && value >= Math.min(...kValues) && value <= Math.max(...kValues)) {
                                return value;
                            }
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Giá trị CVI',
                        font: {
                            weight: 'bold'
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                intersect: false
            }
        }
    });
}

/**
 * Khởi tạo tất cả các biểu đồ BCVI khi trang được tải
 */
function initializeBCVICharts() {
    // Lắng nghe sự kiện click trên các tab để cập nhật biểu đồ
    document.addEventListener('DOMContentLoaded', function() {
        // Tìm tất cả các tab của mô hình
        const modelTabs = document.querySelectorAll('button[data-bs-toggle="tab"][data-bs-target^="#"][role="tab"]');
        
        modelTabs.forEach(tab => {
            tab.addEventListener('shown.bs.tab', function(event) {
                const targetId = event.target.getAttribute('data-bs-target');
                if (targetId) {
                    const modelName = targetId.substring(1).split('-')[0]; // Lấy tên mô hình từ ID tab
                    
                    // Kiểm tra xem có dữ liệu BCVI không
                    if (window.bcviResults && window.bcviResults[modelName]) {
                        // Tạo biểu đồ cho mô hình đang active
                        createBCVIComparisonChart(modelName, window.bcviResults[modelName]);
                        createCVIComparisonChart(modelName, window.bcviResults[modelName]);
                    }
                }
            });
        });
        
        // Tự động tạo biểu đồ cho tab đang active khi trang được tải
        const activeTab = document.querySelector('button[data-bs-toggle="tab"][role="tab"].active');
        if (activeTab) {
            const targetId = activeTab.getAttribute('data-bs-target');
            if (targetId) {
                const modelName = targetId.substring(1).split('-')[0];
                
                // Kiểm tra xem có dữ liệu BCVI không
                if (window.bcviResults && window.bcviResults[modelName]) {
                    // Tạo biểu đồ cho mô hình đang active
                    setTimeout(() => {
                        createBCVIComparisonChart(modelName, window.bcviResults[modelName]);
                        createCVIComparisonChart(modelName, window.bcviResults[modelName]);
                    }, 300); // Chờ một chút để đảm bảo DOM đã sẵn sàng
                }
            }
        }
    });
}

// Khởi tạo khi trang được tải
initializeBCVICharts();
