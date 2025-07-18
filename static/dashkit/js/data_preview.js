document.addEventListener('DOMContentLoaded', function() {
    // Kiểm tra dropdown trường dữ liệu
    const allFeatureSelect = document.getElementById('allFeatureSelect');
    if (!allFeatureSelect || allFeatureSelect.options.length === 0) {
        console.log('Không có trường dữ liệu để hiển thị biểu đồ');
        return;
    }

    let histogramChart = null;

    function updateHistogram(feature) {
        let ctx = document.getElementById('allHistogramCanvas');
        if (!ctx) {
            const container = document.getElementById('allHistogramContainer');
            if (container) {
                container.innerHTML = '<canvas id="allHistogramCanvas" width="400" height="300"></canvas>';
                ctx = document.getElementById('allHistogramCanvas');
            }
        }
        if (histogramChart) {
            histogramChart.destroy();
        }
        // Lấy dữ liệu thực từ window.all_histograms
        if (typeof window.all_histograms !== 'undefined' && window.all_histograms[feature]) {
            const distribution = window.all_histograms[feature];
            if (distribution.labels.length > 0 && ctx) {
                histogramChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: distribution.labels,
                        datasets: [{
                            label: 'Tần suất của ' + feature,
                            data: distribution.values,
                            backgroundColor: 'rgba(255, 99, 132, 0.6)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: true }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: { display: true, text: 'Tần số' }
                            },
                            x: {
                                title: { display: true, text: 'Giá trị' }
                            }
                        }
                    }
                });
            } else if (ctx) {
                ctx.parentNode.innerHTML = '<div class="text-danger">Không có dữ liệu để hiển thị biểu đồ cho trường này.</div>';
            }
        } else if (ctx) {
            ctx.parentNode.innerHTML = '<div class="text-danger">Không có dữ liệu để hiển thị biểu đồ cho trường này.</div>';
        }
    }

    // Event listener cho dropdown
    allFeatureSelect.addEventListener('change', function() {
        updateHistogram(this.value);
    });

    // Khởi tạo với trường đầu tiên
    if (allFeatureSelect.options.length > 0) {
        updateHistogram(allFeatureSelect.options[0].value);
    }
});