document.addEventListener('DOMContentLoaded', function() {
    // Kiểm tra xem có numerical features không
    const featureSelect = document.getElementById('featureSelect');
    if (!featureSelect || featureSelect.options.length === 0) {
        console.log('Không có dữ liệu số để hiển thị biểu đồ');
        return;
    }

    // Tạo mock data cho histogram
    const mockDistributions = {};
    
    // Lấy danh sách features từ select options
    for (let i = 0; i < featureSelect.options.length; i++) {
        const feature = featureSelect.options[i].value;
        mockDistributions[feature] = {
            labels: ['0-10', '10-20', '20-30', '30-40', '40-50', '50+'],
            values: [
                Math.floor(Math.random() * 100),
                Math.floor(Math.random() * 100),
                Math.floor(Math.random() * 100),
                Math.floor(Math.random() * 100),
                Math.floor(Math.random() * 100),
                Math.floor(Math.random() * 100)
            ]
        };
    }
    
    let histogramChart = null;
    
    function updateHistogram(feature) {
        let ctx = document.getElementById('histogramCanvas');
        
        if (!ctx) {
            const container = document.getElementById('histogramContainer');
            if (container) {
                container.innerHTML = '<canvas id="histogramCanvas" width="400" height="300"></canvas>';
                ctx = document.getElementById('histogramCanvas');
            }
        }
        
        if (histogramChart) {
            histogramChart.destroy();
        }
        
        const distribution = mockDistributions[feature];
        if (distribution && ctx) {
            histogramChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: distribution.labels,
                    datasets: [{
                        label: 'Phân phối của ' + feature,
                        data: distribution.values,
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Tần số'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Giá trị'
                            }
                        }
                    }
                }
            });
        }
    }
    
    // Event listener cho dropdown
    featureSelect.addEventListener('change', function() {
        updateHistogram(this.value);
    });
    
    // Khởi tạo với feature đầu tiên
    if (featureSelect.options.length > 0) {
        updateHistogram(featureSelect.options[0].value);
    }
});