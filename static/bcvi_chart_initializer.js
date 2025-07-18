/**
 * bcvi_chart_initializer.js - Khởi tạo các biểu đồ BCVI từ dữ liệu JSON
 */

// Khởi tạo các biểu đồ từ dữ liệu được truyền vào
function initializeBCVICharts(modelsData) {
    console.log('[BCVI Chart Initializer] Starting chart initialization');
    console.log('[BCVI Chart Initializer] Models data:', modelsData);
    
    // Kiểm tra hàm vẽ biểu đồ
    if (typeof createCVIComparisonChart !== 'function' || 
        typeof createBCVIComparisonChart !== 'function') {
        console.error('[BCVI Chart Initializer] Chart drawing functions not available!');
        return;
    }
    
    // Duyệt qua từng model
    for (const model in modelsData) {
        if (!modelsData.hasOwnProperty(model)) continue;
        
        const modelData = modelsData[model];
        console.log(`[BCVI Chart Initializer] Processing model: ${model}`);
        
        try {
            // Vẽ biểu đồ so sánh BCVI
            drawBCVIComparisonChart(model, modelData);
            
            // Vẽ biểu đồ CVI cho từng loại
            for (const cviType in modelData) {
                if (!modelData.hasOwnProperty(cviType)) continue;
                drawCVIChart(model, cviType, modelData[cviType]);
            }
        } catch (error) {
            console.error(`[BCVI Chart Initializer] Error drawing charts for ${model}:`, error);
        }
    }
}

// Vẽ biểu đồ so sánh BCVI
function drawBCVIComparisonChart(model, modelData) {
    const canvasId = `${model}-bcvi-comparison-chart`;
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`[BCVI Chart Initializer] Canvas not found: ${canvasId}`);
        return;
    }
    
    console.log(`[BCVI Chart Initializer] Drawing BCVI comparison chart for ${model}`);
    
    // Thu thập tất cả K-values
    const kValues = new Set();
    for (const cviType in modelData) {
        if (!modelData.hasOwnProperty(cviType)) continue;
        
        modelData[cviType].forEach(item => {
            if (item.k) kValues.add(item.k);
        });
    }
    
    // Chuyển về mảng và sắp xếp
    const sortedKValues = Array.from(kValues).sort((a, b) => a - b);
    
    // Gọi hàm vẽ biểu đồ
    createBCVIComparisonChart(
        canvasId,
        sortedKValues,
        modelData,
        `So sánh các chỉ số BCVI - ${model}`
    );
}

// Vẽ biểu đồ CVI cho từng loại
function drawCVIChart(model, cviType, cviData) {
    const canvasId = `${model}-${cviType}-cvi-comparison-chart`;
    const canvas = document.getElementById(canvasId);
    
    if (!canvas) {
        console.error(`[BCVI Chart Initializer] Canvas not found: ${canvasId}`);
        return;
    }
    
    console.log(`[BCVI Chart Initializer] Drawing CVI chart for ${model}-${cviType}`);
    
    // Thu thập K-values và CVI values
    const kValues = [];
    const values = [];
    
    cviData.forEach(item => {
        if (item.k) kValues.push(item.k);
        if (item.cvi !== undefined) values.push(item.cvi);
    });
    
    // Tạo đối tượng dữ liệu
    const data = {
        [cviType]: values
    };
    
    // Gọi hàm vẽ biểu đồ
    createCVIComparisonChart(
        canvasId,
        kValues,
        data,
        `Biểu đồ CVI - ${model} (${cviType.toUpperCase()})`
    );
}
