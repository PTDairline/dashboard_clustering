import json
import sys
import os

# Thêm đường dẫn để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.metrics import suggest_optimal_k

def test_suggest_optimal_k():
    """Test function suggest_optimal_k with actual model_results.json data"""
    
    # Đọc dữ liệu từ file model_results.json
    try:
        with open('uploads/model_results.json', 'r') as f:
            data = json.load(f)
            
        print("Dữ liệu đã tải từ model_results.json:")
        print(f"Models: {data.get('models', [])}")
        print(f"K range: {data.get('k_range', [])}")
        print(f"Selected k: {data.get('selected_k', 0)}")
        
        # Kiểm tra cấu trúc CVI scores
        if 'cvi_scores' in data:
            print("\nCấu trúc CVI scores hiện tại:")
            for model in data['models']:
                if model in data['cvi_scores']:
                    print(f"  {model}: {list(data['cvi_scores'][model].keys())}")
        
        # Chuyển đổi dữ liệu sang định dạng mà suggest_optimal_k mong đợi
        for model in data['models']:
            if model in data['cvi_scores']:
                print(f"\n--- Testing suggest_optimal_k for {model} ---")
                
                # Tạo plots dictionary với cấu trúc đúng
                plots = {
                    'cvi': [],
                    'silhouette': {'scores': [], 'plot': None},
                    'elbow': {'inertias': [], 'plot': None},
                    'scatter': []
                }
                
                # Chuyển đổi CVI scores
                for k_str in sorted(data['cvi_scores'][model].keys(), key=int):
                    k = int(k_str)
                    cvi_entry = data['cvi_scores'][model][k_str]
                    plots['cvi'].append({
                        'k': k,
                        'Silhouette': cvi_entry['silhouette'],
                        'Calinski-Harabasz': cvi_entry['calinski_harabasz'],
                        'Davies-Bouldin': cvi_entry['davies_bouldin'],
                        'Starczewski': cvi_entry['starczewski'],
                        'Wiroonsri': cvi_entry['wiroonsri']
                    })
                    plots['silhouette']['scores'].append(cvi_entry['silhouette'])
                
                # Thêm elbow data nếu có
                if 'silhouette_scores' in data and model in data['silhouette_scores']:
                    plots['silhouette']['scores'] = data['silhouette_scores'][model]
                
                if 'elbow_inertias' in data and model in data['elbow_inertias']:
                    plots['elbow']['inertias'] = data['elbow_inertias'][model]
                
                print(f"Plots structure created for {model}:")
                print(f"  CVI entries: {len(plots['cvi'])}")
                print(f"  Silhouette scores: {len(plots['silhouette']['scores'])}")
                print(f"  Elbow inertias: {len(plots['elbow']['inertias'])}")
                
                # Test suggest_optimal_k function
                try:
                    k_range = data['k_range']
                    
                    # Test với use_wiroonsri_starczewski=False
                    optimal_k1, reasoning1 = suggest_optimal_k(plots, k_range, use_wiroonsri_starczewski=False)
                    print(f"\nKết quả với Silhouette & Elbow:")
                    print(f"  Optimal k: {optimal_k1}")
                    print(f"  Reasoning: {reasoning1}")
                    
                    # Test với use_wiroonsri_starczewski=True
                    optimal_k2, reasoning2 = suggest_optimal_k(plots, k_range, use_wiroonsri_starczewski=True)
                    print(f"\nKết quả với Wiroonsri & Starczewski:")
                    print(f"  Optimal k: {optimal_k2}")
                    print(f"  Reasoning: {reasoning2}")
                    
                except Exception as e:
                    print(f"LỖI khi gọi suggest_optimal_k cho {model}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                print("-" * 60)
    
    except Exception as e:
        print(f"LỖI khi đọc hoặc xử lý file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_suggest_optimal_k()
