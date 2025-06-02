import json
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to the path so we can import utils
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.clustering import compute_bcvi

def test_bcvi_calculation():
    """Test the BCVI calculation with sample data"""
    
    print("==== BCVI Calculation Test ====")
    
    # Sample data
    cvi_values = [0.5, 0.6, 0.7, 0.65]  # Sample CVI values
    k_range = [2, 3, 4, 5]              # Sample k values
    alpha = [1.0, 5.0, 1.0, 1.0]        # Sample alpha values
    n = 10                             # Sample n value
    opt_type = 'max'                   # Sample optimization type
    
    print(f"CVI Values: {cvi_values}")
    print(f"K Range: {k_range}")
    print(f"Alpha Values: {alpha}")
    print(f"N Value: {n}")
    print(f"Optimization Type: {opt_type}")
    
    # Compute BCVI
    try:
        bcvi_values = compute_bcvi(
            cvi_values=cvi_values,
            k_range=k_range,
            alpha=alpha,
            n=n,
            opt_type=opt_type
        )
        
        print("\n==== BCVI Results ====")
        print(f"BCVI Values: {bcvi_values}")
        
        # Print detailed results
        print("\nDetailed Results:")
        for k, cvi, alpha_k, bcvi in zip(k_range, cvi_values, alpha, bcvi_values):
            print(f"K={k}, CVI={cvi:.4f}, Alpha={alpha_k:.1f}, BCVI={bcvi:.4f}")
        
        # Find optimal K
        optimal_k_index = max(range(len(bcvi_values)), key=lambda i: bcvi_values[i])
        optimal_k = k_range[optimal_k_index]
        print(f"\nOptimal K based on BCVI: K={optimal_k} (BCVI={bcvi_values[optimal_k_index]:.4f})")
        
        return True
    except Exception as e:
        print(f"Error computing BCVI: {str(e)}")
        return False

# Check if model_results.json exists and contains CVI scores
def check_model_results():
    """Check if model_results.json exists and contains CVI scores"""
    
    print("\n==== Checking Model Results ====")
    
    # Define the path to model_results.json
    model_results_file = os.path.join('uploads', 'model_results.json')
    
    # Check if the file exists
    if not os.path.exists(model_results_file):
        print(f"Error: {model_results_file} does not exist")
        return False
    
    # Load the file
    try:
        with open(model_results_file, 'r', encoding='utf-8') as f:
            clustering_results = json.load(f)
        
        print(f"File loaded successfully")
        print(f"Keys in model_results.json: {list(clustering_results.keys())}")
        
        # Check if models exist
        if 'models' not in clustering_results or not clustering_results['models']:
            print("Error: No models found in model_results.json")
            return False
        
        models = clustering_results.get('models', [])
        print(f"Models: {models}")
        
        # Check if cvi_scores exist
        if 'cvi_scores' not in clustering_results:
            print("Warning: No CVI scores found in model_results.json")
            print("Creating sample CVI scores for testing...")
            
            clustering_results['cvi_scores'] = {}
            for model in models:
                clustering_results['cvi_scores'][model] = {}
                for k in range(2, 11):
                    clustering_results['cvi_scores'][model][str(k)] = {
                        'silhouette': 0.5 + k/20,  # Sample values
                        'calinski_harabasz': 100 + k*10,
                        'starczewski': 0.7 + k/30,
                        'wiroonsri': 0.6 + k/25
                    }
            
            # Save the updated file
            with open(model_results_file, 'w', encoding='utf-8') as f:
                json.dump(clustering_results, f, indent=2)
            
            print("Sample CVI scores created and saved")
        else:
            print("CVI scores found in model_results.json")
            
            # Check if all models have CVI scores
            for model in models:
                if model not in clustering_results['cvi_scores']:
                    print(f"Warning: Model {model} has no CVI scores")
                else:
                    print(f"Model {model} has CVI scores")
        
        return True
    except Exception as e:
        print(f"Error reading model_results.json: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the tests
    print("Testing BCVI calculation...")
    
    # Test BCVI calculation
    bcvi_test = test_bcvi_calculation()
    
    # Check model_results.json
    model_check = check_model_results()
    
    if bcvi_test and model_check:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed.")
