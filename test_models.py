import os
import json
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def check_model_data():
    try:
        model_results_file = os.path.join('uploads', 'model_results.json')
        if not os.path.exists(model_results_file):
            logging.error(f"File {model_results_file} not found")
            return
        
        with open(model_results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.debug(f"Keys in model_results.json: {list(data.keys())}")
        
        if 'models' in data:
            logging.debug(f"Models: {data['models']}")
        else:
            logging.error("No 'models' key found in model_results.json")
        
        if 'selected_k' in data:
            logging.debug(f"Selected K: {data['selected_k']}")
        else:
            logging.error("No 'selected_k' key found in model_results.json")
            
        if 'cvi_scores' in data:
            logging.debug(f"CVI scores models: {list(data['cvi_scores'].keys())}")
            
            # Check first model's CVI scores structure
            if data['models']:
                first_model = data['models'][0]
                if first_model in data.get('cvi_scores', {}):
                    logging.debug(f"CVI scores for {first_model}: {list(data['cvi_scores'][first_model].keys())}")
                    
                    # Check the structure of CVI scores for first K
                    if data['cvi_scores'][first_model]:
                        first_k = list(data['cvi_scores'][first_model].keys())[0]
                        logging.debug(f"CVI score types for {first_model}, K={first_k}: {list(data['cvi_scores'][first_model][first_k].keys())}")
                    else:
                        logging.error(f"No K values in CVI scores for {first_model}")
                else:
                    logging.error(f"Model {first_model} not found in CVI scores")
        else:
            logging.error("No 'cvi_scores' key found in model_results.json")
            
        # Check if data is correct for BCVI calculation
        for model in data.get('models', []):
            if model in data.get('cvi_scores', {}):
                model_cvi = data['cvi_scores'][model]
                cvi_types = ['silhouette', 'calinski_harabasz', 'starczewski', 'wiroonsri']
                
                for k_str in sorted(model_cvi.keys(), key=int):
                    k = int(k_str)
                    if k >= 2:
                        missing_cvi_types = []
                        for cvi_type in cvi_types:
                            if cvi_type not in model_cvi[k_str] or model_cvi[k_str].get(cvi_type, 0) == 0:
                                missing_cvi_types.append(cvi_type)
                        
                        if missing_cvi_types:
                            logging.warning(f"Model {model}, K={k} is missing CVI types: {missing_cvi_types}")

if __name__ == "__main__":
    check_model_data()
