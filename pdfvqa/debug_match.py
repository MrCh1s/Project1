import pickle
import pandas as pd
import os

def load_pkl(path):
    print(f"Loading PKL: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_csv(path):
    print(f"Loading CSV: {path}")
    return pd.read_csv(path)

basedir = os.path.dirname(os.path.abspath(__file__))
# Auto find paths
if not os.path.exists("train_doc_info_visual.pkl"):
    # Try parent
    parent = os.path.dirname(basedir)
    if os.path.exists(os.path.join(parent, "train_doc_info_visual.pkl")):
        basedir = parent
        
pkl_path = os.path.join(basedir, "train_doc_info_visual.pkl")
csv_path = os.path.join(basedir, "pdfvqa_taska_train_0509_clean.csv")

if not os.path.exists(csv_path):
    csv_path = os.path.join(basedir, "pdfvqa_taskb_train_0503_clean.csv")

try:
    layout_data = load_pkl(pkl_path)
    qa_data = load_csv(csv_path)
    
    print("\n--- INSPECTION ---")
    keys = list(layout_data.keys())
    print(f"PKL Keys ({len(keys)} total): {keys[:5]}")
    print(f"PKL Key Type: {type(keys[0])}")
    
    print("\n--- QA DATA ---")
    qa_list = qa_data.to_dict('records')
    if len(qa_list) > 0:
        print(f"Sample 0 Keys: {list(qa_list[0].keys())}")
        
    # Handle column name
    sample_id = None
    for k in ['image_id', 'file_name', 'file', 'image_name']:
        if k in qa_list[0]:
            sample_id = str(qa_list[0][k])
            print(f"Found ID column: '{k}'")
            break
            
    if not sample_id:
        print("CRITICAL: No ID column found!")
        exit()
        
    print(f"QA Image ID (str): '{sample_id}'")
    
    print("\n--- MATCHING TEST ---")
    
    def try_match(img_id, layout_keys):
        # 1. Exact
        if img_id in layout_keys: return f"Exact match: {img_id}"
        
        # 2. Int
        base = img_id.split('.')[0]
        if base.isdigit():
            int_key = int(base)
            if int_key in layout_keys: return f"Int match: {int_key}"
            
        return "No match"
        
    res = try_match(sample_id, layout_data)
    print(f"Result for '{sample_id}': {res}")
    
    # Check if ANY match
    print("\nChecking first 100 samples match rate...")
    matches = 0
    for item in qa_list[:100]:
        iid = str(item['image_id'])
        if try_match(iid, layout_data) != "No match":
            matches += 1
            
    print(f"Matches in first 100: {matches}/100")

except Exception as e:
    print(f"Error: {e}")
