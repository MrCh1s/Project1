import pickle
import os

target_file = "train_doc_info_visual.pkl"

# Search logic
if not os.path.exists(target_file):
    if os.path.exists(os.path.join("..", target_file)):
        target_file = os.path.join("..", target_file)

print(f"Loading {target_file}...")
try:
    with open(target_file, 'rb') as f:
        data = pickle.load(f)
        
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys (first 5): {list(data.keys())[:5]}")
        first_key = list(data.keys())[0]
        print(f"Sample Value ({first_key}): {data[first_key]}")
    elif isinstance(data, list):
        print(f"Length: {len(data)}")
        print(f"Sample Item (0): {data[0]}")
except Exception as e:
    print(f"Error: {e}")
