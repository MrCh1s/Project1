import pickle
import os
import pandas as pd

def load_data(file_path):
    print(f"Loading {file_path}...")
    if not os.path.exists(file_path):
        print("File not found.")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    base_dir = os.getcwd()
    # Try parent dir if not found
    if not os.path.exists("train_dataframe.pkl"):
        base_dir = os.path.join(base_dir, "..")
        
    df_path = os.path.join(base_dir, "train_dataframe.pkl")
    layout_path = os.path.join(base_dir, "train_doc_info_visual.pkl")
    
    # 1. Load Keys from Layout
    layout_data = load_data(layout_path)
    if layout_data:
        layout_keys = set(str(k) for k in layout_data.keys())
        print(f"Layout Keys: {len(layout_keys)}")
        print(f"Sample Layout Keys: {list(layout_keys)[:5]}")
    else:
        return

    # 2. Load IDs from DataFrame
    df_data = load_data(df_path)
    if df_data is not None:
        if isinstance(df_data, pd.DataFrame):
            # Check column
            id_col = 'image_id' if 'image_id' in df_data.columns else 'pmcid'
            
            if id_col in df_data.columns:
                print(f"Using ID column: {id_col}")
                df_ids = set(df_data[id_col].astype(str))
                print(f"DataFrame IDs: {len(df_ids)}")
                print(f"Sample DF IDs: {list(df_ids)[:5]}")
                
                # 3. Intersection
                common = layout_keys.intersection(df_ids)
                print(f"Intersection Count: {len(common)}")
                if common:
                    print(f"Sample Common: {list(common)[:5]}")
                else:
                    print("NO OVERLAP FOUND.")
            else:
                print("No 'image_id' or 'pmcid' column in DataFrame.")
                print(df_data.columns)
        else:
            print("Dataframe file is not a pandas DataFrame.")
            print(type(df_data))

if __name__ == "__main__":
    main()
