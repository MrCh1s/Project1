import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import pandas as pd # Explicit import
from collections import Counter

# Thư viện Graph
try:
    import torch_geometric
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_GEOMETRIC = True
except ImportError:
    print("[LỖI] Chưa cài đặt torch_geometric. Hãy chạy: pip install torch-geometric")
    exit()

# Import logic load data từ demo_pdfvqa
try:
    from demo_pdfvqa import load_data
except ImportError:
    # Fallback nếu không import được
    import pickle
    import pandas as pd
    def load_data(file_path):
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f: return pickle.load(f)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        return None

# ==========================================
# 1. GRAPH DATASET
# ==========================================
class GraphVQADataset(Dataset):
    def __init__(self, qa_file, layout_file, image_dir=None, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Load Data
        print(f"Loading QA: {os.path.basename(qa_file)}")
        self.qa_data = load_data(qa_file)
        # Load Layout Data (Support merging Train/Val/Test)
        self.layout_data = {}
        layout_files = []
        
        # Check if single path or list
        if isinstance(layout_file, str):
            layout_files.append(layout_file)
            
            # Auto-detect siblings (Val/Test) if they exist in same dir
            base_dir = os.path.dirname(layout_file)
            for extra in ["val_doc_info_visual.pkl", "test_doc_info_visual.pkl"]:
                p = os.path.join(base_dir, extra)
                if os.path.exists(p) and p != layout_file:
                    layout_files.append(p)
        else:
            layout_files = layout_file
            
        for fpath in layout_files:
            if os.path.exists(fpath):
                print(f"Loading Layout: {os.path.basename(fpath)}...")
                data = load_data(fpath)
                if data:
                    self.layout_data.update(data)
                    print(f"-> Merged {len(data)} keys. Total: {len(self.layout_data)}")
            else:
                print(f"[WARN] Layout file not found: {fpath}")
        
        # Convert QA to list
        if isinstance(self.qa_data, pd.DataFrame):
            self.qa_list = self.qa_data.to_dict('records')
        else:
            self.qa_list = list(self.qa_data.values()) if isinstance(self.qa_data, dict) else self.qa_data

        # Filter: Chỉ giữ lại sample nào có layout info
        self.valid_indices = []
        
        for idx, item in enumerate(self.qa_list):
            if idx > 1000 and len(self.valid_indices) == 0: pass # Removed fail check
            
            # Resolve Image ID
            # Resolve Image ID
            image_id = str(item.get('image_id', ''))
            if not image_id or image_id == 'nan' or image_id == 'None':
                for k in ['pmcid', 'file_name', 'file', 'image_name']:
                    if k in item and pd.notna(item[k]):
                        image_id = str(item[k])
                        break
            
            if not image_id or image_id == 'nan': continue
            
            # Match Logic (Mirroring __getitem__)
            match = False
            # 1. Exact string
            if image_id in self.layout_data: match = True
            
            # 2. Integer match
            if not match:
                 try:
                     base = image_id.split('.')[0]
                     if base.isdigit() and int(base) in self.layout_data: match = True
                 except: pass
            
            # 3. Fallback string
            if not match:
                if image_id.rsplit('.', 1)[0] in self.layout_data: match = True
            
            if match:
                self.valid_indices.append(idx)
        
        if len(self.valid_indices) == 0:
             print("[WARNING] No valid samples found matching QA to Layout! (Check dataset matching)")
             # Fallback: Use all samples but with empty graph (just to let code run if user wants to test model)
             # self.valid_indices = list(range(len(self.qa_list))) 
             # Uncomment above line to force run without layout
        else:
             print(f"Valid Samples (có layout): {len(self.valid_indices)} / {len(self.qa_list)}")
                
        print(f"Final Dataset Size: {len(self.valid_indices)}")

    def len(self):
        return len(self.valid_indices)

    def get(self, idx):
        # Map idx sang index gốc
        real_idx = self.valid_indices[idx]
        item = self.qa_list[real_idx]
        
        # Resolve Image ID
        image_id = str(item.get('image_id', ''))
        if not image_id or image_id == 'nan' or image_id == 'None':
            for k in ['pmcid', 'file_name', 'file', 'image_name']:
                if k in item and pd.notna(item[k]):
                    image_id = str(item[k])
                    break
                    
        # DEBUG
        if not image_id:
            # Skip empty image info
            # print(f"[DEBUG] Skipping empty image_id at idx {idx}")
            pass
            
        # Try various conversions
        match_key = None
        
        # 1. Exact string
        if image_id in self.layout_data: match_key = image_id
        
        # 2. Integer conversion (Common in PDF-VQA pkl)
        if not match_key:
             try:
                 # Extract numeric part from '28721510.pdf_5.png' -> 28721510
                 # Or '28721510' -> 28721510
                 base_name = image_id.split('.')[0] # Take first part before (.)
                 if base_name.isdigit():
                     int_key = int(base_name)
                     if int_key in self.layout_data:
                         match_key = int_key
             except: pass
             
        # 3. Fallback string manipulations
        if not match_key:
             core = image_id.rsplit('.', 1)[0]
             if core in self.layout_data: match_key = core
             
        layout = self.layout_data.get(match_key) if match_key is not None else {}
        
        # --- XÂY DỰNG GRAPH ---
        bboxes = layout.get('bboxes', [])
        # print(f"Box len: {len(bboxes)}")
        if not bboxes or len(bboxes) == 0:
            # Dummy node nếu không có layout (để tránh lỗi)
            # print(f"[DEBUG] No bboxes for {image_id}")
            node_features = torch.zeros((1, 4), dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            # Normalize coordinates
            page_w = layout.get('width', 1000)
            page_h = layout.get('height', 1000)
            
            feats = []
            for box in bboxes:
                if len(box) == 4:
                    x1, y1, x2, y2 = box
                    # Feature: [x1/w, y1/h, x2/w, y2/h, weight/w, height/h]
                    w = x2 - x1
                    h = y2 - y1
                    feats.append([x1/page_w, y1/page_h, x2/page_w, y2/page_h])
                else:
                    feats.append([0,0,0,0])
            
            node_features = torch.tensor(feats, dtype=torch.float) # [NumBox, 4]
            
            # 2. Edges: Fully Connected (Đơn giản hóa) hoặc Spatial
            # Để demo chạy nhanh, ta dùng Fully Connected cho Layout nhỏ (< 20 boxes)
            # Hoặc tạo self-loop đơn giản.
            # Ở bài báo LoSpa, họ dùng quan hệ cha-con. Ở đây ta demo nên dùng Self-loop + Next-box connection
            num_nodes = node_features.size(0)
            
            # Kết nối i -> i+1 (flow đọc văn bản)
            source = []
            target = []
            for i in range(num_nodes - 1):
                # Hai chiều
                source.extend([i, i+1])
                target.extend([i+1, i])
            
            if len(source) == 0: # Chỉ có 1 node hoặc 0
                 edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                 edge_index = torch.tensor([source, target], dtype=torch.long)

        # Encode Question & Answer (Dùng Tokenizer)
        q_vec = self.tokenizer.encode_text(item['question'])
        a_idx = self.tokenizer.encode_answer(str(item['answer']))
        
        # Tạo PyG Data Object
        data = Data(x=node_features, edge_index=edge_index)
        data.question = q_vec.unsqueeze(0) # [1, SeqLen]
        data.y = torch.tensor([a_idx], dtype=torch.long)
        
        return data

# ==========================================
# 2. MODEL (GNN + LSTM)
# ==========================================
class GraphVQA(nn.Module):
    def __init__(self, node_dim=4, vocab_size=5000, num_classes=1000, hidden_dim=128):
        super().__init__()
        
        # Graph Branch
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.gnn1 = GCNConv(hidden_dim, hidden_dim)
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        
        # Question Branch
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        # --- Graph Process ---
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 1. Encode Node Features
        x = F.relu(self.node_encoder(x))
        
        # 2. GNN Layers
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        
        # 3. Global Pooling (Tổng hợp toàn bộ Graph thành 1 vector)
        # batch vector giúp pool đúng các node thuộc cùng 1 graph
        graph_embed = global_mean_pool(x, batch) # [BatchSize, Hidden]
        
        # --- Question Process ---
        # data.question có dạng [Batch*1, SeqLen] do PyG collate
        # Tuy nhiên PyG collate custom attribute hơi khác, ta cần reshape
        # Vì ta gán data.question unsqueeze(0) ở dataset, Collate sẽ nối thành [Batch, 1, SeqLen] hoặc [Batch, SeqLen]
        q_input = data.question 
        if q_input.dim() == 3: q_input = q_input.squeeze(1)
        
        embeds = self.embedding(q_input)
        _, (hidden, _) = self.lstm(embeds)
        text_embed = hidden[-1] # [Batch, Hidden]
        
        # --- Fusion ---
        combined = torch.cat([graph_embed, text_embed], dim=1)
        out = self.classifier(combined)
        
        return out

# ==========================================
# 3. UTILS (Tokenizer Reuse)
# ==========================================
# Copy SimpleTokenizer từ train_pdfvqa.py để không phụ thuộc file
class SimpleTokenizer:
    def __init__(self, questions, answers=None, max_vocab=5000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        all_words = []
        for q in questions: all_words.extend(q.lower().split())
        for word, _ in Counter(all_words).most_common(max_vocab):
            self.word2idx[word] = self.vocab_size
            self.vocab_size += 1
        self.ans2idx = {}
        if answers:
            idx = 0
            for ans, _ in Counter(answers).most_common(1000):
                self.ans2idx[ans] = idx; idx += 1

    def encode_text(self, text, max_len=20):
        tokens = text.lower().split()
        indices = [self.word2idx.get(w, 1) for w in tokens[:max_len]]
        if len(indices) < max_len: indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    
    def encode_answer(self, answer):
        return self.ans2idx.get(answer, -1)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Path Logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    possible_roots = [script_dir, os.path.dirname(script_dir)]
    for r in possible_roots:
        if os.path.exists(os.path.join(r, "train_images")):
            base_dir = r; break
            
    qa_path = os.path.join(base_dir, "train_dataframe.pkl")
    layout_path = os.path.join(base_dir, "train_doc_info_visual.pkl")
    
    # 1. Prepare Tokenizer & Data info
    # Để tránh load lại dataset 2 lần, ta load data raw trước để build vocab
    raw_qa = load_data(qa_path)
    if isinstance(raw_qa, pd.DataFrame): raw_list = raw_qa.to_dict('records')
    else: raw_list = list(raw_qa.values())
    
    print("Building Tokenizer...")
    all_q = [x['question'] for x in raw_list]
    all_a = [str(x['answer']) for x in raw_list]
    tokenizer = SimpleTokenizer(all_q, all_a)
    print(f"Vocab: {tokenizer.vocab_size}, Classes: {len(tokenizer.ans2idx)}")
    
    # 2. Init Dataset
    dataset = GraphVQADataset(qa_path, layout_path, tokenizer=tokenizer)
    
    # PyTorch Geometric DataLoader
    loader = GeoDataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Model
    model = GraphVQA(
        vocab_size=tokenizer.vocab_size, 
        num_classes=len(tokenizer.ans2idx)
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("\n=== START TRAINING GRAPH MODEL (LoSpa Baseline) ===")
    model.train()
    
    for i, batch in enumerate(loader):
        batch = batch.to(DEVICE)
        target = batch.y
        
        # Mask valid answers
        mask = target != -1
        if mask.sum() == 0: continue
        
        optimizer.zero_grad()
        out = model(batch)
        
        # Chỉ tính loss trên các mẫu có valid answer
        loss = criterion(out[mask], target[mask])
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Batch {i} | Loss: {loss.item():.4f}")
            
        if i > 50: 
            print("Done Demo 50 batches.")
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FATAL ERROR] {e}")
