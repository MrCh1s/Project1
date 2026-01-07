import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from collections import Counter
import os

# Import Dataset from demo_pdfvqa
try:
    from demo_pdfvqa import PDFVQADataset
except ImportError:
    print("[LỖI] Không thể import PDFVQADataset từ demo_pdfvqa.py")
    print("Hãy đảm bảo file train_pdfvqa.py và demo_pdfvqa.py nằm cùng thư mục.")
    exit()

# ==========================================
# 1. SIMPLE VQA MODEL
# ==========================================
class SimpleVQA(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=128, hidden_dim=256):
        super(SimpleVQA, self).__init__()
        
        # --- Image Encoder (ResNet18) ---
        resnet = models.resnet18(pretrained=True)
        # Bỏ lớp fully connected cuối cùng
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1]) 
        self.img_dim = 512 # ResNet18 output dim
        
        # --- Text Encoder (Simple LSTM) ---
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # --- Fusion & Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(self.img_dim + hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, images, question_indices):
        # Image: [Batch, 3, 224, 224] -> [Batch, 512, 1, 1] -> [Batch, 512]
        img_features = self.image_encoder(images)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Text: [Batch, SeqLen] -> [Batch, SeqLen, Embed] -> [Batch, Hidden]
        embeds = self.embedding(question_indices)
        _, (hidden, _) = self.lstm(embeds)
        text_features = hidden[-1]
        
        # Fusion
        combined = torch.cat((img_features, text_features), dim=1)
        output = self.classifier(combined)
        return output

# ==========================================
# 2. UTILS (Vocab & Tokenizer)
# ==========================================
class SimpleTokenizer:
    def __init__(self, questions, answers=None, max_vocab=5000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        
        # Xây dựng vocab từ câu hỏi
        all_words = []
        for q in questions:
            all_words.extend(q.lower().split())
        
        # Lấy top từ phổ biến
        counts = Counter(all_words)
        for word, _ in counts.most_common(max_vocab):
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
        # Xây dựng map cho answers (Task A)
        self.ans2idx = {}
        self.idx2ans = {}
        if answers:
            # Chỉ lấy answers xuất hiện > 1 lần để giảm nhiễu
            ans_counts = Counter(answers)
            idx = 0
            for ans, _ in ans_counts.most_common(1000): # Top 1000 answers
                self.ans2idx[ans] = idx
                self.idx2ans[idx] = ans
                idx += 1
            print(f"-> Số lượng classes (answers): {len(self.ans2idx)}")

    def encode_text(self, text, max_len=20):
        tokens = text.lower().split()
        indices = [self.word2idx.get(w, 1) for w in tokens[:max_len]]
        # Padding
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    
    def encode_answer(self, answer):
        return self.ans2idx.get(answer, -1) # -1 nếu không có trong top answers

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def main():
    # Cấu hình
    BATCH_SIZE = 4 # Để nhỏ chạy cho nhanh demo
    EPOCHS = 1     # Demo 1 epoch
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Paths (Tự động tìm như demo)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    
    # Check if data suggests parent dir
    possible_roots = [script_dir, os.path.dirname(script_dir)]
    for r in possible_roots:
        if os.path.exists(os.path.join(r, "train_images")):
            base_dir = r
            print(f"[INFO] Found data root: {base_dir}")
            break
            
    qa_path = os.path.join(base_dir, "pdfvqa_taska_train_0509_clean.csv")
    layout_path = os.path.join(base_dir, "train_doc_info_visual.pkl")
    img_dir = os.path.join(base_dir, "train_images")
    
    if not os.path.exists(qa_path):
        # Fallback thử tìm file khác nếu user đang dùng Task B
        qa_path = os.path.join(base_dir, "pdfvqa_taskb_train_0503_clean.csv")
        
    print(f"Data: {os.path.basename(qa_path)}")
    print(f"Layout Path: {layout_path}, Exists: {os.path.exists(layout_path)}")

    # Load Dataset gốc
    # Transform: Resize ảnh về 224x224 cho ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = PDFVQADataset(qa_path, layout_path, img_dir, transform=transform)
    
    # DEBUG: Check loading
    if full_dataset.qa_list is None:
        print("[CRITICAL ERROR] Failed to load QA list. Checking why...")
        # Try manual load to debug
        try:
            import pandas as pd
            if os.path.exists(qa_path):
                df = pd.read_csv(qa_path)
                print(f"Manual load success. Shape: {df.shape}")
            else:
                print(f"Manual check: File not found {qa_path}")
        except Exception as e:
            print(f"Manual load error: {e}")
        exit()

    print(f"Loaded {len(full_dataset)} samples.")
    
    # Chuẩn bị Tokenizer
    print("Đang xây dựng từ điển...")
    all_questions = [item['question'] for item in full_dataset.qa_list]
    all_answers = [str(item['answer']) for item in full_dataset.qa_list]
    tokenizer = SimpleTokenizer(all_questions, all_answers)
    
    # Dataset Wrapper để trả về Tensor cho Model
    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, tokenizer):
            self.dataset = dataset
            self.tokenizer = tokenizer
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            
            # Xử lý ảnh (nếu lỗi ko load được ảnh thì trả về ảnh đen)
            image = item['image']
            if image is None:
                image = torch.zeros(3, 224, 224)
            
            # Encode Question
            q_vec = self.tokenizer.encode_text(item['question'])
            
            # Encode Answer
            a_idx = self.tokenizer.encode_answer(str(item['answer']))
            
            return image, q_vec, torch.tensor(a_idx, dtype=torch.long)

    train_data = TensorDataset(full_dataset, tokenizer)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Khởi tạo Model
    model = SimpleVQA(
        num_classes=len(tokenizer.ans2idx), 
        vocab_size=tokenizer.vocab_size
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print("\n=== Bắt đầu Training (Demo 1 Epoch) ===")
    model.train()
    
    total_loss = 0
    for i, (imgs, qs, ans) in enumerate(train_loader):
        imgs, qs, ans = imgs.to(DEVICE), qs.to(DEVICE), ans.to(DEVICE)
        
        # Bỏ qua các samples có answer không nằm trong top answers (-1)
        mask = ans != -1
        if mask.sum() == 0: continue
        
        optimizer.zero_grad()
        outputs = model(imgs[mask], qs[mask])
        loss = criterion(outputs, ans[mask])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        if i > 50: # Demo: Chỉ chạy 50 batch đầu rồi dừng
            print("-> Demo: Dừng sớm sau 50 batch.")
            break
            
    print("Training Demo hoàn tất!")

if __name__ == "__main__":
    main()
