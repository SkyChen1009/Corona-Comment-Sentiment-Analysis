# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 22:08:59 2025

@author: user
"""

# === 1. 匯入套件 ===
import pandas as pd
import re, contractions
import string
import numpy as np
import torch
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from gensim.models import KeyedVectors
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from gensim.scripts.glove2word2vec import glove2word2vec
import os
from nltk.tokenize import TweetTokenizer


# === 2. 下載必要資源（只需第一次執行）===
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
# === 3. 初始化工具 ===
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
encoder = LabelEncoder()
lemmatizer = WordNetLemmatizer()
tokenizer = TweetTokenizer(preserve_case=False)
# === 4. 定義前處理函數 ===

# 建立自訂 stopwords，排除 not, no, never
default_stopwords = set(stopwords.words('english'))
custom_stopwords = default_stopwords - {"not", "no", "never"}

def clean_text(text):
    if pd.isna(text):
        return ""

    # 小寫
    text = text.lower()

    # 展開縮寫
    text = contractions.fix(text)

    # 移除網址
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # 移除 @人名（例如 @username）
    text = re.sub(r"@\w+", '', text)

    # 保留字母（排除其它符號）
    text = re.sub(r"[^a-zA-Z \s]", '', text)

    # 去除多餘空白
    text = re.sub(r"\s+", " ", text).strip()

    # 分詞
    tokens = word_tokenize(text)

    # 自訂停用詞（保留 not, no, never）
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]

    return ' '.join(filtered_tokens)



# === 5. 讀取資料集 ===
df_train = pd.read_csv("/content/drive/My Drive/深度學習/Corona_NLP_train.csv", encoding='latin_1')
df_test = pd.read_csv("/content/drive/My Drive/深度學習/Corona_NLP_test.csv", encoding='latin_1')
df_train, df_val = train_test_split(df_train, test_size=0.2, stratify=df_train["Sentiment"], random_state=42)

# === 6. 顯示基本資料統計（確認格式）===
print("訓練集資料筆數：", len(df_train))
print("訓練集欄位：", df_train.columns.tolist())
print("訓練集情緒分布：\n", df_train["Sentiment"].value_counts())

print("驗證集資料筆數：", len(df_val))
print("驗證集欄位：", df_val.columns.tolist())
print("驗證集情緒分布：\n", df_val["Sentiment"].value_counts())

print("測試集資料筆數：", len(df_test))
print("測試集欄位：", df_test.columns.tolist())
print("測試集情緒分布：\n", df_test["Sentiment"].value_counts())

# === 7. 套用前處理函數到推文文字 ===
df_train["CleanedTweet"] = df_train["OriginalTweet"].apply(clean_text)
cleanword_train = df_train["CleanedTweet"]

df_val["CleanedTweet"] = df_val["OriginalTweet"].apply(clean_text)
cleanword_val = df_val["CleanedTweet"]

df_test["CleanedTweet"] = df_test["OriginalTweet"].apply(clean_text)
cleanword_test = df_test["CleanedTweet"]


# === 8. 處理情緒標籤 ===
df_train["SentimentEncoded"] = encoder.fit_transform(df_train["Sentiment"])
df_val["SentimentEncoded"] = encoder.transform(df_val["Sentiment"])
df_test["SentimentEncoded"] = encoder.transform(df_test["Sentiment"])

# === 9. 顯示部分處理結果（供檢查） ===
print(df_train[["OriginalTweet", "CleanedTweet", "Sentiment", "SentimentEncoded"]].head())
print(df_val[["OriginalTweet", "CleanedTweet", "Sentiment", "SentimentEncoded"]].head())
print(df_test[["OriginalTweet", "CleanedTweet", "Sentiment", "SentimentEncoded"]].head())

# 1. 將已前處理好的句子轉為 token list (這部分你已經有了)
def tokenize_text(text):
    return tokenizer.tokenize(text)

train_tokens = [tokenize_text(text) for text in df_train["CleanedTweet"]]
val_tokens = [tokenize_text(text) for text in df_val["CleanedTweet"]]
test_tokens = [tokenize_text(text) for text in df_test["CleanedTweet"]]

def plot_token_lengths(token_lists, title):
    lengths = [len(tokens) for tokens in token_lists]
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Token Length")
    plt.ylabel("Number of Sentences")
    plt.show()

plot_token_lengths(train_tokens, "Train Set Token Lengths")
plot_token_lengths(val_tokens, "Validation Set Token Lengths")
plot_token_lengths(test_tokens, "Test Set Token Lengths")
# 2. 建立詞彙表（手動）
def build_manual_vocab(token_list):
    token_counts = Counter()
    for tokens in token_list:
        token_counts.update(tokens)
    sorted_tokens = [token for token, count in token_counts.items()]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, token in enumerate(sorted_tokens):
        vocab[token] = i + 2
    return vocab

vocab = build_manual_vocab(train_tokens)
# 原始 GloVe 檔案與目標 Word2Vec 檔案路徑
glove_input_file = "/content/drive/My Drive/深度學習/glove.twitter.27B.100d.txt"
word2vec_output_file = "/content/drive/My Drive/深度學習/glove.twitter.27B.100d.word2vec.txt"

# Step 1: 轉換格式（只有第一次會做）
if not os.path.exists(word2vec_output_file):
    print("🔄 檔案不存在，開始轉換 GloVe → Word2Vec 格式...")
    glove2word2vec(glove_input_file, word2vec_output_file)
else:
    print("✅ 已找到轉換後檔案，直接載入")

# Step 2: 載入轉換後的詞向量
w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print("✅ 詞向量模型載入完成，詞彙數量：", len(w2v_model))

# 設定固定隨機種子
np.random.seed(42)

embedding_dim = w2v_model.vector_size
embedding_matrix = np.zeros((len(vocab), embedding_dim))

for word, idx in vocab.items():
    if word in w2v_model:
        embedding_matrix[idx] = w2v_model[word]
    elif word == "<PAD>":
        continue  # 保持為零向量
    elif word == "<UNK>":
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))  # 隨機初始化 <UNK>
    else:
        embedding_matrix[idx] = embedding_matrix[vocab["<UNK>"]]  # 所有 OOV 使用 <UNK> 的向量



# 建立從索引到詞的映射 (itos) - 這在你初始化 embedding matrix 時會用到
itos = {v: k for k, v in vocab.items()}

def get_default_index():
    return vocab.get("<UNK>")

# 3. 將句子轉成數字序列（手動）
def tokens_to_indices_manual(tokens, vocab):
    return torch.tensor([vocab.get(token, get_default_index()) for token in tokens], dtype=torch.long)

train_indices = [tokens_to_indices_manual(tokens, vocab) for tokens in train_tokens]
val_indices = [tokens_to_indices_manual(tokens, vocab) for tokens in val_tokens]
test_indices = [tokens_to_indices_manual(tokens, vocab) for tokens in test_tokens]

print("詞彙表大小:", len(vocab))
print("訓練集第一個句子的索引:", train_indices[0])


def compute_glove_coverage(vocab, w2v_model):
    total = 0
    found = 0
    oov_words = []

    for word in vocab:
        if word in ["<PAD>", "<UNK>"]:
            continue
        total += 1
        if word in w2v_model:
            found += 1
        else:
            oov_words.append(word)

    coverage = found / total
    print(f"總詞彙數（不含特殊符號）: {total}")
    print(f"找到 GloVe 向量的詞彙數: {found}")
    print(f"覆蓋率（Coverage）: {coverage:.2%}")
    print(f"OOV 詞彙數: {len(oov_words)}")
    print(f"前 10 個 OOV 詞彙: {oov_words[:10]}")
    return oov_words

# ⚠️ 注意：這裡要放在你已經載入好 w2v_model 之後！
# 如果你尚未載入就統計會出錯。
oov_words = compute_glove_coverage(vocab, w2v_model)
total_tokens = 0
oov_tokens = 0

for sent in train_tokens:
    for token in sent:
        total_tokens += 1
        if token not in w2v_model:
            oov_tokens += 1

print(f"Token 級別的 OOV 比例: {oov_tokens / total_tokens:.2%}")


# === 4. Padding to same length ===
# === 4. 手動 Padding to same length ===

def manual_padding(indices_list, pad_value, max_len):
    padded_sequences = []
    pad_tensor = torch.tensor([pad_value], dtype=torch.long)
    for seq in indices_list:
        if len(seq) > max_len:
            seq = seq[:max_len]  # 截斷
        padding_needed = max_len - len(seq)
        padding = pad_tensor.repeat(padding_needed)
        padded_seq = torch.cat((seq, padding))
        padded_sequences.append(padded_seq)
    return torch.stack(padded_sequences)


MAX_LEN = 30
pad_value = vocab["<PAD>"]

train_padded = manual_padding(train_indices, pad_value, MAX_LEN)
val_padded = manual_padding(val_indices, pad_value, MAX_LEN)
test_padded = manual_padding(test_indices, pad_value, MAX_LEN)

print("訓練集 padding 後的 shape:", train_padded.shape)
print("驗證集 padding 後的 shape:", val_padded.shape)
print("測試集 padding 後的 shape:", test_padded.shape)
# === 5. Label tensor ===
train_labels = torch.tensor(df_train["SentimentEncoded"].values, dtype=torch.long)
val_labels = torch.tensor(df_val["SentimentEncoded"].values, dtype=torch.long)
test_labels = torch.tensor(df_test["SentimentEncoded"].values, dtype=torch.long)

print("訓練集 label tensor shape:", train_labels.shape)
print("驗證集 label tensor shape:", val_labels.shape)
print("測試集 label tensor shape:", test_labels.shape)

# ========= 超參數 (最佳結果) =========
BEST_BATCH_SIZE = 128
BEST_LR = 0.001
BEST_HIDDEN_DIM = 128
BEST_NUM_LAYERS = 4
BEST_DROPOUT = 0.2
BEST_MAX_LEN = 30
BEST_POOLING = 'max'
BEST_ACTIVATION = 'tanh'
EPOCHS_FINAL = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 定義 GRUClassifier =========
class GRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=256, output_dim=3,
                 padding_idx=0, num_layers=3, dropout=0.2,
                 pooling='max', activation=None):
        super(GRUClassifier, self).__init__()
        num_embeddings, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=False, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.pooling = pooling
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)

        if self.pooling == 'max':
            pooled, _ = output.max(dim=1)
        elif self.pooling == 'mean':
            pooled = output.mean(dim=1)
        elif self.pooling == 'hidden':
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        if self.activation:
            pooled = self.activation(pooled)

        x = self.dropout(pooled)
        x = self.norm(x)
        out = self.fc(x)
        return out

# ========= 核心函式 =========
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / len(loader.dataset)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=encoder.classes_))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return avg_loss, accuracy

# ========= 資料 Padding =========
train_padded_final = manual_padding(train_indices, pad_value=vocab["<PAD>"], max_len=BEST_MAX_LEN)
val_padded_final = manual_padding(val_indices, pad_value=vocab["<PAD>"], max_len=BEST_MAX_LEN)
test_padded_final = manual_padding(test_indices, pad_value=vocab["<PAD>"], max_len=BEST_MAX_LEN)

# ========= DataLoader =========
train_loader = DataLoader(TensorDataset(train_padded_final, train_labels),
                          batch_size=BEST_BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(val_padded_final, val_labels),
                        batch_size=BEST_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(test_padded_final, test_labels),
                         batch_size=BEST_BATCH_SIZE, shuffle=False)

# ========= 最終完整訓練流程 =========
model = GRUClassifier(
    embedding_matrix=embedding_matrix,
    hidden_dim=BEST_HIDDEN_DIM,
    output_dim=len(encoder.classes_),
    padding_idx=vocab["<PAD>"],
    num_layers=BEST_NUM_LAYERS,
    dropout=BEST_DROPOUT,
    pooling=BEST_POOLING,
    activation=BEST_ACTIVATION
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=BEST_LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

best_val_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(EPOCHS_FINAL):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS_FINAL} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ 已儲存最佳模型 (best_model.pt)")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("⏹️ Early Stopping triggered.")
            break

    torch.cuda.empty_cache()

print("\n🎯 最終訓練完成，最佳模型已保存！")


# ========= 測試集最終評估 =========
model.load_state_dict(torch.load("best_model.pt"))
model.to(device)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\n✅ [Test Result] Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")