# Corona Comment Sentiment Analysis
### 日期與時間: 2025年5月2日 23:59

## 專案概述
本專案使用雙向LSTM（BiLSTM）模型結合Word2Vec詞向量技術，對與新冠疫情相關的社交媒體留言進行情感分析。數據集為 Corona_NLP_train.csv 和 Corona_NLP_test.csv，包含推文內容及其情感標籤（例如正面、負面、中立等）。本專案旨在通過自然語言處理（NLP）技術，分析疫情期間公眾的情感傾向。

## 檔案結構
- BiLSTM_Comment_SentimentAnalysis.ipynb: 主程式碼檔案，包含數據預處理、Word2Vec詞向量生成、模型構建、訓練與視覺化等步驟。
- Corona_NLP_train.csv: 訓練數據集，包含推文及情感標籤。
- Corona_NLP_test.csv: 測試數據集，用於模型評估。
R- EADME.md: 本說明文件。

## 目前進度
截至2025年5月2日，專案已完成以下步驟（參考 BiLSTM_Comment_SentimentAnalysis.ipynb）：
1. 套件載入: 已成功導入必要的Python套件，包括 torch、pandas、nltk、gensim 等，並確認GPU可用（torch.cuda.is_available() 輸出為 True）。
2. 數據讀取: 已載入並顯示 Corona_NLP_train.csv 和 Corona_NLP_test.csv 的前5筆數據，確認數據格式正確。
3. 數據清理: 對推文進行預處理，包括去除URL、轉小寫、移除標點符號、分詞、去除停用詞及非英文單詞。清理後的結果已儲存在 clean_tokens 欄位。
4. Word2Vec向量化: 已使用 gensim 訓練Word2Vec模型，將清理後的單詞轉換為詞向量，並將推文轉換為固定長度的索引序列（input_indices），準備用於模型輸入。
5. 模型訓練與視覺化: 已完成模型訓練，並生成訓練/驗證損失曲線及驗證準確率曲線。然而，訓練過程顯示以下問題：
6. 低準確率: 驗證準確率僅為 0.67，表明模型性能不佳。
8. 字體錯誤: 繪圖時出現 findfont: Font family 'WenQuanYi Micro Hei' not found 錯誤，影響中文標籤的顯示（已將圖表標籤改為英文以繞過問題）。
9. 訓練速度慢: 需用GPU 跑，用CPU 會跑非常久。

## 下一步計劃
- 強化分詞功能，讓分詞後的tokenized series 更乾淨。
- 調整模型超參數（如學習率、隱藏層大小、批量大小）。
- 考慮使用預訓練詞向量（如 GloVe 或 BERT）替代自訓練的Word2Vec。
- 現在仍有overfitting 問題，從步驟七的圖表可以得知。

## 0505 完成進度
- hyperparameter 的微調為以下：
- 添加超參數調優（步驟5.5）：
- 1. 在步驟5.5中引入了基於 ParameterGrid 的網格搜索，針對以下超參數進行調優：
- 2. 學習率（learning_rate）：測試 [0.001, 0.0001]。
- 3. 隱藏層維度（hidden_dim）：測試 [128, 256]。
- 4. Dropout 比率（dropout）：測試 [0.3, 0.5]。
- 5. 批次大小（batch_size）：測試 [32, 64]。
- 6. LSTM 層數（num_layers）：測試 [1, 2]。
- 7. 通過訓練和驗證集上的表現（驗證準確率）選出最佳參數組合，更新了 BATCH_SIZE、HIDDEN_DIM、DROPOUT 和 NUM_LAYERS，並將最佳學習率應用於最終模型訓練。
```
Best parameters: {'batch_size': 32, 'dropout': 0.3, 'hidden_dim': 128, 'learning_rate': 0.001, 'num_layers': 1}, Validation Accuracy: 0.7031952375167051
```
- BiLSTM 的增強：
- 1. 嵌入層 Dropout：在嵌入層輸出後添加了 Dropout（embedded = self.dropout(embedded)），靈感來自 BiLSTM 的 self.dropout(self.embedding(text))。
- 2. 可選的隱藏狀態拼接：新增 use_hidden_states 參數，允許選擇使用 LSTM 的隱藏狀態（hidden[-2,:,:] 和 hidden[-1,:,:]，如 BiLSTM）或原始的序列輸出（lstm_out[:, -1, :] 和 lstm_out[:, 0, :]）進行拼接。
 
## 0506 完成進度：
- 1. 用GloVe 取代word2Vec，acc 平均來到68%。
- 2. 目前overfitting 問題已解決，training and validation loss 都有明顯下降。

## 如何運行
- 環境設置: 安裝 Python 3.10.13（或相容版本）。
- 安裝依賴套件：
```
pip install torch pandas numpy nltk gensim matplotlib scikit-learn
```
- 下載 nltk 資源：
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
- 數據準備: 確保 Corona_NLP_train.csv 和 Corona_NLP_test.csv 位於與筆記本相同的目錄下。
- 執行notebook: 打開 BiLSTM_Comment_SentimentAnalysis.ipynb 並按順序執行所有單元格。

### 硬體要求: 
- 建議使用支援 CUDA 的 GPU 以加速訓練。
- 最低配置：8GB RAM，2GB GPU 記憶體。
