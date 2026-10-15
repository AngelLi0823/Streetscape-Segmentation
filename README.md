# Homework 1 — Semantic Segmentation

## 一、專案介紹

本作業使用 PyTorch 實作語意分割（Semantic Segmentation）模型，目標是將輸入影像中的每個像素分類至對應的語意類別。

模型採用 **DeepLabV3 搭配 ResNet-50 Backbone**，共進行三個訓練階段：

1. 初步訓練（Baseline Training）
2. 第一次微調（Fine-tuning）
3. 第二次類別加權微調（Class-weighted Fine-tuning）

最終提交的模型為第二次類別加權微調後儲存的最佳模型。

## 二、使用環境與套件

本專案使用 Google Colab 作為開發與訓練環境，並使用 GPU 加速模型訓練。

主要使用的 Python 套件：

- PyTorch
- Torchvision
- NumPy
- Pillow（PIL）
- Python 標準函式庫：pathlib、random、csv、os、time、shutil、zipfile 等

訓練過程使用 CUDA GPU 與自動混合精度（Automatic Mixed Precision），以降低 GPU 記憶體使用量並提升訓練效率。

## 三、資料集與資料前處理

### 3.1 資料集

本作業使用 `DL_HW1_Public` 資料集。

資料數量如下：

| 資料類型 | 數量 |
|---|---:|
| Training Images | 5,000 張 |
| Ground Truth Masks | 5,000 張 |
| Test Images | 1,000 張 |

程式會從 Google Drive 搜尋資料集 ZIP 檔案，並將資料解壓縮至 Google Colab 的 `/content/` 目錄。

主要資料夾路徑：

```text
/content/DL_HW1_Public/
├── train/
│   ├── image/
│   └── gt/
└── test/
    └── image
```
### 3.2 資料檢查

訓練前會進行以下檢查：

- 確認影像與 Ground Truth Mask 的數量正確。
- 根據檔名配對影像與對應的 Mask。
- 檢查影像與 Mask 的尺寸是否一致。
- 確認 Mask 為單通道標籤影像。
- 檢查 Mask 中的類別 ID 是否連續且符合類別範圍。
- 統計各類別的像素數量與像素比例（Pixel Frequency）。

### 3.3 Training / Validation 切分

使用固定隨機種子 `SEED = 42`，將 5,000 張訓練資料切分為：

| 資料用途 | 數量 |
|---|---:|
| Training | 4,500 張 |
| Validation | 500 張 |

### 3.4 影像與 Mask 前處理

影像前處理包含：

- 將輸入影像轉換為 RGB 格式。
- 使用 `ToTensor()` 將影像轉換為 Tensor。
- 使用 ImageNet 的 Mean 與 Standard Deviation 進行正規化。

正規化參數如下：

```python
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
```

Ground Truth Mask 保留原始整數類別 ID，並轉換為 `torch.long`，不進行影像正規化。

微調階段另外使用隨機水平翻轉（Random Horizontal Flip）作為資料增強方法，並同步翻轉影像與 Mask。

## 四、模型架構

本專案使用 **DeepLabV3-ResNet50** 進行語意分割。

模型設定如下：

| 項目 | 設定 |
|---|---|
| 模型架構 | DeepLabV3 |
| Backbone | ResNet-50 |
| 類別數量 | 24 |
| 深度學習框架 | PyTorch |
| Loss Function | Cross Entropy Loss |
| Optimizer | AdamW |

初步訓練使用 ImageNet 預訓練的 ResNet-50 Backbone 權重，並建立對應 24 個語意類別的分割模型。

後續兩次微調則載入前一階段儲存的最佳模型權重，繼續進行訓練。

## 五、訓練流程

### 5.1 第一階段：Baseline Training

使用 DeepLabV3-ResNet50 建立初始語意分割模型。

主要訓練設定：

| 參數 | 數值 |
|---|---|
| Batch Size | 16 |
| Maximum Epochs | 30 |
| Early Stopping Patience | 8 |
| Backbone Learning Rate | 0.0001 |
| Classifier Learning Rate | 0.0005 |
| Weight Decay | 0.0001 |
| Random Seed | 42 |

使用 AdamW Optimizer 與 Cross Entropy Loss 進行訓練，並透過 Validation Dataset 評估模型表現。

訓練過程使用 Automatic Mixed Precision，並依據驗證結果儲存最佳模型。


### 5.2 第二階段：第一次 Fine-tuning

載入 Baseline Training 所儲存的最佳模型，使用較低的 Learning Rate 繼續微調。

此階段加入隨機水平翻轉作為資料增強，並使用 `ReduceLROnPlateau` 調整學習率。

主要設定：

| 參數 | 數值 |
|---|---|
| Batch Size | 16 |
| Maximum Epochs | 8 |
| Early Stopping Patience | 4 |
| Backbone Learning Rate | 0.00002 |
| Classifier Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Image Size | 640 × 384 |

此階段的模型與訓練紀錄儲存在獨立資料夾中，不覆寫 Baseline 模型。

### 5.3 第三階段：Class-weighted Fine-tuning

載入第一次 Fine-tuning 的最佳模型，並使用類別加權的 Cross Entropy Loss 進行第二次微調。

此階段根據 Training Dataset 中各類別的像素頻率計算類別權重，以調整不同類別對 Loss Function 的影響。


主要設定：

| 參數 | 數值 |
|---|---|
| Batch Size | 16 |
| Maximum Epochs | 8 |
| Early Stopping Patience | 4 |
| Backbone Learning Rate | 0.00001 |
| Classifier Learning Rate | 0.00005 |
| Weight Decay | 0.0001 |
| Training Loss | Weighted Cross Entropy Loss |
| Validation Loss | Cross Entropy Loss |

此階段沿用原始的 Training / Validation 固定切分，並使用隨機水平翻轉進行資料增強。

此模型為本次作業最終選用的模型。

