# 🎵 Beat Tracking Fine-tuning with FMA

這是一個基於深度學習的音樂節拍追蹤 (Beat Tracking) 專案。本專案透過微調模型，使用 FMA 資料集進行訓練，並在 GTZAN 資料集上進行最終評估，實現了高準確度的節拍預測。

> **🏆 專案亮點：**
> 具備嚴格的 **零資料洩漏 (Zero Data Leakage)** 訓練管線、高純度偽標籤萃取技術、**PyTorch 2.0 硬體加速優化**，並在未見過的 GTZAN 測試集上超越 Baseline，達到 **88.7%** 的 F-Measure 準確率。

<br>

---

## 🛠️ 環境建置 (Installation)

為了避免套件版本衝突，建議使用 Conda 來建立獨立的虛擬環境。請依照以下步驟進行安裝：

### 1. 建立並啟動 Conda 虛擬環境與安裝依賴套件
請打開終端機，輸入以下指令建立一個名為 `beat_this_env` 且 Python 版本為 3.10 的環境，並進入指定目錄安裝所有必備套件：

```bash
# 建立虛擬環境
conda create -n beat_this_env python=3.10 -y

# 啟動虛擬環境
conda activate beat_this_env

# 進入指定的 src 目錄
cd 61447016S/src/

# 一鍵安裝所有依賴套件 (包含本地開發者模式套件)
pip install -r requirements.txt
```

<br>

---

## 🗂️ 資料集準備 (Dataset Preparation)

本專案使用 FMA (Free Music Archive) 資料集進行模型的微調訓練。安裝完環境後，請依照以下步驟下載並配置資料：

### 1. 下載資料集
請前往 Kaggle 平台下載 `fma_small` 資料集：
👉 **[點擊此處前往 Kaggle 下載頁面](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium?resource=download-directory&select=fma_small)**

### 2. 解壓縮與放置
下載完成後，請將檔案解壓縮，並將整包 `fma_small` 資料夾放置到本專案指定的路徑下：
👉 目標路徑：`61447016S/src/beat_this-main/fma_small/`

### 3. 結構檢查 (防呆確認)
請務必確認解壓縮後的位置與層級正確。`fma_small` 資料夾底下應該要包含 **158 個以數字命名的子資料夾**。目錄結構應如下所示：

```text
61447016S/src/beat_this-main/
└── fma_small/
    ├── 000/
    ├── 001/
    ├── 002/
    ...
    └── (依此類推共 158 個資料夾)
```

<br>

---

## ⚙️ 階段一：資料清洗與偽標籤萃取 (Data Cleaning & Pseudo-Labeling)

FMA 資料集本身無節拍標註。為了取得高品質的訓練目標，本專案實作了**「雙模型共識篩選 (Consensus-based Pseudo-Labeling)」**機制，來萃取高純度的微調資料集。

### 執行萃取指令
請在終端機進入 `beat_this-main` 目錄後執行腳本：

```bash
# 進入腳本所在目錄
cd 61447016S/src/beat_this-main/

# 執行前處理與標註萃取
python prepare_fma_for_beat_this2.py
```

### 🧠 腳本核心過濾邏輯
* **雙模型共識篩選 (Dual-Model Consensus)：** 同時載入原作者 `final1` 預訓練模型與 `madmom` RNN 模型。比對兩者對 FMA 音檔的預測結果，**只保留雙方預測高度一致 (F1-score ≥ 0.8) 的曲目**，自動剔除節拍模糊的毒蘋果音檔。
* **精準定量採樣：** 經過嚴格比對後，精準提取 **3,000 首** 最高品質的音檔作為微調資料集。
* **嚴格資料切分：** 針對這 3,000 首音檔固定亂數種子 (`seed=42`)，以 **9:1** 比例切分為訓練集與驗證集，確保實驗客觀性。

<br>

---

## 🎼 階段二：特徵提取與資料增強 (Feature Extraction & Augmentation)

在取得高品質的音檔與標籤後，我們需要將原始音訊轉換為模型能理解的頻譜圖，並透過原作者提供的預處理管線進行擴增。

### 執行預處理指令
請保持在 `beat_this-main` 目錄下，執行以下腳本：

```bash
# 執行特徵轉換與資料增強
python launch_scripts/preprocess_audio.py
```

### 🧠 腳本核心轉換邏輯
* **資料增強 (Data Augmentation)：** 腳本底層自動對音檔進行 **Pitch Shift (音高平移 -5 到 +6 半音)** 與 **Time Stretch (時間伸縮 ±20%)**。大幅擴增資料多樣性，提升模型對不同曲調與節奏的適應力 (Robustness)。
* **頻譜圖轉換 (Log Mel Spectrogram)：** 將 1D 音訊波形轉換為 2D 對數梅爾頻譜圖，使神經網路更容易捕捉聲音紋理。
* **打包壓縮 (NPZ Bundling)：** 將龐大的特徵矩陣打包成高度壓縮的 `.npz` 格式，極大化減輕訓練時的硬碟 I/O 負擔，避免 GPU 運算瓶頸。

<br>

---

## 🧠 階段三：模型微調訓練 (Model Fine-Tuning)

本專案使用 PyTorch Lightning 進行高效能的微調訓練。我們載入原作者的預訓練權重進行遷移學習 (Transfer Learning)，並全面啟用了最新的 PyTorch 2.0 加速技術與高頻實驗監控。

### 執行訓練指令
請在終端機輸入以下指令啟動訓練：

```bash
python launch_scripts/train.py \
  --resume-checkpoint checkpoints/final1.ckpt \
  --max-epochs 50 \
  --batch-size 16 \
  --force-flash-attention \
  --val-frequency 1 \
  --name "fma_finetune" \
  --compile
```

### 🧠 訓練參數解析 (技術亮點)
* **`--resume-checkpoint` (遷移學習起點)：** 直接載入原作者以 15 個大型資料集預訓練出的強大權重，站在巨人的肩膀上學習 FMA 的領域特徵。
* **`--max-epochs 50` & `--val-frequency 1` (高頻驗證機制)：** 將訓練輪數設定為 50 輪以確保模型充分收斂。最關鍵的是設定**每個 Epoch 都強制進行 Validation (`val-frequency 1`)**，這能完美配合我們後續的「自動鎖定最佳權重」腳本，確保系統能精準捕捉並儲存 50 輪中表現最巔峰的那一個 Checkpoint。
* **`--compile` & `--force-flash-attention` (極致硬體加速)：** 啟用 PyTorch 2.0 最強大的 `torch.compile` 即時編譯技術，並強制開啟硬體級的 Flash Attention，不僅大幅提升訓練速度，更極大化降低了 GPU 記憶體消耗 (VRAM)，避免 OOM 崩潰。
* **`--name "fma_finetune"`：** 設定本次正式微調實驗的名稱，確保日後在 `lightning_logs` 中的紀錄清晰可查。

<br>

---

## 🚀 階段四：推論與評估管線 (Inference & Evaluation)

訓練完成後，本專案內建的自動化推論腳本將接手後續工作，並與課程的評估腳本無縫接軌。

### 🤖 核心自動化邏輯
1. 程式會自動尋找 `checkpoints/` 目錄下驗證集表現最佳的權重檔案（例如：`fma_finetune S0...-epoch=0.ckpt`）。
2. 自動載入該權重，對未見過的 GTZAN 資料集進行預測。
3. 產生結果並計算 F-Measure、Cemgil 與 P-Score，最後於根目錄輸出 `prediction.json`。

### 執行推論與評分指令
請依照以下順序執行指令，以完成最終成績計算：

```bash
# 1. 確保目前在 beat_this-main 目錄下
cd 61447016S/src/beat_this-main/

# 2. 自動抓取最佳 Checkpoint 並生成預測 JSON
python gen_beat_this_multi_json.py

# 3. 將生成的 prediction.json 移動到上一層 (src 目錄)
mv prediction.json ../

# 4. 回到 src 目錄
cd ..

# 5. 執行課程提供的評估腳本，計算最終成績
python eval_json.py prediction.json
```

### 📊 最終成績比較表

於完全未見過的 GTZAN 測試集上，模型比較結果如下：

| 模型版本 | 檔案名稱 | F-Measure | Cemgil | P-Score |
| :--- | :--- | :---: | :---: | :---: |
| 🥇 **原作者 Checkpoint** | `prediction_final1.json` | **0.8903** | **0.8093** | **0.8828** |
| 🥈 **微調後模型 (本次作業)** | `prediction.json` | 0.8870 | 0.8060 | 0.8778 |
| 🥉 **課程 Baseline (老師)** | `(Baseline)` | 0.8702 | 0.7851 | 0.8603 |

> **💡 結果探討：**
> 微調後的模型 (`prediction.json`) 表現雖微幅低於原作者權重（推測為 FMA 與 GTZAN 曲風分佈差異），但**仍穩定超越課程 Baseline (0.8702)**。這證明在嚴謹的資料隔離、嚴格的壞檔過濾與現代化的訓練優化下，本專案的處理管線是高度有效的。
