<div align="center">
  <img src="logo.png" alt="Just Pose Banner" width="450">
</div>

<div align="center">
  <br>
  <strong>「讓長輩的每一次伸展，都充滿趣味與溫暖的陪伴。」</strong>
  <br><br>
</div>

**Just Pose** 是一個專為長輩與居家復健者打造的「AI 體感互動系統」。透過電腦視訊鏡頭，玩家無需穿戴任何設備，即可進行手勢操作與體感運動。我們結合了 MediaPipe 骨架追蹤、AR 即時影像處理，並深度整合 Google Gemini AI 及 TTS 語音系統，打造出能即時給予語音指導的 AI 專屬教練，以及通關後專屬的「長輩圖」獎勵機制。

---

### 設計理念
我們的設計深度融入了「自我決定論」(Self-Determination Theory, SDT) 的三大核心要素：**自主性 (Autonomy)**、**勝任感 (Competence)** 與**關聯性 (Relatedness)**。我們期盼讓居家運動不再是一項枯燥的功課，而是讓使用者能在輕鬆、無壓力的過程中，主動且投入地享受每一次的伸展。

---

### 專案目錄結構 (Project Structure)

本儲存庫包含專案的完整開發歷程與核心程式碼，主要目錄說明如下：

- `UI_and_interface/`：**核心主程式資料夾**（包含前端介面與後端伺服器，啟動專案請進入此目錄）。
- `some_algorithms/` & `evaluation/`：演算法開發區（記錄我們選定與評估「關鍵動作演算法」的測試過程）。
- `WIPS/`：初期開發的工作進度（Work In Progress），包含未整合的實驗性程式碼，可一窺團隊的開發脈絡。
- `F5-TTS/` (或其他 TTS 資料夾)：語音合成 (TTS) 相關的依賴與測試模組。

---

## 核心功能 (Core Features)

*(建議在這裡列出 3-5 點你們最自豪的功能，例如：)*
-  **困難動作拆解 (C)**：
   我們利用自製的演算法，將一部困難的影片進行拆解，讓使用者分段分步地學習。
   
-  **長輩友善復健系統**：
   利用辨認手勢的模型，抓取使用者的手部特徵，以此取代滑鼠鍵盤。在選擇關卡、商品等物時，讓使用者可以簡易無負擔的方式進行操作。
  
-  **實時動作評分 (C)**：
   整合 AR 疊圖技術，讓使用者能即時比對自身與標準動作的差異並精準修正；同時結合 MediaPipe 骨架偵測，針對動作準確度進行自動化評分。
   
-  **乖孫教你做運動(R)**:
   由 Gemini 提供動作糾錯，並整合 TTS 語音合成技術，將建議轉化為親友（如孫子）的親切語音，提升使用者的運動動機與心理陪伴感。
   
-  **長輩圖分享功能(R)**:
   可拍攝使用者完成動作的畫面，並用Gemini生成專屬早安長輩圖。加入互動，讓運動可以呼朋引伴，提高動力 。
  
-  **獎勵商店(A)**:
   根據動作評分，使用者可獲得星星作為獎勵。本系統具備「商店」，使用者可用星星購買可愛配飾，讓長輩們的心態更年輕。

---

### 技術架構 (Tech Stack)

- **前端 (Frontend)**: HTML5, CSS3, JavaScript , WebSockets
- **後端 (Backend)**: Python 3.9, FastAPI, Uvicorn.
- **電腦視覺 (Computer Vision)**: OpenCV, MediaPipe
- **人工智慧 (AI)**: Google GenAI API.
- **第三方服務 **: ImgBB (圖像託管).

---

## 快速啟動 (Getting Started)

### 1. 環境需求
本專案建議在 **Python 3.9** 環境下運行。
請先安裝必要的依賴套件（完整清單請參考 `condaenv.txt` 或執行以下指令）：

```powershell
# 建立並啟動虛擬環境 (建議)
conda create -n face python=3.9
conda activate face

# 安裝核心套件
pip install -r ..\requirements.txt
pip install fastapi uvicorn requests python-dotenv opencv-python mediapipe google-genai numpy pillow scipy
```

### 2. 環境變數設定 (API Keys)
為了確保資訊安全，專案的核心 AI 與上傳功能已改為從環境變數讀取。請在專案根目錄（`UI_and_interface/`）下建立一個 `.env` 檔案，並填入以下資訊：

```env
GEMINI_API_KEY=your_gemini_key_here
IMGBB_API_KEY=your_imgbb_key_here
```
*(註：我們已在提交的檔案中預先設定好，若無設定，程式啟動時會拋出錯誤提醒。)*

### 3. 啟動伺服器
進入主程式資料夾後，執行以下指令啟動 FastAPI 伺服器：

```powershell
python main.py
```
伺服器啟動後，請打開瀏覽器（**強烈建議使用 Google Chrome** 以確保視訊與語音功能正常運作），輸入以下網址即可進入系統：
 **http://localhost:8000**

---

**備註 (Notes for Judges / Testing)**
為了方便競賽評審與開發測試，我們提供一組測試用的 API Key。請勿將此 Key 用於商業用途，僅供測試評估使用：

```env
GEMINI_API_KEY=AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k
IMGBB_API_KEY=417601fefc1f8b6cbb1caede5c5f9bc5
```

**開發團隊**: Just Pose Team
**聯絡方式**: anguszheng11@gmail.com
