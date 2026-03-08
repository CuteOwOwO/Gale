# Just Pose

![Just Pose Banner](logos/UI_and_interface/logo.png.png)

> **「讓長輩的每一次伸展，都充滿趣味與溫暖的陪伴。」**

**Just Pose** 是一個專為長輩與居家復健者打造的「AI 體感互動系統」。透過電腦視訊鏡頭，玩家無需穿戴任何設備，即可進行手勢操作與體感運動。我們結合了 MediaPipe 骨架追蹤、AR 即時影像處理，並深度整合 Google Gemini AI 及 TTS系統 。打造出能即時給予語音指導的 AI 專屬教練，以及通關後專屬的長輩圖獎勵機制。

---


---

## 🛠️ 技術架構 (Tech Stack)

- **前端 (Frontend)**: HTML5, CSS3, JavaScript (Vanilla), WebSockets, Web Speech API (TTS).
- **後端 (Backend)**: Python 3.9, FastAPI, Uvicorn.
- **電腦視覺 (Computer Vision)**: OpenCV, MediaPipe (Pose & Selfie Segmentation).
- **人工智慧 (AI)**: Google GenAI API (`gemini-2.5-flash`).
- **第三方服務 (Cloud APIs)**: ImgBB API (圖像託管).

---

## 🚀 快速啟動 (Getting Started)

### 1. 環境需求
本專案建議在 Python 3.9 環境下運行（已在 conda 環境 `face` 測試通過）。
請先安裝必要的依賴套件，完整清單請參考 `condaenv.txt`：

```powershell
# 建立並啟動虛擬環境 (建議)
conda create -n face python=3.9
conda activate face

# 安裝核心套件
pip install -r ..\requirements.txt
pip install fastapi uvicorn requests python-dotenv opencv-python mediapipe google-genai numpy pillow scipy
```
2. 環境變數設定 (API Keys)
專案的核心 AI 與上傳功能依賴外部 API。我們已改為從環境變數讀取，不再寫死在程式碼中以確保安全。

請在專案根目錄（UI_and_interface/）下建立一個 .env 檔案，並填入以下資訊：

程式碼片段
GEMINI_API_KEY=your_gemini_key_here
IMGBB_API_KEY=your_imgbb_key_here
(若無設定，程式會在啟動時拋出錯誤提醒。)

3. 啟動伺服器
進入資料夾後，執行以下指令啟動 FastAPI 伺服器：

PowerShell
python main.py
伺服器啟動後，請打開瀏覽器（建議使用 Google Chrome 以確保視訊與語音功能正常），輸入以下網址進入系統：
👉 http://localhost:8000

💡 備註 (Notes for Judges / Testing)
為了方便競賽評審與開發測試，我們提供一組測試用的 API Key。請勿將此 Key 用於商業用途，僅供測試評估使用：

GEMINI_API_KEY=AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k

IMGBB_API_KEY=417601fefc1f8b6cbb1caede5c5f9bc5

開發團隊: Just Pose Team
聯絡方式: [請填寫聯絡方式或GitHub連結]
