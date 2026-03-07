
from google import genai
from google.genai import types
import os
import io
from PIL import Image
import time
import base64
import sys # 新增：用於處理系統編碼

# --- 核心修正：解決 Windows 編碼錯誤 ---
# 強制讓 Python 的輸出與輸入都使用 UTF-8，避免 ascii 錯誤
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ---------------------------

# --- 設定區 (API Key 和模型) ---
API_KEY = "AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k"  
MODEL_NAME = "models/gemini-3.1-flash-image-preview" 
OUTPUT_DIR = "senior_cards"

client = genai.Client(api_key=API_KEY)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_elderly_greeting_card_base64(user_image_base64):
    """
    接收 Base64 圖片，回傳 Gemini 生成的長輩圖 Base64
    """
    print("[系統] 長輩圖產生器 (記憶體模式) 啟動...")
    try:
        # 將 Base64 轉回二進位 Bytes，給 Gemini 吃
        img_bytes = base64.b64decode(user_image_base64)
    except Exception as e:
        print(f"[錯誤] Base64 解碼失敗: {e}")
        return None

    image_prompt = (
        "Transform the provided image of a person exercising into a classic 'Senior's Greeting Card' (Taiwanese '長輩圖' style). "
        "1. VISUAL STYLE: Use extremely vibrant colors. Add elements like blooming lotuses, bright sunbeams, or colorful nature scenery into the background. "
        "2. SUBJECT: Keep the person exercising as the focus, but integrate them into this colorful greeting card theme. "
        "3. TEXT CONTENT: Add an inspirational text in Traditional Chinese such as '運動身體好，平安沒煩惱' or '活出自信，動出健康'. "
        "4. TYPOGRAPHY: The text must be in a LARGE, BOLD font with bright colors (like yellow or magenta) and thick outlines. "
        "5. LAYOUT (CRITICAL): Place the text strictly in the empty background area. DO NOT cover the person's face or body."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                image_prompt
            ],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"])
        )
        
        if response.candidates and response.candidates[0].content.parts:
            raw_data = response.candidates[0].content.parts[0].inline_data.data
            
            # 將 Gemini 吐出來的二進位圖片，轉成 Base64 準備回傳給網頁
            if raw_data.startswith(b'\x89PNG') or raw_data.startswith(b'\xff\xd8'):
                final_base64 = base64.b64encode(raw_data).decode('utf-8')
            else:
                final_base64 = raw_data.decode('utf-8') if isinstance(raw_data, bytes) else raw_data
                
            print("[成功] 專屬長輩圖生成完畢！")
            return final_base64

    except Exception as e:
        print(f"[錯誤] API 呼叫失敗: {str(e)}")
        return None
