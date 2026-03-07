
# from google import genai
# from google.genai import types
# import os
# import io
# from PIL import Image
# import time
# import base64
# import sys # 新增：用於處理系統編碼

# # --- 核心修正：解決 Windows 編碼錯誤 ---
# # 強制讓 Python 的輸出與輸入都使用 UTF-8，避免 ascii 錯誤
# if sys.platform == "win32":
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# # ---------------------------

# # --- 設定區 (API Key 和模型) ---
# API_KEY = "AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k"  
# MODEL_NAME = "models/gemini-3-pro-image-preview" 
# OUTPUT_DIR = "senior_cards"

# client = genai.Client(api_key=API_KEY)

# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# def generate_elderly_greeting_card():
#     print("[系統] 長輩圖產生器 啟動...")

#     # 請確保路徑正確，建議使用 r"路徑" 避免斜線錯誤
#     input_img_path = r"C:\Users\Angus\Desktop\dance\get_frame\taichi.png" 

#     if not os.path.exists(input_img_path):
#         print(f"[錯誤] 找不到輸入圖片: {input_img_path}")
#         return

#     print(f"      -> 載入運動素材: {os.path.basename(input_img_path)}")

#     try:
#         mime_type = "image/png" if input_img_path.lower().endswith(".png") else "image/jpeg"
#         with open(input_img_path, "rb") as f:
#             img_bytes = f.read()
#     except Exception as e:
#         print(f"[錯誤] 讀取檔案失敗: {e}")
#         return

#     print("------------------------------------------------")
#     print(f"[系統] 正在呼叫 AI 轉化為「長輩圖」風格...")
    
#     # AI 設計的 Prompt (保持不變，但現在 Python 處理中文會變穩定)
#     image_prompt = (
#         "Transform the provided image of a person exercising into a classic 'Senior's Greeting Card' (Taiwanese '長輩圖' style). "
#         "1. VISUAL STYLE: Use extremely vibrant colors. Add elements like blooming lotuses, bright sunbeams, or colorful nature scenery into the background. "
#         "2. SUBJECT: Keep the person exercising as the focus, but integrate them into this colorful greeting card theme. "
#         "3. TEXT CONTENT: Add an inspirational text in Traditional Chinese such as '運動身體好，平安沒煩惱' or '活出自信，動出健康'. "
#         "4. TYPOGRAPHY: The text must be in a LARGE, BOLD font with bright colors (like yellow or magenta) and thick outlines. "
#         "5. LAYOUT (CRITICAL): Place the text strictly in the empty background area. DO NOT cover the person's face or body."
#     )

#     try:
#         response = client.models.generate_content(
#             model=MODEL_NAME,
#             contents=[
#                 types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
#                 image_prompt
#             ],
#             config=types.GenerateContentConfig(
#                 response_modalities=["IMAGE"]
#             )
#         )
        
#         if response.candidates and response.candidates[0].content.parts:
#             for part in response.candidates[0].content.parts:
#                 if part.inline_data:
#                     raw_data = part.inline_data.data
                    
#                     if raw_data.startswith(b'\x89PNG') or raw_data.startswith(b'\xff\xd8'):
#                         final_data = raw_data
#                     else:
#                         final_data = base64.b64decode(raw_data)

#                     if final_data:
#                         timestamp = int(time.time())
#                         save_path = f"{OUTPUT_DIR}/senior_card_{timestamp}.png"
                        
#                         with open(save_path, "wb") as f:
#                             f.write(final_data)
                            
#                         print(f"[成功] 長輩圖生成完畢！檔案位置: {save_path}")
                        
#                         try:
#                             img = Image.open(io.BytesIO(final_data))
#                             img.show()
#                         except:
#                             pass
#                     return

#         print("[失敗] AI 回應中沒有包含圖片資料。")

#     except Exception as e:
#         # 這裡會捕捉錯誤，並因為上面的編碼修正，讓你能看懂報錯原因
#         print(f"[錯誤] API 呼叫失敗: {str(e)}")

# if __name__ == "__main__":
#     generate_elderly_greeting_card()
import google.generativeai as genai

# 設定你的 API Key
GOOGLE_API_KEY = "AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k"
genai.configure(api_key=GOOGLE_API_KEY)

print("你的 API Key 可調用的生成模型列表：\n")
print("-" * 40)

# 取得並過濾出支援 'generateContent' (文字/多模態生成) 的模型
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"模型名稱: {m.name}")
        print(f"模型描述: {m.description}")
        print(f"輸入 Token 上限: {m.input_token_limit}")
        print(f"輸出 Token 上限: {m.output_token_limit}")
        print("-" * 40)