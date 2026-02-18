#本來的想要以徒生徒


from google import genai
from google.genai import types
import os
import io
from PIL import Image
import time

# --- 設定區 (API Key 和模型) ---
API_KEY = "AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k"  
MODEL_NAME = "gemini-2.5-flash-image-preview" 
OUTPUT_DIR = "generated_results"
# ---------------------------

client = genai.Client(api_key=API_KEY)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_with_nano_banana_hardcoded():
    print("[系統] 程式啟動...")

    # ==========================================
    #  [👉 在這裡填寫路徑]
    #  請分別填入你的「使用者照片」和「骨架圖」的完整路徑
    #  (Windows 注意路徑斜線方向，用 '/' 或 '\\' 都可以)
    # ==========================================
    user_img_path = r"hamburger.png" # <--- 改這裡
    pose_img_path = r"C:\Users\Angus\Desktop\dance\captured_poses\origin.jpg" # <--- 改這裡
    # ==========================================


    # --- 檢查路徑是否存在 ---
    if not os.path.exists(user_img_path):
        print(f"[錯誤] 找不到使用者照片: {user_img_path}\n請檢查路徑是否正確。")
        return
    if not os.path.exists(pose_img_path):
        print(f"[錯誤] 找不到骨架圖片: {pose_img_path}\n請檢查路徑是否正確。")
        return

    print(f"      -> 載入長相素材: {os.path.basename(user_img_path)}")
    print(f"      -> 載入骨架素材: {os.path.basename(pose_img_path)}")

    # --- 讀取檔案轉成 Bytes ---
    try:
        with open(user_img_path, "rb") as f:
            user_bytes = f.read()
        with open(pose_img_path, "rb") as f:
            pose_bytes = f.read()
    except Exception as e:
        print(f"[錯誤] 讀取檔案失敗: {e}")
        return

    print("------------------------------------------------")
    print(f"[系統] 正在呼叫 {MODEL_NAME} 進行融合繪圖...")
    print("      (請耐心等候，約需 5-15 秒...)")
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                # 素材 1: 使用者長相 (JPEG)
                types.Part.from_bytes(data=user_bytes, mime_type="image/jpeg"),
                # 素材 2: 骨架動作 (PNG)
                types.Part.from_bytes(data=pose_bytes, mime_type="image/png"),
                
                # --- Prompt ---
                "幫我把圖1中的男性 畫成跟圖2的女性一樣的姿勢"
                "姿勢和圖二相同"
                # "Generate a high-quality illustration based on these two images.",
                # "1. SOURCE: Use the person's appearance from the first image.",
                # "2. POSE: Strictly follow the skeleton pose in the second image.",
                # "3. STYLE: A photorealistic photograph. Cinematic lighting, highly detailed, realistic skin texture.",
                # "4. OUTPUT: A full-body photograph of the person in a realistic environment.",
                # "5. front side of the person , with appreance like the first image",
                # "6. Real style . As real as possible"
                # "7. front side , looking forward"
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"]
            )
        )
        
        # --- 處理回傳 ---
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    raw_data = part.inline_data.data
                    
                    # Debug 訊息：讓我們看看 API 到底吐了什麼回來
                    print(f"------------------------------------------------")
                    print(f"[DEBUG] 資料類型: {type(raw_data)}")
                    print(f"[DEBUG] 資料開頭 (前20 bytes): {raw_data[:20]}")
                    
                    final_data = None
                    
                    # 判斷邏輯 1: 如果開頭已經是 PNG 或 JPG 的 Magic Number -> 代表已經是圖片，不需要解碼！
                    if raw_data.startswith(b'\x89PNG') or raw_data.startswith(b'\xff\xd8'):
                        print("[系統] 偵測到原始圖片數據 (Raw Bytes)，直接存檔！")
                        final_data = raw_data
                        
                    # 判斷邏輯 2: 如果還是需要解碼 (通常 SDK 不會這樣，但以防萬一)
                    else:
                        print("[系統] 偵測到可能是 Base64 編碼，正在解碼...")
                        try:
                            # 嘗試解碼
                            final_data = base64.b64decode(raw_data)
                        except Exception as e:
                            print(f"[錯誤] 解碼失敗，資料可能不是圖片: {e}")

                    # --- 存檔與顯示 ---
                    if final_data:
                        timestamp = int(time.time())
                        save_path = f"{OUTPUT_DIR}/fixed_result_{timestamp}.png"
                        
                        with open(save_path, "wb") as f:
                            f.write(final_data)
                            
                        print(f"[成功] 圖片已修正並儲存！")
                        print(f"      -> 檔案位置: {save_path}")
                        
                        # 嘗試彈出視窗顯示
                        try:
                            img = Image.open(io.BytesIO(final_data))
                            img.show()
                        except Exception as e:
                            print(f"[警告] 雖然存檔成功，但 Python 無法預覽圖片 (可能缺少顯示器環境?): {e}")
                    else:
                        print("[失敗] 無法解析圖片資料。")
                    return

        print("[系統] API 回應中沒有包含圖片資料 (inline_data)。")
        print("API 回應文字:", response.text)

    except Exception as e:
        print(f"[錯誤] API 呼叫失敗: {e}")

if __name__ == "__main__":
    generate_with_nano_banana_hardcoded()