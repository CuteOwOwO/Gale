from google import genai
from google.genai import types
import os

# --- 設定區 ---
# 建議使用 Gemini 1.5 Flash，因為速度最快，適合即時講評
# 如果想要更精準的分析，可以改用 "gemini-1.5-pro"
API_KEY = "AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k"  
MODEL_NAME = "gemini-2.5-flash" 

# 圖片路徑 (請修改成你的實際路徑)
TEACHER_IMG_PATH = r"C:\Users\Angus\Desktop\dance\captured_poses\auto_223952_274593_skel.png"  # 正確的標準動作 (老師/孫子)
USER_IMG_PATH = r"C:\Users\Angus\Desktop\dance\captured_poses\auto_221933_286498_skel.png"  # 使用者目前的動作 (阿嬤)
# -------------

client = genai.Client(api_key=API_KEY)

def get_ai_instruction():
    print(f"[系統] 正在讀取圖片...")
    
    # 檢查檔案是否存在
    if not os.path.exists(TEACHER_IMG_PATH) or not os.path.exists(USER_IMG_PATH):
        print("[錯誤] 找不到圖片檔案，請檢查路徑。")
        return

    # 讀取圖片 Bytes
    with open(TEACHER_IMG_PATH, "rb") as f: teacher_bytes = f.read()
    with open(USER_IMG_PATH, "rb") as f: user_bytes = f.read()

    print(f"[系統] 呼叫 Gemini ({MODEL_NAME}) 進行動作分析...")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                # 圖片 1: 標準動作 (Target)
                types.Part.from_bytes(data=teacher_bytes, mime_type="image/png"),
                
                # 圖片 2: 使用者動作 (Current)
                types.Part.from_bytes(data=user_bytes, mime_type="image/png"),
                
                # --- 核心 Prompt (賦予角色與任務) ---
                "You are a caring grandson helping his grandmother with rehabilitation exercises.",
                "IMAGE 1 is the CORRECT POSE (The Goal).",
                "IMAGE 2 is the CURRENT POSE (Grandma).",
                
                "TASK: Compare the two skeletons visually. Identify the most significant mistake in Image 2 (e.g., arm too low, leaning wrong side).",
                
                "OUTPUT GUIDELINES:",
                "1. Output ONLY the spoken instruction.",
                "2. Language: Traditional Chinese (Taiwanese Mandarin style).",
                "3. Tone: Warm, encouraging, patient, and filial.",
                "4. Length: Keep it under 20 words (short and clear for TTS).",
                "5. Example: '阿嬤，右手再舉高一點點，對，像我這樣！'",
                "6. If the pose is correct, praise her."
            ],
            config=types.GenerateContentConfig(
                temperature=0.7, # 讓語氣稍微活潑一點
                max_output_tokens=100
            )
        )
        
        # 取得文字結果
        instruction_text = response.text.strip()
        
        print("-" * 30)
        print(f"【AI 孫子說】: {instruction_text}")
        print("-" * 30)
        
        return instruction_text

    except Exception as e:
        print(f"[錯誤] API 呼叫失敗: {e}")
        return None

if __name__ == "__main__":
    # 執行分析
    instruction = get_ai_instruction()
    
    # (選配) 如果你有接 TTS，這裡就可以直接把 instruction 丟進去唸出來
    # play_tts(instruction)