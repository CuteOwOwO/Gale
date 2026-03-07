# core/coach_ai.py
import base64
import os
from google import genai
from google.genai import types

# 🌟 請記得填入你實際的 API KEY
API_KEY = "AIzaSyCIoKxxbMM_yewszbrOvTHMvmnadniMv9k"  
MODEL_NAME = "gemini-2.5-flash" 

client = genai.Client(api_key=API_KEY)

def get_coach_instruction(user_img_base64: str, video_id: str, frame_id: int) -> str:
    print("[系統] 正在呼叫 Gemini 教練進行動作比對...", flush=True)
    
    # 組合出老師標準姿勢的圖片路徑 (跟遊戲引擎讀取的是同一張)
    teacher_img_path = f"static/frames/{video_id}_{frame_id}.jpg"
    
    if not os.path.exists(teacher_img_path):
        print(f"[錯誤] 找不到老師的圖片: {teacher_img_path}")
        return "阿嬤，我沒看到標準動作，請稍後再試！"

    try:
        # 讀取老師的圖片轉成 Bytes
        with open(teacher_img_path, "rb") as f: 
            teacher_bytes = f.read()
            
        # 將前端傳來的使用者 Base64 圖片轉回 Bytes
        user_bytes = base64.b64decode(user_img_base64)
        
    except Exception as e:
        print(f"[錯誤] 圖片讀取失敗: {e}")
        return "阿嬤，你的畫面有點不清楚喔！"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=teacher_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=user_bytes, mime_type="image/jpeg"),
                "You are a caring grandson helping his grandmother with rehabilitation exercises.",
                "IMAGE 1 is the CORRECT POSE (The Goal).",
                "IMAGE 2 is the CURRENT POSE (Grandma).",
                
                "TASK: Compare the two skeletons visually. Identify the most significant mistake in Image 2 (e.g., arm too low, leaning wrong side).",
                
                "OUTPUT GUIDELINES:",
                "1. Output ONLY the spoken instruction.",
                "2. Language: Traditional Chinese (Taiwanese Mandarin style).",
                "3. Tone: Warm, encouraging, patient, and filial.",
                "5. Example: '阿嬤，右手再舉高一點點，身體要再向右轉一點點喲！'",
                "6. If the pose is correct, praise her."
            ],
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2000)
        )
        instruction = response.text.strip()
        print(f"【AI 教練分析結果】: {instruction}", flush=True)
        return instruction
        
    except Exception as e:
        print(f"[錯誤] Gemini API 呼叫失敗: {e}")
        return "阿嬤，我現在有點累，晚點再幫你看動作喔！"