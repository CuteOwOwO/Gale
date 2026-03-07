# main.py
import cv2
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# === 模組化引入 ===
from core.pose import PoseEngine   
from core.hand import HandEngine  
from core.lobby_choose import LobbyLogic 
from core.game_engine import GameEngine  
from core.myqrcode import generate_qr_code_base64
from core.imgbb_upload import upload_base64_to_imgbb
from core.ai import generate_elderly_greeting_card_base64
from core.coach_ai import get_coach_instruction

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
WIDTH, HEIGHT = 1280, 720 

@app.get("/index.html")
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/settings.html")
async def get_settings():
    with open("settings.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
    
@app.get("/game.html")
async def get_game():
    with open("game.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
    
@app.get("/main_play.html")
async def get_main_play():
    with open("main_play.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_lobby(websocket: WebSocket):
    await websocket.accept()
    print("大廳前端已連線！")
    
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    # 初始化大廳模組
    hand_engine = HandEngine()       
    current_logic = LobbyLogic()      

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 視覺處理 (用 hand_engine)
            frame_rgb, results = hand_engine.process_frame(frame)
            message = {}
            
            if results.multi_hand_landmarks:
                message = current_logic.process(results.multi_hand_landmarks)
            
            if message:
                await websocket.send_text(json.dumps(message))
            
            await asyncio.sleep(0.03)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"大廳錯誤: {e}")
    finally:
        cap.release() # 確保切換頁面時釋放相機



# 遊戲專用 (AR 疊圖與評分)
# ==========================================
@app.websocket("/ws/game/{video_id}/{frame_id}")
async def websocket_game(websocket: WebSocket, video_id: str, frame_id: int):
    await websocket.accept()
    print(f"🎮 遊戲前端已連線！影片: {video_id}, 影格: {frame_id}")
    
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    game_engine = GameEngine(video_id=video_id, frame_id=frame_id)

    # 用來標記前端是否發送了拍照指令 
    capture_requested = False
    analyze_requested = False

    # 背景監聽器，專門聽前端傳來的話 
    async def listen_to_client():
        nonlocal capture_requested, analyze_requested
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("action") == "capture":
                    print("後端收到拍照指令！準備暫停遊戲並呼叫 Gemini...", flush=True)
                    capture_requested = True
                
                # 🌟 新增這段：如果聽到呼叫教練的指令
                elif message.get("action") == "analyze_pose":
                    print("後端收到教練分析指令！準備擷取疑問姿勢...", flush=True)
                    analyze_requested = True 
        except Exception:
            pass

    # 啟動這隻背景耳朵
    listen_task = asyncio.create_task(listen_to_client())
    try:
        while True:
            if capture_requested:
                print("⏳ 正在擷取最完美的姿勢照片...", flush=True) # 加上 flush=True 強制立刻印出
                user_img_base64 = game_engine.get_overlay_frame_base64()
                
                if user_img_base64:
                    print("🤖 1. 交給 Gemini 生成專屬長輩圖... (約需 5~10 秒)", flush=True)
                    
                   
                    gemini_base64 = await asyncio.to_thread(
                        generate_elderly_greeting_card_base64, 
                        user_img_base64
                    )
                    
                    if gemini_base64:
                        print("☁️ 2. 正在上傳至 ImgBB 取得公開網址...", flush=True)
                        
                        
                        img_url = await asyncio.to_thread(
                            upload_base64_to_imgbb, 
                            gemini_base64
                        )
                        
                        qr_code_base64 = ""
                        if img_url:
                            print(f"🔗 3. 產生 QR Code... (網址: {img_url})", flush=True)
                            # QR Code 是本地運算且極快，直接跑就好
                            qr_code_base64 = generate_qr_code_base64(img_url)
                        
                        print("✅ 4. 大功告成！傳送最終結果給前端顯示！", flush=True)
                        await websocket.send_json({
                            "status": "success",
                            "final_image": gemini_base64,
                            "qr_code": qr_code_base64
                        })
                    else:
                        print("❌ Gemini 生成失敗", flush=True)
                
                break

            if analyze_requested:
                print("正在擷取疑問姿勢照片...", flush=True)
                user_img_base64 = game_engine.get_overlay_frame_base64()

                if user_img_base64:
                    print("正在請 Gemini 教練分析動作差異... (約需幾秒鐘)", flush=True)

                  
                    ai_text = await asyncio.to_thread(
                        get_coach_instruction, 
                        user_img_base64, 
                        video_id, 
                        frame_id
                    )

                    # 將 Gemini 產生出來的真實文字傳給前端
                    await websocket.send_json({
                        "status": "coach_success",
                        "coach_text": ai_text
                    })

                # 重置標記，讓迴圈繼續跑，確保遊戲不斷線
                analyze_requested = False
                continue

            # === 正常遊戲流程 (原本的程式碼) ===
            ret, frame = cap.read()
            if not ret: break

            current_score, base64_image, is_calling_coach = game_engine.process_frame(frame)
            
            await websocket.send_json({
                "score": current_score,
                "image": base64_image,
                "call_coach": is_calling_coach
            })
            
            await asyncio.sleep(0.03)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"遊戲連線錯誤: {e}")
    finally:
        listen_task.cancel() # 遊戲結束，關閉耳朵
        cap.release()

if __name__ == "__main__":
   
    import uvicorn
   
    uvicorn.run(app, host="0.0.0.0", port=8000)