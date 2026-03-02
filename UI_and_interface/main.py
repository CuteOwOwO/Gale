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

    # 初始化遊戲引擎，傳入精準參數
    game_engine = GameEngine(video_id=video_id, frame_id=frame_id)

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 呼叫 GameEngine 處理畫面 (這個函式會回傳分數與 Base64 圖片)
            current_score, base64_image = game_engine.process_frame(frame)
            
            # 將資料打包成 JSON 傳給前端
            await websocket.send_json({
                "score": current_score,
                "image": base64_image
            })
            
            await asyncio.sleep(0.03)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"遊戲連線錯誤: {e}")
    finally:
        cap.release() # 確保遊戲結束或跳離時釋放相機

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)