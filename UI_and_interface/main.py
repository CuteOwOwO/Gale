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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("前端已連線！")
    
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    # === 初始化模組 ===
    pose_engine = PoseEngine()      
    hand_engine = HandEngine()       
    current_logic = LobbyLogic()      

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. 視覺處理 (改成用 hand_engine)
            frame_rgb, results = hand_engine.process_frame(frame)
            
            # 預設空訊息
            message = {}

            
            if results.multi_hand_landmarks:
                message = current_logic.process(results.multi_hand_landmarks)
            
            # 如果你要 debug，可以在這裡印出來
            # if message.get('action') != 'none':
            #     print(f"傳送訊息: {message}")

            if message:
                await websocket.send_text(json.dumps(message))
            
            await asyncio.sleep(0.03)

    except Exception as e:
        import traceback
        traceback.print_exc() # 這樣可以看到更完整的錯誤訊息
        print(f"錯誤: {e}")
    finally:
        cap.release()

@app.get("/game.html")
async def get_game():
    with open("game.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)