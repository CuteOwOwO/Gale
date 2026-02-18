import time

class LobbyLogic:
    def __init__(self):
        self.is_tracking_mode = False
        self.hand_raised_start_time = 0
        self.last_action_time = 0
        # 設定一個可信度門檻 (建議 0.5)
        self.VISIBILITY_THRESHOLD = 0.5 

    def process(self, landmarks):
        """
        輸入: 骨架點 (landmarks)
        輸出: 要傳給前端的 JSON message
        """
        # 預設訊息
        if self.is_tracking_mode:
            message = {"action": "none", "status": "已連線：請將手移至邊緣切換"}
        else:
            message = {"action": "none", "status": "待機中：請舉起任一手啟動"}

        # === 定義關鍵點 (鏡像後的對應) ===
        # MediaPipe 的 Left (19) 在鏡像畫面中對應 "真實右手"
        real_right_hand = landmarks[19] 
        real_right_shoulder = landmarks[11]

        # MediaPipe 的 Right (20) 在鏡像畫面中對應 "真實左手"
        real_left_hand = landmarks[20]  
        real_left_shoulder = landmarks[12]

        # === 關鍵修改：檢查手部是否「可見」 ===
        is_right_visible = real_right_hand.visibility > self.VISIBILITY_THRESHOLD
        is_left_visible = real_left_hand.visibility > self.VISIBILITY_THRESHOLD

        # Debug 用：你可以看到現在 AI 覺得哪隻手是真的存在的
        print(f"左手可見: {is_left_visible} ({real_left_hand.visibility:.2f}) | 右手可見: {is_right_visible} ({real_right_hand.visibility:.2f})")

        # === 1. 智慧游標 (Smart Cursor) ===
        active_hand = None

        # 邏輯：
        # 1. 如果兩隻手都看得到 -> 誰舉得高跟誰
        # 2. 如果只看得到左手 -> 跟左手
        # 3. 如果只看得到右手 -> 跟右手
        # 4. 都沒看到 -> 不更新游標
        
        if is_left_visible and is_right_visible:
            if real_left_hand.y < real_right_hand.y:
                active_hand = real_left_hand
            else:
                active_hand = real_right_hand
        elif is_left_visible:
            active_hand = real_left_hand
        elif is_right_visible:
            active_hand = real_right_hand
        
        # 只有在有「可見的手」時才更新游標座標
        if active_hand:
            cx_percent = max(0, min(1, active_hand.x))
            cy_percent = max(0, min(1, active_hand.y))
            message["hand_x"] = int(cx_percent * 100)
            message["hand_y"] = int(cy_percent * 100)
        else:
            # 如果兩隻手都不見了，可以選擇隱藏游標 (選擇性)
            # message["hand_x"] = -100 
            pass

        # === 2. 啟動機制 (One-time Activation) ===
        if not self.is_tracking_mode:
            # 必須同時滿足：1. 手是可見的 2. 手舉得夠高
            right_trigger = is_right_visible and ((real_right_hand.y < real_right_shoulder.y * 1.3) )
            left_trigger = is_left_visible and (real_left_hand.y < real_left_shoulder.y * 1.3)

            if right_trigger or left_trigger:
                if self.hand_raised_start_time == 0:
                    self.hand_raised_start_time = time.time()
                
                duration = time.time() - self.hand_raised_start_time
                if duration > 1:
                    self.is_tracking_mode = True
                else:
                    message["status"] = f"啟動中... {1 - duration:.1f}"
            else:
                self.hand_raised_start_time = 0 

        # === 3. 邊緣觸發 (Edge Trigger) ===
        current_time = time.time()
        if self.is_tracking_mode and (current_time - self.last_action_time > 1.5):
            LEFT_ZONE = 0.3
            RIGHT_ZONE = 0.7

            # 觸發也要檢查 visibility，避免隱形的手誤觸
            if is_left_visible and (real_left_hand.x < LEFT_ZONE ):
                message["action"] = "left"
                self.last_action_time = current_time
                print("觸發：向左 (左手)")
                
            elif is_right_visible and (real_right_hand.x > RIGHT_ZONE):
                message["action"] = "right"
                self.last_action_time = current_time
                print("觸發：向右 (右手)")

           
                
        message["is_tracking"] = self.is_tracking_mode
        
        return message