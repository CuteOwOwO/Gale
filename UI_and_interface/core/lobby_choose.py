import time
import math

class LobbyLogic:
    def __init__(self):
        self.last_action_time = 0
        self.COOLDOWN = 1.0  # 動作觸發冷卻時間 (秒)
        
        # 閾值設定
        self.POINTING_THRESHOLD = 0.1 # 用來判斷指向是否足夠明顯 (X軸差距)
        
    def process(self, multi_hand_landmarks):
        """
        輸入: results.multi_hand_landmarks (MediaPipe Hands 的偵測結果列表)
        輸出: 要傳給前端的 JSON message
        """
        current_time = time.time()
        message = {
            "action": "none", 
            "status": "待機中：請伸出食指指向左/右",
            "hand_x": -100,
            "hand_y": -100
        }

        # 如果沒有偵測到任何手，直接回傳預設值
        if not multi_hand_landmarks:
            return message

        # 遍歷偵測到的每一隻手 
        for landmarks in multi_hand_landmarks:
            # MediaPipe Hands 的關鍵點索引：
            # 0: 手腕 (Wrist)
            # 5: 食指根部 (Index MCP)
            # 8: 食指指尖 (Index Tip)
            # 12: 中指指尖, 16: 無名指指尖, 20: 小指指尖
            
            wrist = landmarks.landmark[0]
            index_mcp = landmarks.landmark[5]
            index_tip = landmarks.landmark[8]
            middle_tip = landmarks.landmark[12]
            ring_tip = landmarks.landmark[16]
            pinky_tip = landmarks.landmark[20]

            # 更新游標位置 (使用食指指尖) 
            cx_percent = max(0, min(1, index_tip.x))
            cy_percent = max(0, min(1, index_tip.y))
            message["hand_x"] = int(cx_percent * 100)
            message["hand_y"] = int(cy_percent * 100)

            # === 2. 手勢識別邏輯 ===
            
            # 指向手勢
            # 食指必須伸直，其他手指 (中、無名、小) 必須彎曲
            # 簡單判斷：食指指尖離手腕的距離 > 食指根部離手腕的距離
            # 且 其他手指指尖離手腕的距離 < 其他手指根部 (
            
            if self.is_pointing_gesture(landmarks.landmark):
                message["status"] = "偵測到指向手勢..."
                
                
                diff_x = index_tip.x - index_mcp.x
                
                # 檢查冷卻時間，避免重複觸發
                if current_time - self.last_action_time > self.COOLDOWN:
                    
                    # 向左指：指尖的 X 小於 根部的 X 
                    if diff_x < -self.POINTING_THRESHOLD:
                        message["action"] = "left"
                        message["status"] = "觸發：向左"
                        print(">>> ACTION: LEFT")
                        self.last_action_time = current_time
                        
                    # 向右指：指尖的 X 大於 根部的 X 
                    elif diff_x > self.POINTING_THRESHOLD:
                        message["action"] = "right"
                        message["status"] = "觸發：向右"
                        print(">>> ACTION: RIGHT")
                        self.last_action_time = current_time

            break 

        return message

    def is_pointing_gesture(self, lm_list):
        """
        判斷是否為「食指指向」手勢
        邏輯：食指伸直，中指、無名指、小指彎曲
        """
        # 取得關鍵點
        wrist = lm_list[0]
        index_tip = lm_list[8]
        index_mcp = lm_list[5]
        middle_tip = lm_list[12]
        middle_mcp = lm_list[9]
        ring_tip = lm_list[16]
        ring_mcp = lm_list[13]
        pinky_tip = lm_list[20]
        pinky_mcp = lm_list[17]

        # 1食指伸直判定 (指尖到手腕距離 > 根部到手腕距離)
        dist_idx_tip = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y)
        dist_idx_mcp = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)
        is_index_straight = dist_idx_tip > dist_idx_mcp

        # 其他手指彎曲判定 (指尖到手腕距離 < 根部到手腕距離，或指尖低於根部)
        dist_mid_tip = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y)
        dist_mid_mcp = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
        
        dist_ring_tip = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y)
        dist_ring_mcp = math.hypot(ring_mcp.x - wrist.x, ring_mcp.y - wrist.y)
        
        # 稍微放寬判定 (* 1.2)，避免手指太長導致誤判
        is_others_curled = (dist_mid_tip < dist_mid_mcp * 1.2) and \
                           (dist_ring_tip < dist_ring_mcp * 1.2)
                           # 小指有時候會翹起來，視情況可以不判斷小指

        return is_index_straight and is_others_curled