import cv2
import mediapipe as mp

class HandEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        輸入: 原始 BGR 圖片 (從 cv2.VideoCapture 讀進來的)
        輸出: (處理後的 RGB 鏡像圖片, MediaPipe Hands 結果物件)
        """
        # 1. 轉 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. 鏡像翻轉 (Flip)
        frame_rgb = cv2.flip(frame_rgb, 1) 
        
        # 3. 推論
        results = self.hands.process(frame_rgb)
        
        return frame_rgb, results