# core/pose.py
import cv2
import mediapipe as mp

class PoseEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # 這裡可以調整偵測信心度
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        輸入: 原始 BGR 圖片
        輸出: (處理後的 RGB 圖片, MediaPipe 結果物件)
        """
        # 影像處理 (轉 RGB 並鏡像)
        # 鏡像：畫面左邊是使用者的左手，畫面右邊是使用者的右手
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1) 
        results = self.pose.process(frame_rgb)
        
        return frame_rgb, results