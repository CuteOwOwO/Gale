# core/game_engine.py
import random
import cv2
import mediapipe as mp
import numpy as np
import base64
import json
import os


from core.pose_utils import get_torso_geometry, compare_poses_strict

class GameEngine:
    def __init__(self, video_id: str, frame_id: int):
        self.mp_pose = mp.solutions.pose
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.ghost_alpha = 0.3 

        # 讀取 config.json
        with open("static/config.json", "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # 檢查是否有這個影片的設定，沒有就給預設值防呆
        if video_id in config_data:
            self.video_path = config_data[video_id]["video_path"]
        else:
            print(f"⚠️ 找不到影片 {video_id} 的設定，使用預設影片")
            self.video_path = "../mp4_home/old.mp4"

        self.target_frame = frame_id
        
        print(f"[{video_id}] 正在準備 AR 素材 (影片: {self.video_path}, 影格: {self.target_frame})...")
        self._pre_process_teacher()

        self.filter_type = random.choice(["rabbit", "sunglasses"]) 
        
        if self.filter_type == "rabbit":
            filter_path = "static/rabbit.png"
        elif self.filter_type == "sunglasses":
            filter_path = "static/sunglasses.png"
            
        if os.path.exists(filter_path):
            self.filter_img = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
            print(f"🎁 盲盒開獎！這局抽到的濾鏡是：{self.filter_type}!")
        else:
            self.filter_img = None
            print(f"⚠️ 找不到濾鏡圖片: {filter_path}")

    def _pre_process_teacher(self):
        """
        內部方法：讀取老師的指定影格，擷取骨架、去背遮罩與邊框
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
        success, image = cap.read()
        cap.release()
        
        if not success:
            print("❌ 老師影像讀取失敗，請確認路徑！")
            self.t_landmarks = None
            return

        # 1. 抓取老師骨架
        with self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            self.t_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        # 2. 抓取老師去背遮罩
        with self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
            results = selfie_seg.process(img_rgb)
            self.t_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

        # 3. 製作霓虹邊框
        edges = cv2.Canny(self.t_mask, 100, 200)
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        self.t_neon = np.zeros_like(image)
        self.t_neon[edges_dilated > 0] = [0, 255, 255] # 黃色霓虹

        self.t_img = image
        
        # 4. 預先計算老師軀幹中心與大小
        if self.t_landmarks:
            self.t_center, self.t_size = get_torso_geometry(self.t_landmarks, self.t_img.shape)
        else:
            self.t_center, self.t_size = None, None

    def overlay_transparent(self, background, overlay, x, y):
        """將帶有透明通道的 PNG 完美疊加到背景上，並防止超出邊界報錯"""
        bg_h, bg_w, bg_channels = background.shape
        ol_h, ol_w, ol_channels = overlay.shape

        if ol_channels < 4:
            return background # 如果圖片沒有透明通道，就不疊加

        # 計算超出邊界時的安全裁切範圍
        y1, y2 = max(0, y), min(bg_h, y + ol_h)
        x1, x2 = max(0, x), min(bg_w, x + ol_w)
        y1o, y2o = max(0, -y), min(ol_h, bg_h - y)
        x1o, x2o = max(0, -x), min(ol_w, bg_w - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return background

        # 進行 Alpha Blending
        alpha = overlay[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv = 1.0 - alpha

        for c in range(3):
            background[y1:y2, x1:x2, c] = (alpha * overlay[y1o:y2o, x1o:x2o, c] +
                                          alpha_inv * background[y1:y2, x1:x2, c])
        return background

    def process_frame(self, user_frame):
        """
        處理使用者的單張影像：計算分數、AR疊圖、畫骨架，並回傳 Base64
        """
        frame = cv2.flip(user_frame, 1)
        
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(frame_rgb)
        current_score = 0
        
        # 預設沒有呼叫教練
        is_calling_coach = False 
        
        if results.pose_landmarks and self.t_landmarks and self.t_center is not None:
            u_landmarks = results.pose_landmarks.landmark
            
    
            left_shoulder_y = u_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = u_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            left_wrist_y = u_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
            right_wrist_y = u_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            left_wrist_vis = u_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].visibility
            right_wrist_vis = u_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility

            
            height_margin = 0.1 

            if (left_wrist_y < left_shoulder_y - height_margin and 
                right_wrist_y < right_shoulder_y - height_margin and 
                left_wrist_vis > 0.7 and right_wrist_vis > 0.7):
                
                is_calling_coach = True
                print(">>> 使用者正在明確地呼叫教練！")
           
            
            # 計算分數
            current_score = compare_poses_strict(u_landmarks, self.t_landmarks)
            
            # 計算使用者的軀幹並進行 AR 疊圖
            u_center, u_size = get_torso_geometry(u_landmarks, frame.shape)
            
            if u_center is not None and u_size > 0:
                scale = u_size / self.t_size
                tx = u_center[0] - self.t_center[0] * scale
                ty = u_center[1] - self.t_center[1] * scale
                M = np.float32([[scale, 0, tx], [0, scale, ty]])
                
                warped_t_img = cv2.warpAffine(self.t_img, M, (w, h))
                warped_t_mask = cv2.warpAffine(self.t_mask, M, (w, h))
                warped_t_neon = cv2.warpAffine(self.t_neon, M, (w, h))
                
                mask_3ch = cv2.merge([warped_t_mask, warped_t_mask, warped_t_mask])
                mask_indices = np.where(mask_3ch > 128)
                
                overlay = frame.copy()
                overlay[mask_indices] = warped_t_img[mask_indices]
                cv2.addWeighted(overlay, self.ghost_alpha, frame, 1 - self.ghost_alpha, 0, frame)
                
                neon_indices = np.where(warped_t_neon > 0)
                frame[neon_indices] = warped_t_neon[neon_indices]

            # 畫上使用者的骨架 (根據分數變色)
            skel_color = (0, 0, 255) 
            if current_score > 60: skel_color = (0, 255, 255) 
            if current_score > 85: skel_color = (0, 255, 0)   

            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=skel_color, thickness=2, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=skel_color, thickness=2)
            )

            if hasattr(self, 'filter_img') and self.filter_img is not None:
                import math
                
                # -------------------------
                # 🐰 抽中兔耳朵的邏輯
                # -------------------------
                if self.filter_type == "rabbit":
                    left_ear = u_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
                    right_ear = u_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
                    
                    lx, ly = int(left_ear.x * w), int(left_ear.y * h)
                    rx, ry = int(right_ear.x * w), int(right_ear.y * h)
                    
                    ear_dist = math.hypot(lx - rx, ly - ry)
                    
                    if ear_dist > 20: 
                        target_w = int(ear_dist * 1.8)
                        orig_h, orig_w = self.filter_img.shape[:2]
                        target_h = int(orig_h * (target_w / orig_w)) 
                        
                        if target_w > 0 and target_h > 0:
                            resized_filter = cv2.resize(self.filter_img, (target_w, target_h))
                            # 兔耳朵旋轉 (記得加上負號反轉)
                            angle = -math.degrees(math.atan2(ly - ry, lx - rx))
                            M = cv2.getRotationMatrix2D((target_w // 2, target_h // 2), angle, 1.0)
                            rotated_filter = cv2.warpAffine(resized_filter, M, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                            
                            cx, cy = (lx + rx) // 2, (ly + ry) // 2
                            offset_x = cx - target_w // 2
                            offset_y = cy - int(target_h * 1.2) 
                            
                            frame = self.overlay_transparent(frame, rotated_filter, offset_x, offset_y)
                            
                # -------------------------
                # 抽中墨鏡的邏輯
                # -------------------------
                elif self.filter_type == "sunglasses":
                    left_eye = u_landmarks[self.mp_pose.PoseLandmark.LEFT_EYE.value]
                    right_eye = u_landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE.value]
                    left_ear = u_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
                    right_ear = u_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
                    
                    lex, ley = int(left_eye.x * w), int(left_eye.y * h)
                    rex, rey = int(right_eye.x * w), int(right_eye.y * h)
                    lax, lay = int(left_ear.x * w), int(left_ear.y * h)
                    rax, ray = int(right_ear.x * w), int(right_ear.y * h)
                    
                    face_width = math.hypot(lax - rax, lay - ray)
                    
                    if face_width > 20: 
                        target_w = int(face_width * 1.6)
                        orig_h, orig_w = self.filter_img.shape[:2]
                        target_h = int(orig_h * (target_w / orig_w)) 
                        
                        if target_w > 0 and target_h > 0:
                            resized_filter = cv2.resize(self.filter_img, (target_w, target_h))
                            # 墨鏡旋轉 (以眼睛為基準，記得加上負號反轉)
                            angle = -math.degrees(math.atan2(ley - rey, lex - rex))
                            M = cv2.getRotationMatrix2D((target_w // 2, target_h // 2), angle, 1.0)
                            rotated_filter = cv2.warpAffine(resized_filter, M, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                            
                            # 墨鏡位置 (對準兩眼中間)
                            cx, cy = (lex + rex) // 2, (ley + rey) // 2
                            offset_x = cx - target_w // 2
                            offset_y = cy - target_h // 2 
                            
                            frame = self.overlay_transparent(frame, rotated_filter, offset_x, offset_y)
        self.latest_overlay_frame = frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        

       
        return current_score, jpg_as_text, is_calling_coach
    def get_overlay_frame_base64(self):
        """
        取得包含骨架與 AR 疊圖的使用者畫面 (Base64格式)
        """
        if hasattr(self, 'latest_overlay_frame'):
            # 將備份的完妝畫面編碼為 JPG，再轉 Base64
            _, buffer = cv2.imencode('.jpg', self.latest_overlay_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return jpg_as_text
        return None