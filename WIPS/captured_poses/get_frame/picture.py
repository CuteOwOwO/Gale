import cv2
import mediapipe as mp

# --- 設定區 ---
INPUT_IMAGE = r'C:\Users\Angus\Desktop\dance\gemini_response\senior_cards\senior_card_1772437414.png'   # 你的輸入照片路徑
OUTPUT_IMAGE = 'output.jpg' # 輸出結果路徑
# -------------

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_single_image(image_path, output_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"錯誤：找不到圖片 {image_path}")
        return

    # 建立 Pose 物件
    with mp_pose.Pose(
        static_image_mode=True, 
        min_detection_confidence=0.5) as pose:

        # 轉換 BGR 到 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # 如果有偵測到骨架，就畫在圖上
        if results.pose_landmarks:
            print("成功偵測到骨架！")
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                # 關鍵點：保持預設樣式 (彩色點)
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                # 連線：設定為白色 (BGR: 255, 255, 255)
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
        else:
            print("未偵測到骨架。")

        # 儲存結果
        cv2.imwrite(output_path, image)
        print(f"結果已存至: {output_path}")

        # (選擇性) 顯示圖片
        cv2.imshow('Pose Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    process_single_image(INPUT_IMAGE, OUTPUT_IMAGE)