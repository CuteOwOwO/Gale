import cv2
import os

# --- 設定區 ---
VIDEO_PATH = 'old.mp4'       # 你的影片路徑
OUTPUT_FILE = 'ground_truth.txt' # 輸出的標記檔
# -------------

def label_ground_truth(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    window_name = "Labeling Tool (Space:Pause, A/D:Step, M:Mark, C:Clear, Q:Save)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    marked_frames = []
    is_paused = False
    current_frame_pos = 0
    frame = None

    print("=== 開始標記 (V2 安全版) ===")
    print("即使程式發生錯誤，也會嘗試在關閉前自動存檔。")

    # 先讀第一幀，確保 frame 不為空
    ret, frame = cap.read()
    if not ret:
        print("無法讀取影片第一幀")
        return

    try:  # <--- 加入 try 區塊，保護資料
        while True:
            # --- 核心邏輯修正區 ---
            if frame is None:
                # 如果因為任何原因 frame 是空的，嘗試重新讀取當前位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("無法讀取畫面，可能是影片結束或格式問題，即將存檔退出。")
                    break

            display_img = frame.copy()

            # --- 介面顯示 ---
            # 1. 資訊
            cv2.putText(display_img, f"Frame: {current_frame_pos}/{total_frames}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 2. 狀態
            status_text = "PAUSED (A/D to step)" if is_paused else "PLAYING"
            cv2.putText(display_img, status_text, (30, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if is_paused else (0, 255, 0), 2)

            # 3. 數量
            cv2.putText(display_img, f"Marked: {len(marked_frames)}", (30, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

            # 4. 標記狀態
            if current_frame_pos in marked_frames:
                cv2.putText(display_img, "=== MARKED ===", (300, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            cv2.imshow(window_name, display_img)

            # --- 按鍵處理 ---
            key = cv2.waitKey(30 if not is_paused else 0) & 0xFF 

            if key == ord('q'):
                print("使用者主動結束。")
                break

            elif key == ord(' '):
                is_paused = not is_paused

            elif key == ord('m'):
                if current_frame_pos not in marked_frames:
                    marked_frames.append(current_frame_pos)
                    marked_frames.sort()
                    print(f"[標記] Frame {current_frame_pos}")
                else:
                    print(f"[重複] Frame {current_frame_pos} 已存在")

            elif key == ord('c'):
                if marked_frames:
                    removed = marked_frames.pop()
                    print(f"[移除] Frame {removed}")

            # --- 播放與跳幀控制 (修正後) ---
            if not is_paused:
                # 正常播放
                ret, next_frame = cap.read()
                if ret and next_frame is not None:
                    frame = next_frame
                    current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                else:
                    print("影片播放結束，自動暫停。")
                    is_paused = True
                    # 修正：不要讀取超出範圍的幀，停留在最後一幀
                    current_frame_pos = max(0, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    # 重新讀取最後一幀給 display_img 用
                    ret, frame = cap.read() 
            else:
                # 暫停時的控制
                target_pos = current_frame_pos
                if key == ord('a'): # 上一幀
                    target_pos = max(0, current_frame_pos - 1)
                elif key == ord('d'): # 下一幀
                    target_pos = min(total_frames - 1, current_frame_pos + 1)
                
                if target_pos != current_frame_pos:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                    ret, next_frame = cap.read()
                    if ret and next_frame is not None:
                        frame = next_frame
                        current_frame_pos = target_pos

    except Exception as e:
        print(f"\n!!!! 發生未預期的錯誤: {e} !!!!")
    
    finally:
        # --- 無論如何都會執行的存檔區 ---
        print("\n正在執行安全存檔程序...")
        cap.release()
        cv2.destroyAllWindows()

        if marked_frames:
            # 移除重複並排序 (雙重保險)
            unique_frames = sorted(list(set(marked_frames)))
            
            with open(OUTPUT_FILE, 'w') as f:
                for frame_idx in unique_frames:
                    f.write(f"{frame_idx}\n")
            print(f"[存檔成功] 已搶救/儲存 {len(unique_frames)} 個關鍵點到 '{OUTPUT_FILE}'")
            print(f"數據內容: {unique_frames}")
        else:
            print("[結束] 沒有標記任何數據。")

if __name__ == "__main__":
    if os.path.exists(VIDEO_PATH):
        label_ground_truth(VIDEO_PATH)
    else:
        print(f"找不到影片檔案: {VIDEO_PATH}")