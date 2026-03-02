import os
import shutil
import re

# 設定來源與目的資料夾路徑 (使用 r 前綴來避免 Windows 路徑的斜線跳脫問題)
source_dir = r"C:\Users\Angus\Desktop\dance\final_best_poses"
dest_dir = r"C:\Users\Angus\Desktop\dance\UI_and_interface\static\frames"

# 確保目的資料夾存在，如果沒有就會自動建立
os.makedirs(dest_dir, exist_ok=True)

# 準備好用來抓取檔名數字的正規表達式 (比對 best_pose_frame_數字.jpg)
pattern = re.compile(r"best_pose_frame_(\d+)\.jpg", re.IGNORECASE)

moved_count = 0

print(f"🔍 開始掃描資料夾: {source_dir}")

# 掃描來源資料夾中的所有檔案
for filename in os.listdir(source_dir):
    match = pattern.match(filename)
    
    # 如果檔名符合我們的規則
    if match:
        # 抓出括號裡對應的影格數字 (例如 '35')
        frame_number = match.group(1)
        
        # 組裝新的檔名
        new_filename = f"taichi_{frame_number}.jpg"
        
        # 取得完整的來源與目的檔案路徑
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, new_filename)
        
        try:
            # 🚀 執行移動檔案 
            # (💡小提醒：如果你想保留原本的檔案以防萬一，可以把 shutil.move 改成 shutil.copy2)
            shutil.move(source_path, dest_path)
            print(f"✅ 成功: {filename} -> {new_filename}")
            moved_count += 1
        except Exception as e:
            print(f"❌ 發生錯誤 ({filename}): {e}")

print(f"🎉 執行完畢！總共處理了 {moved_count} 個檔案。")