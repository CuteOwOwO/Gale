import requests
import base64
import os

# 🛑 第一步：把你在 ImgBB 申請到的 API Key 貼在下面
IMGBB_API_KEY = "417601fefc1f8b6cbb1caede5c5f9bc5"

# 🛑 第二步：隨便找一張你電腦裡確定存在的圖片來測試 (也可以用之前的太極圖)
TEST_IMAGE_PATH = r"C:\Users\Angus\Desktop\dance\get_frame\taichi.png" 

def test_upload():
    print("=== ImgBB API 獨立上傳測試開始 ===")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 找不到測試圖片: {TEST_IMAGE_PATH}")
        print("💡 請修改 TEST_IMAGE_PATH 指向一張真實存在的圖片！")
        return

    print(f"📦 1. 正在讀取並轉換圖片: {os.path.basename(TEST_IMAGE_PATH)} ...")
    try:
        with open(TEST_IMAGE_PATH, "rb") as image_file:
            # 讀取二進位檔案，並轉成 Base64 字串
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ 圖片讀取失敗: {e}")
        return

    print("🚀 2. 正在發送 API 請求到 ImgBB ... (請稍候)")
    url = "https://api.imgbb.com/1/upload"
    payload = {
        "key": IMGBB_API_KEY,
        "image": base64_image
    }

    try:
        response = requests.post(url, data=payload)
        
        # 檢查 HTTP 狀態碼是否為 200 (OK)
        if response.status_code == 200:
            result = response.json()
            img_url = result['data']['url']
            delete_url = result['data']['delete_url'] # ImgBB 還會貼心附上刪除連結
            
            print("\n🎉 測試成功！ImgBB 順利接收並回傳資料：")
            print(f"🔗 圖片公開網址: {img_url}")
            print(f"🗑️ (若想刪除這張測試圖，可點擊此連結: {delete_url})")
            print("\n💡 結論：這代表你的 API Key 和網路連線都完全沒問題，可以放心整合進 main.py 了！")
        else:
            print("\n❌ 上傳失敗！")
            print(f"HTTP 狀態碼: {response.status_code}")
            print(f"伺服器錯誤訊息: {response.text}")
            print("💡 請檢查你的 API Key 是否複製完整，或圖片是否過大。")

    except Exception as e:
        print(f"\n❌ 網路請求發生未知錯誤: {e}")
        print("💡 請檢查你的網路連線是否正常。")

if __name__ == "__main__":
    test_upload()