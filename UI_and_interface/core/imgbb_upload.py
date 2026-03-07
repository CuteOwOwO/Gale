import requests

# 將剛剛拿到的 API Key 貼在下面這裡
IMGBB_API_KEY = "417601fefc1f8b6cbb1caede5c5f9bc5" 

def upload_base64_to_imgbb(base64_image: str) -> str:
    """
    將 Base64 圖片上傳到 ImgBB，並回傳公開網址
    """
    url = "https://api.imgbb.com/1/upload"
    
    # ImgBB 的 API 很簡單，只要把 key 和 image 包在 payload 裡用 POST 傳出去就好
    payload = {
        "key": IMGBB_API_KEY,
        "image": base64_image
    }
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            link = response.json()['data']['url']
            return link
        else:
            print(f"⚠️ ImgBB 上傳失敗: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ ImgBB 上傳錯誤: {e}")
        return None