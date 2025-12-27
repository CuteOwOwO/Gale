import google.generativeai as genai
import PIL.Image
import os

# --- 設定 API Key ---
# 建議將 Key 設為環境變數，或直接在此貼上 (注意資安)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDdVPF-G3jCnrkTcSv82daMObWzEoukeRQ"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def compare_poses(ref_image_path, user_image_path):
    """
    使用 Gemini API 比對兩張圖片的動作差異
    """
    # 1. 載入模型 (建議使用 1.5 Flash 速度較快，或 1.5 Pro 精準度較高)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 2. 讀取圖片
    try:
        ref_img = PIL.Image.open(ref_image_path)
        user_img = PIL.Image.open(user_image_path)
    except FileNotFoundError:
        return "錯誤：找不到圖片檔案，請檢查路徑。"

    # 3. 設計 Prompt (提示詞) - 這是最關鍵的部分
    # 我們告訴 Gemini 它是什麼角色，以及我們需要什麼格式的回應
    prompt = """
    你是一位專業的運動與姿態分析專家。我會給你兩張圖片：
    - 第一張圖片 (Image 1) 是「標準參考動作」。
    - 第二張圖片 (Image 2) 是「使用者的嘗試動作」。

    請忽略圖片背景，專注於人物的「骨架姿態」與「肢體角度」。
    請進行詳細的比對分析，並回覆以下內容：

    **主要差異**：列出使用者與標準動作最大的 3 個差異點 (例如：左手手肘抬得不夠高、右膝蓋彎曲角度不足、背部沒有挺直)。
    **具體建議**：針對上述差異，給出具體的修正指令 (例如：請將左手再往上舉約 15 度)。

    請用繁體中文回答。
    """

    # 4. 發送請求
    # Gemini 支援直接傳入 [Text, Image1, Image2] 的列表
    print("正在分析圖片中，請稍候...")
    response = model.generate_content([prompt, ref_img, user_img])

    # 5. 回傳結果
    return response.text

# --- 主程式執行區 ---
if __name__ == "__main__":
    # 假設你有兩張圖：standard.jpg 和 user.jpg
    # 請確保資料夾中有這兩張圖片
    reference_pic = "captured_poses\\auto_221933_286498_overlay.jpg" 
    user_pic = "captured_poses\\auto_221934_540161_overlay.jpg"

    # 檢查檔案是否存在 (為了示範避免報錯)
    if os.path.exists(reference_pic) and os.path.exists(user_pic):
        result = compare_poses(reference_pic, user_pic)
        print("-" * 30)
        print("【Gemini 動作比對報告】")
        print(result)
        print("-" * 30)
    else:
        print(f"請準備兩張圖片並命名為 {reference_pic} 與 {user_pic} 以進行測試。")