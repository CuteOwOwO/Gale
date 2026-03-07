# try_qrcode.py 的修改
import qrcode
import io
import base64

def generate_qr_code_base64(image_url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(image_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    
    # 將圖片存入記憶體緩衝區
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    
    # 轉成 Base64
    qr_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return qr_base64