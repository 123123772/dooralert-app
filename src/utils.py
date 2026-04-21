import base64
import cv2
import numpy as np

def numpy_to_base64(img: np.ndarray) -> str:
    """将OpenCV图像（BGR）转换为base64字符串，用于Flet显示"""
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"