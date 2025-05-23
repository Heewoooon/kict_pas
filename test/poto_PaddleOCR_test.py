# from paddleocr import PaddleOCR, draw_ocr
# import cv2
# from PIL import Image

# ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # 또는 lang='ko'

# image_path = 'car_ocr_test1.png'
# img = cv2.imread(image_path)

# # OCR 수행
# result = ocr.ocr(image_path, cls=True)

# for line in result[0]:
#     box, (text, confidence) = line
#     print(f"인식: {text} (정확도: {confidence:.2f})")

from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='korean')
result = ocr.ocr('../video/car_ocr_test1.png')

for line in result[0]:
    print(line)
    
