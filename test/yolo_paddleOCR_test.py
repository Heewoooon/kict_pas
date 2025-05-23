import cv2
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR  
import numpy as np

# === 설정 ===
RTSP_URL = '../video/output_DJI_0328.MP4'
TARGET_CLASS_NAME = "license_plate"  # YOLO 모델에서 번호판 클래스 이름
TARGET_CLASS_ID = 0  # 번호판 클래스 ID (YOLO 모델에 따라 다름)

# YOLO 모델 로딩
model = YOLO("yolo11n.pt")  # 또는 다른 YOLOv8/11 모델

# PaddleOCR 로딩 (GPU 사용)
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 비디오 스트림
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("RTSP 스트림을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    results = model.predict(
        source=frame,
        iou = 0.3
    )[0]

    for box in results.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())


        plate_crop = frame[y1:y2, x1:x2]

        # OCR 적용
        ocr_result = ocr.ocr(plate_crop, cls=True)
        if ocr_result[0] :
            for line in ocr_result[0]:
                text = line[1][0]
                confidence = line[1][1]

                # 결과 출력
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{text} ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("YOLO + OCR RTSP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
