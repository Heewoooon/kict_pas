import cv2
from ultralytics import YOLO
import easyocr

# YOLO
# model = YOLO("../model/korea_0423_batch.pt")
model = YOLO("yolo11n.pt")

# OCR 로더
reader = easyocr.Reader(['en', 'ko'])

video_path = '../video/DJI_0330.MP4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        iou = 0.3
    )[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = frame[y1:y2, x1:x2]  # 번호판 영역 crop

        # OCR 적용
        ocr_results = reader.readtext(cropped)

        # OCR 결과 시각화
        for (_, text, _) in ocr_results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 출력
    cv2.imshow("License Plate Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
