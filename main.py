from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR

# 1. 모델 로딩
license_plate_detector = YOLO("./models/license_plate_detector.pt")  # YOLO 번호판 모델 경로
ocr = PaddleOCR(use_angle_cls=True, lang='korean')  # 한국어 + 영어 + 숫자 OCR 지원

# 2. 비디오 로딩
cap = cv2.VideoCapture("./video/DJI_0330.MP4.mp4")  # 영상 경로

# 3. 프레임 순회
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLO로 번호판 검출
    results = license_plate_detector.track(frame, persist=True)

    if results and results[0].boxes is not None:
        for bbox in results[0].boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]

            # 5. PaddleOCR 실행
            ocr_result = ocr.ocr(plate_crop)

            if ocr_result and len(ocr_result) > 0:
                rec_texts = ocr_result[0].get("rec_texts", [])
                rec_scores = ocr_result[0].get("rec_scores", [])

                # 6. score 0.5 이상 필터링
                filtered_texts = [text for text, score in zip(rec_texts, rec_scores) if score > 0.5]
                text = ''.join(filtered_texts)

                # 7. 시각화
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 8. 프레임 출력
    cv2.imshow("YOLO + PaddleOCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. 자원 해제
cap.release()
cv2.destroyAllWindows()
