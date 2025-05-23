import cv2
import easyocr

# EasyOCR Reader 객체 생성 (한국어 + 영어 지원)
reader = easyocr.Reader(['en', 'ko'])

video_path = '../video/output_DJI_0329.MP4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # OCR 적용 (frame은 BGR이므로 RGB로 변환)
    results = reader.readtext(frame)

    # 결과 표시
    for (bbox, text, conf) in results:
        # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        pts = [tuple(map(int, point)) for point in bbox]
        x_min, y_min = pts[0]
        x_max, y_max = pts[2]

        # 사각형 그리기
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # 텍스트 출력
        cv2.putText(frame, text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 출력
    cv2.imshow("Text Detection with EasyOCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
