import cv2
from ultralytics import YOLO


video_path = '../video/output_DJI_0329.MP4'
cap = cv2.VideoCapture(video_path)

# YOLO
model = YOLO("../model/korea_0423_batch.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        tracker="../config/bytetrack.yaml",
        conf=0.3,
        show=False,
        persist=True,
        verbose=False,
        imgsz=640,
        iou=0.3,
        device=0,
        agnostic_nms=True
    )[0]

    # 결과 시각화
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 프레임 출력
    cv2.imshow("YOLO Detection", frame)

    # 종료 키 (q)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
