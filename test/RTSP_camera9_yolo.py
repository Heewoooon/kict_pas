import cv2
import numpy as np
import onnxruntime as ort
from threading import Thread
from queue import Queue
import time

# YOLO 모델 로드
session = ort.InferenceSession("yolov5s.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# RTSP 스트림 리스트
rtsp_urls = [f"rtsp://user:pass@192.168.0.10{i}/stream" for i in range(1, 10)]
frame_queues = [Queue(maxsize=1) for _ in range(len(rtsp_urls))]

def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    return img.astype(np.float32)[np.newaxis, ...]

def stream_reader(index, url):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if ret:
            timestamp = time.time()
            if frame_queues[index].full():
                frame_queues[index].get()
            frame_queues[index].put((timestamp, frame))

def synchronized_processor():
    while True:
        aligned_frames = []
        current_time = time.time()

        for q in frame_queues:
            matched = None
            while not q.empty():
                ts, frame = q.queue[0]
                if abs(ts - current_time) <= 0.2:  # 허용 오차 0.2초
                    matched = q.get()
                    break
                else:
                    q.get()  # 오래된 프레임 버림
            aligned_frames.append(matched)

        if all(f is not None for f in aligned_frames):
            for i, (_, frame) in enumerate(aligned_frames):
                inp = preprocess(frame)
                outputs = session.run(None, {input_name: inp})[0]
                print(f"[SYNC Cam{i+1}] Detections: {outputs.shape}")

# 각 스트림 스레드 실행
for i, url in enumerate(rtsp_urls):
    Thread(target=stream_reader, args=(i, url), daemon=True).start()

Thread(target=synchronized_processor, daemon=True).start()

cv2.waitKey(0)