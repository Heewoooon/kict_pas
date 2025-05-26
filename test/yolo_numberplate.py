import os
import requests

def safe_download(url, dir='downloads'):
    """지정된 URL에서 파일을 다운로드하고 로컬에 저장합니다."""
    os.makedirs(dir, exist_ok=True)
    filename = os.path.join(dir, url.split('/')[-1])
    
    if os.path.exists(filename):
        print(f"이미 존재함: {filename}")
        return filename

    print(f"다운로드 중: {filename}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("다운로드 완료.")
    else:
        raise Exception(f"다운로드 실패: {url}")

    return filename


# 샘플 비디오 및 모델 파일 다운로드
video_path = safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-video.mp4")
model_path = safe_download("https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-model.pt")

print(f"비디오 파일 위치: {video_path}")
print(f"모델 파일 위치: {model_path}")
