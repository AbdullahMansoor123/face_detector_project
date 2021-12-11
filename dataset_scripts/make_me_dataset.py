import cv2
from facenet_pytorch import MTCNN
import numpy as np

mtcnn = MTCNN()

#load the video
v_cap = cv2.VideoCapture('test_video_3.mp4')

#get the frame count
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
print(v_len)
for i in range(v_len):
    #load the fames
    success, frame = v_cap.read()
    if not success:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

#saving images with specific name format
save_paths = [f'image_{i}.jpg' for i in range(len(frames))]

for frame, path in zip(frames,save_paths):
    mtcnn(frames,save_path=path)
