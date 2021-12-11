from facenet_pytorch import MTCNN
from PIL import Image
import os
path = 'Users/Abdullah Mansoor/PycharmProjects/opencv/video_face_recognition/face_video/images/asd'

mtcnn = MTCNN()

num_files = list(range(226)) #226 is the number of other images

save_paths = [f'detected_{i}.jpg' for i in num_files]

for file, new_path in zip(sorted(os.listdir(path)), save_paths):
    if file[-1] =='g': #only if file filename end with "g" like "jpg"
        img = Image.open(path +'/'+file)
        mtcnn(img, save_path = new_path)
