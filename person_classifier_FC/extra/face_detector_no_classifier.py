#import libraries
from cv2 import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt


class FaceDetector(object):
  """
  Face detector class
  """
  def __init__(self, mtcnn):
    self.mtcnn = mtcnn

  def draw(self, frame, boxes, probs, landmarks):
    """
    Draw bounding box, probs and landmarks 
    """
    for box, prob, lm in zip(boxes, probs, landmarks):
        #draw  rectangle
        box = box.astype('int') # we want box typle values in integer
        cv2.rectangle(frame,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0,0,255), thickness=3)
        # show probability
        cv2.putText(frame, str(prob),(box[2],box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        #Draw Landmarks
        lm = lm.astype('int')
        cv2.circle(frame, tuple(lm[0]),5, (0,0,255),-1)
        cv2.circle(frame, tuple(lm[1]),5, (0,0,255),-1)
        cv2.circle(frame, tuple(lm[2]),5, (0,0,255),-1)
        cv2.circle(frame, tuple(lm[3]),5, (0,0,255),-1)
        cv2.circle(frame, tuple(lm[4]),5, (0,0,255),-1)

    return frame

  def run(self,test_video):
    v_cap = cv2.VideoCapture(test_video)
    while True:
      success, frame = v_cap.read()
      frame = cv2.resize(frame, (600, 600))
      # frame = Image.fromarray(frame)
      try:
        # Detect face box, probability and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
        #draws a frame on the image
        self.draw(frame, boxes,probs,landmarks)
      except Exception as e: #just show the image if no image on frame
        print(e)
        pass

      cv2.imshow('Face_detect',frame)
      if cv2.waitKey(1)& 0xFF == ord('q'):
        break
    v_cap.release()
    cv2.destroyAllWindows()

# test_video = 0 #for webcam
# test_video = 'videos/test_video_me.mp4'
mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
fcd.run(0)

