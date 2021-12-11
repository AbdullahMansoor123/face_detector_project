#import libraries
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

class_map = ['abdullah', 'not_abdullah']

class FaceDetector(object):
    """
    Face detector class
    """
    def __init__(self, mtcnn,classifier):
        self.mtcnn = mtcnn
        self.classifier = classifier

    # def class_to_label(self, x):
    # """
    # For a given label value, return corresponding string label.
    # """
    #     return self.class_map[int(x)]
    ##Detector##
    def draw(self,prediction, frame, boxes, probs, landmarks):
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
            cv2.putText(frame, class_map[int(prediction)],(box[2],box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            #Draw Landmarks
            # lm = lm.astype('int')
            # cv2.circle(frame, tuple(lm[0]),5, (0,0,255),-1)
            # cv2.circle(frame, tuple(lm[1]),5, (0,0,255),-1)
            # cv2.circle(frame, tuple(lm[2]),5, (0,0,255),-1)
            # cv2.circle(frame, tuple(lm[3]),5, (0,0,255),-1)
            # cv2.circle(frame, tuple(lm[4]),5, (0,0,255),-1)

        return frame
    ##Classifier##
    def is_it_me(self,frame):
        """
        Run Classifier on face
        """
        rgb_img =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        PIL_img = Image.fromarray(rgb_img.astype('uint8'), 'RGB')
        transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        transformed_img = transform(PIL_img)
        transformed_img = torch.unsqueeze(transformed_img, 0)#fake batch dimension

        #Inference
        with torch.no_grad():
            outputs = self.classifier(transformed_img)
            prediction = torch.argmax(outputs, dim=1)
        return prediction

    ##Face Recognizer##
    def run(self,test_video):
        v_cap = cv2.VideoCapture(test_video)
        while True:
            success, frame = v_cap.read()

            frame = cv2.resize(frame, (600, 600))
            try:
                # Detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

                ## Predict the image class##
                prediction = self.is_it_me(frame)

                #draws a frame on the image
                self.draw(prediction,frame, boxes,probs,landmarks)

            except Exception as e: #just keep showing video feed if no image on frame is detected
                print(e)
                pass

            cv2.imshow('Face_detect',frame)
            if cv2.waitKey(1)& 0xFF == ord('q'):
                break
        v_cap.release()
        cv2.destroyAllWindows()


