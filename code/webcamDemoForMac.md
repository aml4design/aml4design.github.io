---
layout: default
title: "Webcam Demo for Mac"
has_children: false
parent: "Code"
nav_order: 1
# nav_exclude: true
---

This is the code snippet of the demo that will be shown in Lecture 5.  

```python
# Before running this file, run following codes in your terminal:
# note that MPS(GPU) acceleration is available on MacOS 12.3+

# conda install pytorch torchvision torchaudio -c pytorch-nightly 
# export PYTORCH_ENABLE_MPS_FALLBACK=1


import torch
import numpy as np
import cv2
from time import time


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a given video.
    """

    def __init__(self):
        """
        Initializes the class with youtube url and output file.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        if torch.backends.mps.is_available():
            print("#### GPU running! ####")
            self.device = 'mps'  
        else:
            self.device = 'cpu'


    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        It returna a trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        Returns labels and coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        Returns a corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        Returns a frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        Returns void
        """
        player = cv2.VideoCapture(0)
        while (True):
            start_time = time()
            ret, frame = player.read()

            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            if results[0].size != 0:
                print("Labels: ")
                for label_val in results[0]:
                    print(self.class_to_label(label_val))
                        # Display the resulting frame
            
            
                    cv2.imshow('Frame', frame)
                # Press Q on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        # When everything is done, release the video capture object
        player.release()
        # Closes all the frames
        cv2.destroyAllWindows()

# Create a new object and execute.
a = ObjectDetection()
a()
```

Written by 
[Tianhao He](https://github.com/Tianhao1997)