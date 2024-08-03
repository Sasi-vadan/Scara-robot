#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import Tuple, Optional
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch 

class RealSenseCamera:
    def _init_(self):
        super()._init_()
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        
        # Initialize YOLOv8 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

    def capture(self):
        # Start streaming
        self.pipeline.start(self.config)
        # Warm up
        for _ in range(60):
            frames = self.pipeline.wait_for_frames()

    def release(self):
        self.pipeline.stop()

    def update_frame(self) -> None:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.curr_frame = aligned_frames
        self.curr_frame_time = time.time_ns()

    def color_frame(self) -> Optional[np.ndarray]:
        frame = self.curr_frame.get_color_frame()
        if not frame:
            return None
        frame = np.asanyarray(frame.get_data())
        return frame

    def detect_objects(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame)
        return results

def main():
    camera = RealSenseCamera()
    camera.capture()
    
    while True:
        camera.update_frame()
        frame = camera.color_frame()
        if frame is not None:
            results = camera.detect_objects(frame)
            # Display results
            frame_with_boxes = results.render()[0]  
#Render bounding boxes on the frame
            cv2.imshow("Detection", frame_with_boxes)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    main()
    
    
    
    
# init color
def __init__(self) -> None:
self.area_low_threshold = 15000
self.detected_name = None
self.hsv_range = {
"green": ((40, 50, 50), (90, 256, 256)),
# "blueA": ((91, 100, 100), (105, 256, 256)),
# "yellow": ((20, 240, 170), (30, 256, 256)),
"yellow": ((15, 46, 43), (30, 256, 256)),
"redA": ((0, 100, 100), (6, 256, 256)),
"redB": ((170, 100, 100), (179, 256, 256)),
# "orange": ((8, 100, 100), (15, 256, 256)),
"blue": ((100, 43, 46), (124, 256, 256)),
}
# process of image
result = []
for color, (hsv_low, hsv_high) in self.hsv_range.items():
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
in_range = cv2.inRange(hsv_frame, hsv_low, hsv_high)
# Dilate and corrode color areas
kernel = np.ones((5, 5), np.uint8)
in_range = cv2.morphologyEx(in_range, cv2.MORPH_CLOSE, kernel)
in_range = cv2.morphologyEx(in_range, cv2.MORPH_OPEN, kernel)
contours, hierarchy = cv2.findContours(
in_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = list(
filter(lambda x: cv2.contourArea(x) > self.area_low_threshold, contours)
)
rects = list(map(cv2.minAreaRect, contours))
boxes = list(map(cv2.boxPoints, rects))
boxes = list(map(np.int32, boxes))
if len(boxes) != 0:
if color.startswith("red"):
color = "red"
for box in boxes:
result.append(ColorDetector.DetectResult(color, box))
# self.detected_name = result
self.detected_name = result[0].color
return result

