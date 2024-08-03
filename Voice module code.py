#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Tuple, Optional
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
import serial  # For reading from the DFRobot voice recognition sensor

class RealSenseCamera:
    def __init__(self):
        super().__init__()
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        
        # Initialize YOLOv8 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()
        
        # Initialize voice recognition sensor (assuming connected via serial)
        self.ser = serial.Serial('COM3', 9600)  # Adjust COM port and baud rate as necessary

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

    def read_voice_command(self) -> str:
        if self.ser.in_waiting > 0:
            command = self.ser.readline().decode().strip()
            return command
        return ""

def main():
    camera = RealSenseCamera()
    camera.capture()
    
    while True:
        camera.update_frame()
        frame = camera.color_frame()
        if frame is not None:
            results = camera.detect_objects(frame)
            frame_with_boxes = results.render()[0]
            cv2.imshow("Detection", frame_with_boxes)
        
        # Read voice command
        voice_command = camera.read_voice_command()
        
        if voice_command:
            print(f"Voice command received: {voice_command}")
            # Process the command and take action
            if voice_command == "cube":
                # Handle the cube command (e.g., move the robotic arm to interact with a cube)
                pass
            elif voice_command == "cuboid":
                # Handle the cuboid command
                pass
            elif voice_command == "cylinder":
                # Handle the cylinder command
                pass
            elif voice_command == "hexagonal prism":
                # Handle the hexagonal prism command
                pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# In[ ]:




