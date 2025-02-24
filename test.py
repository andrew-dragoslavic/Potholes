from ultralytics import YOLO
import numpy as np
from sort import Sort
import cv2

cap = cv2.VideoCapture('test.mp4')

model = YOLO("best.pt")
