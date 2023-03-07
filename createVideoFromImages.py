import cv2
import numpy as np
import glob

frameSize = (800, 600)

out = cv2.VideoWriter('output_video.mov',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

for filename in sorted(glob.glob('/Users/tomer/PycharmProjects/data-attendance/FaceDetectionDB/P1E_S1/P1E_S1_C1/*.jpg')):
    print(filename)
    img = cv2.imread(filename)
    out.write(img)

out.release()