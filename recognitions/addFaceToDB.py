import datetime
import os

import cv2


def addFaceToDB(face_id, face_name, filePath):
    print("\n [INFO] Initializing face capture ...")

    # Assuming you have a FileStorage object named 'fs' that contains the video data

    # Save the FileStorage object to a temporary file
    # Open the video file with cv2.VideoCapture
    cap = cv2.VideoCapture(filePath)

    count = 0
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    ret, img = cap.read()
    timeString = datetime.datetime.now().strftime('%d-%m-%Y')

    # Define the path where you want to create the directory
    pathToUserDir = "dataset/User-" + str(face_name) + "-" + str(face_id) + "/" + timeString + "/"
    if not ret:
        print("----Empty Video----")
        return
    # Check if the directory already exists
    if not os.path.exists(pathToUserDir):
        # Use the os.makedirs() function to create the directory
        os.makedirs(pathToUserDir)
    else:
        print("Directory already exists.")
    print(count)
    while count < 100 and ret:
        # img = cv2.flip(img, -1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite(pathToUserDir + "/" + str(face_name) + '-' + str(
                count) + ".jpg",
                        gray[y:y + h, x:x + w])

            print(str(int(100 * (count / 60))) + " %")
            count += 1
        ret, img = cap.read()

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cap.release()
