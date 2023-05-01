import datetime
import os
import pickle

import cv2
import numpy as np
from PIL import Image

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_SHAPE = (480, 640, 3)


class RecognitionService():

    def __init__(self, isLogin=True):
        self.face_recogniser = cv2.face.LBPHFaceRecognizer_create()
        self.face_recogniser.read('trainer.yml')
        self.face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.loggedInDB = self.loadDict()
        self.islogin = isLogin

    def addFaceToDB(self, face_id, face_name, filePath):
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

                print(str(int(100 * (count / 100))) + " %")
                count += 1
            ret, img = cap.read()

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cap.release()
        self.trainRecognizer()
        return

    def trainRecognizer(self):
        # Path for face image database
        path = 'dataset'

        recognizer_1_8 = cv2.face.LBPHFaceRecognizer_create()
        # recognizer_2_8 = cv2.face.LBPHFaceRecognizer_create(radius=2)
        # recognizer_2_9 = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=9)

        detector = self.face_detector

        # function to get the images and label data
        def getImagesAndLabels(path):

            folderPaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            for folderPath in folderPaths:
                if ".DS_Store" in folderPath:
                    continue
                last_folder = [os.path.join(folderPath, f) for f in os.listdir(folderPath)][0]
                imagePaths = [os.path.join(last_folder, f) for f in os.listdir(last_folder)]
                id = int(os.path.split(folderPath)[-1].split("-")[-1])

                for imagePath in imagePaths:
                    if ".DS_Store" in imagePath:
                        continue
                    PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                    img_numpy = np.array(PIL_img, 'uint8')
                    faces = detector.detectMultiScale(img_numpy)

                    for (x, y, w, h) in faces:
                        faceSamples.append(img_numpy[y:y + h, x:x + w])
                        ids.append(id)

            return faceSamples, ids

        print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer_1_8.train(faces, np.array(ids))

        recognizer_1_8.write('trainer.yml')
        # Updating the face recognition model
        self.face_recogniser = cv2.face.LBPHFaceRecognizer_create().read('trainer.yml')
        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        return

    @staticmethod
    def saveDict(d):
        with open('loggedInDB.pkl', 'wb') as fp:
            pickle.dump(d, fp)

    @staticmethod
    def loadDict():
        with open('loggedInDB.pkl', 'rb') as fp:
            return pickle.load(fp)

    def isIloggedIn(self, i):
        return self.loggedInDB[i]

    def recognise(self, cameraIP):
        if cameraIP == "0":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(cameraIP)
        cap.set(cv2.CAP_PROP_FPS, 5)  # setting the frame rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        while True:
            ret, frame = cap.read()
            frame = np.ndarray.tobytes(frame)
            if ret:
                self.processImage(image=frame)

    def detectFace(self, image):
        frame = np.frombuffer(image, dtype=np.uint8).reshape(IMAGE_SHAPE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
        )

    def logUserI(self, id):
        self.loggedInDB[id] = self.islogin
        self.saveDict(self.loggedInDB)

    def processImage(self, image):
        faces = self.detectFace(image)
        for (x, y, w, h) in faces:
            frame = np.frombuffer(image, dtype=np.uint8).reshape(IMAGE_SHAPE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            id, confidence = self.face_recogniser.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                if self.loggedInDB[id] != self.islogin:
                    self.loggedInDB[id] = self.islogin
                    print("user with id:" + str(id) + " logged in")
