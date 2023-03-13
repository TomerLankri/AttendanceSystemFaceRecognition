import grpc
import numpy

import faceRecognition_pb2_grpc
import cv2
import numpy as np
from faceRecognition_pb2 import FaceRecognitionProcess, FaceRecognitionResponse, FaceDetectionResponse, Empty
import pickle
from PIL import Image
import os
from numproto import ndarray_to_proto, proto_to_ndarray

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
IMAGE_SHAPE = (480, 640, 3)


class RecognitionService(faceRecognition_pb2_grpc.RecognitionsServicer):

    def __init__(self, isLogin=True):
        self.face_recogniser = cv2.face.LBPHFaceRecognizer_create()
        self.face_recogniser.read('trainer.yml')
        self.face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.loggedInDB = self.loadDict()
        self.islogin = isLogin

    def test(self):
        print("test successful")

    def addFace(self, request, context):

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)  # set video width
        cam.set(4, 480)  # set video height

        # For each person, enter one numeric face id
        face_id = input('\n enter user id end press <return> ==>  ')
        face_name = input('\n enter user name end press <return> ==>  ')

        print("\n [INFO] Initializing face capture. Look the camera and move your head around ...")
        # Initialize individual sampling face count
        count = 0

        while True:

            ret, img = cam.read()
            # img = cv2.flip(img, -1)  # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # Save the captured image into the datasets folder
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(face_name) + '.' + str(count) + ".jpg",
                            gray[y:y + h, x:x + w])
                #
                # cv2.imshow('image', img)
                print(str(int(100 * (count / 60))) + " %")
            k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 60:  # Take 30 face sample and stop video
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        # cv2.destroyAllWindows()
        # Retrain model
        self.trainRecognizer(Empty(), context)
        return Empty()

    def trainRecognizer(self, request, context):
        # Path for face image database
        path = 'dataset'

        recognizer_1_8 = cv2.face.LBPHFaceRecognizer_create()
        recognizer_2_8 = cv2.face.LBPHFaceRecognizer_create(radius=2)
        recognizer_2_9 = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=9)

        detector = self.face_detector

        # function to get the images and label data
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []

            for imagePath in imagePaths:
                if ".DS_Store" in imagePath:
                    continue
                PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
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
        return Empty()

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

    def recognise(self, request, context):
        if request.cameraIP == "0":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(request.cameraIP)
        cap.set(cv2.CAP_PROP_FPS, 5)  # setting the frame rate
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        while True:
            ret, frame = cap.read()
            frame = np.ndarray.tobytes(frame)
            if ret:
                self.processImage(FaceRecognitionProcess(image=frame),context)

    def detectFace(self, request, context):
        frame = np.frombuffer(request.image,dtype=np.uint8).reshape(IMAGE_SHAPE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
        )

    def logUserI(self, id):
        self.loggedInDB[id] = self.islogin
        self.saveDict(self.loggedInDB)

    def processImage(self, request, context):
        faces = self.detectFace(request, context)
        for (x, y, w, h) in faces:
            frame = np.frombuffer(request.image, dtype=np.uint8).reshape(IMAGE_SHAPE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            id, confidence = self.face_recogniser.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                if self.loggedInDB != self.islogin:
                    self.loggedInDB[id] = self.islogin
                    print("user with id:"+str(id)+" logged in")
