import grpc
import faceRecognition_pb2_grpc
import cv2
import numpy as np
from faceRecognition_pb2 import FaceRecognitionProcess, FaceRecognitionResponse, FaceIDResponse
from tensorflow import keras
import tensorflow as tf


class RecognitionService(faceRecognition_pb2_grpc.RecognitionsServicer):

    def __init__(self):
        self.face_ID_model1 = keras.models.load_model("recognitions/face_ID_model1.h5")
        self.face_ID_model2 = keras.models.load_model("recognitions/face_ID_model2.h5")
        self.face_ID_model3 = keras.models.load_model("recognitions/face_ID_model3.h5")
        self.contains_face_model = keras.models.load_model("recognitions/contains_Face_Model.h5")
        self.loggedInDB = {}
        for i in range(26):
            self.loggedInDB[i + 1] = False

    def preprocessID(self, image):
        resized_image = tf.image.resize(image, [299, 299])
        final_image = keras.applications.xception.preprocess_input(resized_image)
        return np.array([final_image, ])

    def isIloggedIn(self, i):
        return self.loggedInDB[i]

    def recognise(self, request, context):
        import ipaddress
        try:
            ip = ipaddress.ip_address(request.cameraIP)
        except ValueError:
            context.set_code(1)
            context.abort(grpc.StatusCode.NOT_FOUND, "Bad IP address")

        cap = cv2.VideoCapture(request.cameraIP)
        cap.set(cv2.CAP_PROP_FPS, 5)
        while True:
            ret, frame = cap.read()
            if ret:
                response = self.processImage(FaceRecognitionProcess(image=np.array(frame)))
                if response.containsFace:
                    self.loggedInDB[response.faceID] = True

    def processImage(self, request, context):
        predContainsFace = self.contains_face_model.predict(request)
        if predContainsFace > 0.5:
            responses = [self.getFaceID1(request, context).faceiD, self.getFaceID2(request, context).faceID,
                         self.getFaceID3(request, context).faceID]
            resDict = {}
            for r in responses:
                if r in resDict:
                    resDict[r] += 1
                else:
                    resDict[r] = 1
            return FaceRecognitionResponse(faceID=max(resDict, key=resDict.get)
                                           , containsFace=True)
        else:
            return FaceRecognitionResponse(containsFace=False)

    def getFaceID1(self, request, context):
        predID = self.face_ID_model1.predict(self.preprocessID(request.image))
        return FaceIDResponse(faceID=np.argmax(predID))

    def getFaceID2(self, request, context):
        predID = self.face_ID_model2.predict(self.preprocessID(request.image))
        return FaceIDResponse(faceID=np.argmax(predID))

    def getFaceID3(self, request, context):
        predID = self.face_ID_model3.predict(self.preprocessID(request.image))
        return FaceIDResponse(faceID=np.argmax(predID))
