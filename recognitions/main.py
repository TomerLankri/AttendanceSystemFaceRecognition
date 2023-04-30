import faceRecognition_pb2_grpc
import concurrent.futures

import grpc

import faceRecognition_pb2_grpc
import recognitionService
from faceRecognition_pb2 import recognitionInit
from faceRecognition_pb2_grpc import RecognitionsStub


def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10), )
    service = recognitionService.RecognitionService(isLogin=True)
    faceRecognition_pb2_grpc.add_RecognitionsServicer_to_server(service, server)
    server.add_insecure_port("[::]:3001")
    print("running")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
    channel = grpc.insecure_channel("localhost:50051", options=(('grpc.enable_http_proxy', 0),))
    client = RecognitionsStub(channel)
    init = recognitionInit(cameraIP="0")
    # client.trainRecognizer(Empty())
    client.recognise(init)

    # while True:
    #     k = input("Press esc to leave, A to add face, R to run the program")
    #     if k == 27:
    #         break
    #     elif k == "A" or k == "a":
    #         client.addFace(Empty())
    #     elif k == "R" or k == "r":
    #         client.recognise(recognitionInit(cameraIP="0"))
    #
