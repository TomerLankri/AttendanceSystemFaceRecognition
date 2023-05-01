import threading

from flask import Flask, request
from flask_cors import CORS

from recognitions.recognitionService import RecognitionService
from utils import save_video_filestorage, getTable

app = Flask(__name__)
CORS(app, origins='http://localhost:3000', methods=['POST', 'GET'], allow_headers=['Content-Type'])


@app.route('/addFace', methods=['POST'])
def add_face():
    # Check if 'video' file is present in the request
    if 'video' not in request.files:
        return "No video file found in the request", 400
    video_file = request.files['video']
    output_filename = save_video_filestorage(video_file, "/Users/tomer/PycharmProjects/a/recognitions/videos")
    # Check if the file has a valid file extension
    if not video_file or video_file.content_type not in {'webm', 'video/webm'}:
        return "Invalid file format. Allowed formats are: 'webm', 'video/webm'", 400

    s = threading.Thread(target=app.config['recogniser'].addFaceToDB,
                         args=(request.form["id"], request.form["name"], output_filename))
    s.start()
    return "Video file uploaded successfully", 200


@app.route('/getImageTable', methods=['GET'])
def getImageTable():
    return getTable()


@app.route('/initAdmissionService', methods=['POST'])
def initAdmissionService(camera_ip):
    app.config['recogniser'].recognise(camera_ip)
    t = threading.Thread(target=app.config['recogniser'].recognise, args=(camera_ip))
    t.start()
    # Return a 200 response immediately
    return 'API response', 200


if __name__ == '__main__':
    app.config['recogniser'] = RecognitionService()
    app.run(port=3001)
