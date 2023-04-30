from flask import Flask, request
from flask_cors import CORS

from addFaceToDB import addFaceToDB
from getImageTable import getTable
from recognitionService import RecognitionService

app = Flask(__name__)
CORS(app, origins='http://localhost:3000', methods=['POST', 'GET'], allow_headers=['Content-Type'])

from werkzeug.utils import secure_filename
import os
import datetime


def save_video_filestorage(file_storage, output_dir):
    # Save the FileStorage object to a temporary file
    timeString = datetime.datetime.now().strftime('%M-%H-%d-%m-%Y')
    filename = secure_filename(file_storage.filename + timeString) + ".webm"
    tmp_filename = f'/tmp/{filename}.webm'
    file_storage.save(tmp_filename)

    # Set output file path for video
    output_filename = os.path.join(output_dir, filename)
    # Move the temporary input video file to the output directory
    try:
        os.rename(tmp_filename, output_filename)
    except OSError as e:
        print(f'Error: Failed to save video file. {e}')
        return None

    return output_filename


@app.route('/addFace', methods=['POST'])
def add_face():
    # Check if 'video' file is present in the request
    if 'video' not in request.files:
        return "No video file found in the request", 400
    video_file = request.files['video']
    output_filename = save_video_filestorage(video_file, "/Users/tomer/PycharmProjects/a/recognitions/videos")
    # Check if the file has a valid file extension
    allowed_extensions = {'mp4', 'avi', 'mov', 'webm', 'video/webm'}
    if not video_file or video_file.content_type not in allowed_extensions:
        return "Invalid file format. Allowed formats are: mp4, avi, mov", 400
    print("output Filename = " + output_filename)
    addFaceToDB(request.form["id"], request.form["name"], output_filename)
    return "Video file uploaded successfully", 200


@app.route('/getImageTable', methods=['GET'])
def getImageTable():
    return getTable()


if __name__ == '__main__':
    recogniser = RecognitionService()
    app.run(port=3001)
