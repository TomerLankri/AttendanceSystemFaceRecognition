import base64
import datetime
import os

import cv2
from werkzeug.utils import secure_filename


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


def getTable():
    # Initialize a list to store the response
    response_list = []
    dataset_dir = "/Users/tomer/PycharmProjects/a/recognitions/dataset"

    # Traverse the directory structure
    for user_dir in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, user_dir)):
            user_parts = user_dir.split('-')
            if len(user_parts) == 3:
                user_name = user_parts[1]
                user_id = user_parts[2]

                for date_dir in os.listdir(os.path.join(dataset_dir, user_dir)):
                    if os.path.isdir(os.path.join(dataset_dir, user_dir, date_dir)):
                        try:
                            date_obj = datetime.datetime.strptime(date_dir, "%d-%m-%Y")
                            date_str = date_obj.strftime("%Y-%m-%d")
                        except ValueError:
                            continue

                        image_path = os.path.join(dataset_dir, user_dir, date_dir,
                                                  os.listdir(os.path.join(dataset_dir, user_dir, date_dir))[0])

                        # Read the image file and encode it using base64
                        with open(image_path, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode()

                        # Add the user, date, and image data to the response list
                        response_list.append({
                            "name": user_name,
                            "_id": user_id,
                            "date": date_str,
                            "image_data": image_data
                        })

    # Return the JSON response
    res = dict()
    res["data"] = response_list
    return res


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
