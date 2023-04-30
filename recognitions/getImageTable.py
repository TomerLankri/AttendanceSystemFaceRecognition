import base64
import datetime
import os


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
