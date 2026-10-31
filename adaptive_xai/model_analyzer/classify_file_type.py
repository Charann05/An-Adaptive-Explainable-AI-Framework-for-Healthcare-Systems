import os

def get_file_info(MODEL_FOLDER, files):
    file_info = []

    for file in files:
        file_path = os.path.join(MODEL_FOLDER, file)

        info = {
            "name": file,
            "extension": os.path.splitext(file)[1].lower()
        }

        file_info.append(info)

    return file_info