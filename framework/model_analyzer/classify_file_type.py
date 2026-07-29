import os
import magic

def get_file_info(MODEL_FOLDER, files):
    file_info = []

    for file in files:
        file_path = os.path.join(MODEL_FOLDER, file)

        info = {
            "name" : file,
            "extension" : os.path.splitext(file)[1],
            "magic-type" : magic.from_file(file_path)
        }

        file_info.append(info)
    return file_info

    # print("-" *100)
    # for item in file_info:
    #     print(item)
    # print("-" *100)