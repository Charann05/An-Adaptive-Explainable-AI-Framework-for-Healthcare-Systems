import os
from pathlib import Path
from classify_file_type import get_file_info

def load_files():
    print("Loading file...")
    MODEL_FOLDER = "../source_model"

    files = os.listdir(MODEL_FOLDER)

    # print("=" *50)
    print("\nFiles Found")
    # print("=" *50)

    # for file in files:
    #     print(file)
    # print(Path(MODEL_FOLDER))
    # get_file_info(MODEL_FOLDER, files)
    return Path(MODEL_FOLDER), files
# load_files()
