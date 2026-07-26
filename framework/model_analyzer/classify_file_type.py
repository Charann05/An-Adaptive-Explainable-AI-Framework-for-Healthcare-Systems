import os
import magic
from load_model_files import files, MODEL_FOLDER

file_info = []

for file in files:
    file_path = os.path.join(MODEL_FOLDER, file)

    info = {
        "name" : file,
        "extention" : os.path.splitext(file)[1],
        "magic-type" : magic.from_file(file_path)
    }

    file_info.append(info)

print("-" *100)
for item in file_info:
    print(item)
print("-" *100)