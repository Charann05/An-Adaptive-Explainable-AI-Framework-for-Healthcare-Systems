import os
import magic
from load_model_files import files, MODEL_FOLDER

for file in files:
    file_path = os.path.join(MODEL_FOLDER, file)

    print(file)
    print(magic.from_file(file_path))
    print("-" *50)