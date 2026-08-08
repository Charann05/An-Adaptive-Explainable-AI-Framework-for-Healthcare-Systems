from pathlib import Path

def load_files():

    print("Loading files...")
    MODEL_FOLDER = Path("framework/source_model")
    files = []

    for file in MODEL_FOLDER.rglob("*"):
        if file.is_file():
            files.append(str(file.relative_to(MODEL_FOLDER)))

    print("Files Found")
    # for file in files:
    #     print(file)

    return MODEL_FOLDER, files