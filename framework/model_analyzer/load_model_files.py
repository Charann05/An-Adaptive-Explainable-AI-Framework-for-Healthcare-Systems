import os

MODEL_FOLDER = "../source_model"

files = os.listdir(MODEL_FOLDER)

print("=" *50)
print("Files Found")
print("=" *50)

for file in files:
    print(file)
