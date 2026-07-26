from pathlib import Path
from analyzer import evidences
from constants import SOURCE_WEIGHTS

REQUIREMENTS_MAP = {
    "torch" : ("framework", "PyTorch"),
    "torchvision" : ("framework", "PyTorch"),
    "tensorflow" : ("framework", "TensorFlow"),
    "keras" : ("framework", "Keras"),

    "scikit-learn" : ("framework", "Scikit-Learn"),
    "sklearn" : ("framework", "Scikit-Learn"),
    "xgboost" : ("framework", "XGBoost"),

    "tim" : ("framework", "PyTorch"),
    "transformers" : ("framework", "HuggingFace/Pytorch"),

}

def clean_package(line):
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    for sep in ["==", ">=", "<=", "~=", "<", ">"]:
        if sep in line:
            line = line.split(sep)[0]
    return line

requirements = Path("requirements.txt")

with requirements.open() as content:
    lines = content.readlines()

    for line in lines:
        package = clean_package(line)

        category, value = REQUIREMENTS_MAP(package)
        evidences[category].append({
                                        "value" : value,
                                        "source" : "requirements",
                                        "weight" : SOURCE_WEIGHTS("requirements")
                                    })