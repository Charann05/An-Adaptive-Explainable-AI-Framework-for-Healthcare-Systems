import os
import constants
from pathlib import Path


def clean_package(line):
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    for sep in ["==", ">=", "<=", "~=", "<", ">"]:
        if sep in line:
            line = line.split(sep)[0]
    return line

def scan_requirements(MODEL_FOLDER, evidences, file_info):
    print("Scannig requirement files...")
    for file in file_info:

        if file["name"] == "requirements.txt" and file["extension"] == ".txt":

            requirements = Path(MODEL_FOLDER) / file["name"]

            if not os.path.exists(requirements):
                return

            with open(requirements, "r", encoding="utf-8") as lines:

                for line in lines:
                    package = clean_package(line)

                    if package in constants.FRAMEWORK_MAP:
                        value = constants.FRAMEWORK_MAP[package]
                        evidences["framework"].append({
                                                        "value" : value,
                                                        "source" : "requirements",
                                                        "weight" : constants.SOURCE_WEIGHTS["requirements"]
                                                    })