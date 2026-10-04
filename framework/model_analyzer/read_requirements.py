import os
from . import constants
from pathlib import Path
from .. import cli_display


def clean_package(line):
    line = line.strip()

    if not line or line.startswith("#"):
        return None

    for sep in ["==", ">=", "<=", "~=", "<", ">"]:
        if sep in line:
            line = line.split(sep)[0]
    return line

def scan_requirements(MODEL_FOLDER, evidences, file_info):
    found = False

    for file in file_info:
        if file["name"] == "requirements.txt" and file["extension"] == ".txt":
            requirements = Path(MODEL_FOLDER) / file["name"]

            if not os.path.exists(requirements):
                break

            cli_display.print_status("Scanning requirements.txt")
            found = True

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
            break

    if not found:
        cli_display.print_error("No requirements file was found")