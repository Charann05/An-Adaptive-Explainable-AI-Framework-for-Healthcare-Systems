from pathlib import Path
from analyzer import evidences
import constants


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

        if package in constants.FRAMEWORK_MAP:
            category, value = constants.FRAMEWORK_MAP[package]
            evidences[category].append({
                                            "value" : value,
                                            "source" : "requirements",
                                            "weight" : constants.SOURCE_WEIGHTS["requirements"]
                                        })