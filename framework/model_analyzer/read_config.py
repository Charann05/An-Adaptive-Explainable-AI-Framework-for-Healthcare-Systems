import os
import json
from .constants import SOURCE_WEIGHTS
from pathlib import Path
from .. import cli_display

def scan_config(MODEL_FOLDER, evidences, file_info):
    found = False

    for file in file_info:

        if file["name"] == "config.json" and file["extension"] == ".json":

            config_path = Path(MODEL_FOLDER) / file["name"]

            if(os.path.exists(config_path)):

                cli_display.print_status("Scanning config.json")
                found = True

                with open(config_path) as content:

                    config = json.load(content)

                    evidences["architecture"].append({
                                                        "value" : config.get("model") or
                                                                config.get("architecture") or
                                                                config.get("network") or
                                                                config.get("model_type") or
                                                                config.get("network") or
                                                                config.get("backbone") or
                                                                config.get("architectures"),
                                                        "source" : "config",
                                                        "weight" : SOURCE_WEIGHTS["config"]
                                                    })
                    evidences["num_classes"].append({   
                                                        "value" : config.get("num_classes") or
                                                                config.get("n_classes") or
                                                                config.get("num_labels") or
                                                                config.get("n_labels") or
                                                                config.get("number_of_classes") or
                                                                config.get("label_count") or
                                                                config.get("output_classes"),
                                                        "source" : "config",
                                                        "weight" : SOURCE_WEIGHTS["config"]
                                                    })      

    if not found:
        cli_display.print_error("No config file was found")    