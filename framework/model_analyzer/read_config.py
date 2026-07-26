import os
import json
from load_model_files import MODEL_FOLDER
from analyzer import evidences
from constants import SOURCE_WEIGHTS

config_path = os.path.join(MODEL_FOLDER, 'config.js')

if(os.path.exists(config_path)):

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
        evidences["nums_classes"].append({   
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