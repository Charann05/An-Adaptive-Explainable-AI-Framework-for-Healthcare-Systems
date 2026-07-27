
SOURCE_WEIGHTS = {
    "config" : 10,
    "python" : 8,
    "model" : 7,
    "requirements" : 6,
    "readme" : 5
}

FRAMEWORK_MAP = {
    "torch" : ("framework", "PyTorch"),
    "torchvision" : ("framework", "PyTorch"),
    "tensorflow" : ("framework", "TensorFlow"),
    "keras" : ("framework", "Keras"),

    "scikit-learn" : ("framework", "Scikit-Learn"),
    "sklearn" : ("framework", "Scikit-Learn"),
    "xgboost" : ("framework", "XGBoost"),

    "timm" : ("framework", "PyTorch"),
    "transformers" : ("framework", "HuggingFace/Pytorch")
}