ARCHITECTURE_TO_CATEGORY = {
    "cnn": "CNN",              
    "resnet": "CNN", "efficientnet": "CNN", "densenet": "CNN",
    "vgg": "CNN", "mobilenet": "CNN", "inception": "CNN",
    "randomforest": "RandomForest",
    "decisiontree": "DecisionTree",
    "xgb": "XGBoost", "xgboost": "XGBoost",
    "linearregression": "Linear",
    "logisticregression": "Logistic",
    # CNN / image models
    "cnn": "CNN",
    "vgg": "CNN",
    "vgg16": "CNN",
    "vgg19": "CNN",
    "resnet": "CNN",
    "resnet18": "CNN",
    "resnet34": "CNN",
    "resnet50": "CNN",
    "resnet101": "CNN",
    "densenet": "CNN",
    "mobilenet": "CNN",
    "efficientnet": "CNN",
    "efficientnet-b0": "CNN",
    "efficientnetb0": "CNN",
    "inception": "CNN",
    "inceptionv3": "CNN",

    # Tree models
    "randomforest": "RandomForest",
    "random forest": "RandomForest",
    "decisiontree": "DecisionTree",
    "decision tree": "DecisionTree",
    "xgboost": "XGBoost",

    # Linear models
    "linear regression": "Linear",
    "linear": "Linear",
    "logistic regression": "Logistic",
    "logistic": "Logistic",
}

def normalize_category(raw_category: str, raw_architecture: str = None) -> str:
    if raw_architecture:
        key = raw_architecture.strip().lower().replace(" ", "").replace("_", "")
        for arch_prefix, category in ARCHITECTURE_TO_CATEGORY.items():
            if arch_prefix in key:
                return category

    if raw_category:
        key = raw_category.strip().lower().replace(" ", "").replace("_", "")
        for arch_prefix, category in ARCHITECTURE_TO_CATEGORY.items():
            if arch_prefix in key:
                return category

    return "Unknown"