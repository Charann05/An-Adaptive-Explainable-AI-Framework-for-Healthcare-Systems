ARCHITECTURE_TO_CATEGORY = {
    "resnet": "CNN", "efficientnet": "CNN", "densenet": "CNN",
    "vgg": "CNN", "mobilenet": "CNN", "inception": "CNN",
    "randomforest": "RandomForest",
    "decisiontree": "DecisionTree",
    "xgb": "XGBoost", "xgboost": "XGBoost",
    "linearregression": "Linear",
    "logisticregression": "Logistic",
}

def normalize_category(raw_category: str, raw_architecture: str = None) -> str:
    if raw_architecture:
        key = raw_architecture.strip().lower()
        for arch_prefix, category in ARCHITECTURE_TO_CATEGORY.items():
            if arch_prefix in key:          # substring match, not exact
                return category

    if raw_category:
        key = raw_category.strip().lower()
        for arch_prefix, category in ARCHITECTURE_TO_CATEGORY.items():
            if arch_prefix in key:
                return category

    return "Unknown"