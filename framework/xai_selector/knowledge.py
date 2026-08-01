XAI_TECHNIQUES = {

    "GradCAM": {
        "models": ["CNN"],
        "inputs": ["Image"],
        "description": "Best suited for CNN image models."
    },

    "TreeInterpreter": {
        "models": ["DecisionTree", "RandomForest"],
        "inputs": ["Tabular"],
        "description": "Explains individual predictions of tree models."
    },

    "FeatureImportance": {
        "models": ["DecisionTree", "RandomForest", "XGBoost"],
        "inputs": ["Tabular"],
        "description": "Provides global feature ranking."
    },

    "SHAP": {
        "models": [
            "CNN",
            "DecisionTree",
            "RandomForest",
            "XGBoost",
            "Linear",
            "Logistic",
            "Unknown"
        ],
        "inputs": [
            "Image",
            "Tabular",
            "Text"
        ],
        "description": "General feature attribution method."
    },

    "LIME": {
        "models": ["Any"],
        "inputs": [
            "Image",
            "Tabular",
            "Text"
        ],
        "description": "Model-agnostic local explanation."
    }

}


CATEGORY_PRIORITY = {

    "CNN": [
        "GradCAM",
        "SHAP",
        "LIME"
    ],

    "RandomForest": [
        "TreeInterpreter",
        "SHAP",
        "FeatureImportance",
        "LIME"
    ],

    "DecisionTree": [
        "TreeInterpreter",
        "FeatureImportance",
        "SHAP",
        "LIME"
    ],

    "XGBoost": [
        "SHAP",
        "FeatureImportance",
        "LIME"
    ],

    "Linear": [
        "SHAP",
        "LIME"
    ],

    "Logistic": [
        "SHAP",
        "LIME"
    ],

    "Unknown": [
        "LIME",
        "SHAP"
    ]
}