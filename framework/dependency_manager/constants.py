XAI_DEPENDENCIES = {
    "TEST": [
        {
            "pip": "emoji",
            "import": "emoji"
        }
    ],

    "SHAP": [
        {
            "pip": "shap",
            "import": "shap"
        }
    ],

    "LIME": [
        {
            "pip": "lime",
            "import": "lime"
        }
    ],
    "TreeInterpreter": [
        {
            "pip": "treeinterpreter",
              "import": 
              "treeinterpreter"
        }
    ],
    
    "FeatureImportance": [],

    "GradCAM": [
        {
            "pip": "grad-cam",
            "import": "pytorch_grad_cam"
        }
    ]
}


MODEL_DEPENDENCIES = {
    "pytorch": [
        {
            "pip": "torch",
            "import": "torch",
        },
        {
            "pip": "timm",
            "import": "timm",
        },
    ],
    "tensorflow": [
        {
            "pip": "tensorflow",
            "import": "tensorflow",
        }
    ],
    "keras": [
        {
            "pip": "tensorflow",
            "import": "tensorflow",
        }
    ],
    "sklearn": [
        {
            "pip": "scikit-learn",
            "import": "sklearn",
        },
        {
            "pip": "joblib",
            "import": "joblib",
        },
    ],
    "scikit-learn": [
        {
            "pip": "scikit-learn",
            "import": "sklearn",
        },
        {
            "pip": "joblib",
            "import": "joblib",
        },
    ],
    "xgboost": [
        {
            "pip": "xgboost",
            "import": "xgboost",
        }
    ],
}