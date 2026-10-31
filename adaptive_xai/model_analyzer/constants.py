
SOURCE_WEIGHTS = {
    "config" : 10,
    "dataset": 9,
    "python" : 8,
    "requirements" : 6,
    "readme" : 5
}

FRAMEWORK_MAP = {
    "torch": "PyTorch",
    "torchvision": "PyTorch",
    "tensorflow": "TensorFlow",
    "keras": "Keras",

    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "xgboost": "XGBoost",

    "timm": "PyTorch",
    "transformers": "HuggingFace/PyTorch"
}

ARCHITECTURE_MAP = {
    "resnet18": "ResNet-18",
    "resnet34": "ResNet-34",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",

    "densenet121": "DenseNet-121",
    "densenet169": "DenseNet-169",
    "densenet201": "DenseNet-201",
    "densenet264": "DenseNet-264",

    "efficientnet": "EfficientNet",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b1": "EfficientNet-B1",
    "efficientnet_b2": "EfficientNet-B2",
    "efficientnet_b3": "EfficientNet-B3",
    "efficientnet_b4": "EfficientNet-B4",
    "efficientnet_b5": "EfficientNet-B5",
    "efficientnet_b6": "EfficientNet-B6",
    "efficientnet_b7": "EfficientNet-B7",

    "efficientnetv2_s": "EfficientNetV2-S",
    "efficientnetv2_m": "EfficientNetV2-M",
    "efficientnetv2_l": "EfficientNetV2-L",

    "vgg": "VGG",
    "vgg16": "VGG16",
    "vgg19": "VGG19",
    "vgg16_bn": "VGG16-BN",
    "vgg19_bn": "VGG19-BN",
    "keras.applications.vgg16": "VGG16",
    "keras.applications.vgg19": "VGG19",
    "torchvision.models.vgg16": "VGG16",
    "torchvision.models.vgg19": "VGG19",
    "torchvision.models.vgg16_bn": "VGG16-BN",
    "torchvision.models.vgg19_bn": "VGG19-BN",

    "xception": "Xception",
    "keras.applications.xception": "Xception",
    "tf.keras.applications.xception": "Xception",

    "convnext": "ConvNeXt",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "convnext_base": "ConvNeXt-Base",
    "convnext_large": "ConvNeXt-Large",

    "convnextv2": "ConvNeXtV2",
    "convnextv2_tiny": "ConvNeXtV2-Tiny",
    "convnextv2_base": "ConvNeXtV2-Base",
    "convnextv2_large": "ConvNeXtV2-Large",

    "lstm": "LSTM",
    "torch.nn.lstm": "LSTM",
    "nn.lstm": "LSTM",
    "keras.layers.lstm": "LSTM",
    "tf.keras.layers.lstm": "LSTM",

    "gru": "GRU",
    "torch.nn.gru": "GRU",
    "nn.gru": "GRU",
    "keras.layers.gru": "GRU",
    "tf.keras.layers.gru": "GRU",

    "bertmodel": "BERT",
    "berttokenizer": "BERT",
    "bertconfig": "BERT",
    "bert": "BERT",

    "biobert": "BioBERT",
    "clinicalbert": "ClinicalBERT",
    "pubmedbert": "PubMedBERT",
    "bluebert": "BlueBERT",
    "scibert": "SciBERT",

    "xgboost": "XGBoost",
    "xgboost.sklearn": "XGBoost",
    "xgbclassifier": "XGBClassifier",
    "xgbregressor": "XGBRegressor",
    "xgbranker": "XGBRanker",
    "xgbmodel": "XGBModel",

    "decisiontree": "Decision Tree",
    "decision tree": "Decision Tree",
    "decisiontreeclassifier": "DecisionTreeClassifier",
    "decisiontreeregressor": "DecisionTreeRegressor",

    "randomforest": "Random Forest",
    "random forest": "Random Forest",
    "randomforestclassifier": "RandomForestClassifier",
    "randomforestregressor": "RandomForestRegressor",

    "logisticregression": "Logistic Regression",
    "logistic regression": "Logistic Regression",
    "logit": "Logistic Regression",

    "linearregression": "Linear Regression",
    "linear regression": "Linear Regression",
    "ols": "Linear Regression"
}

CLASS_PARAMETER_NAMES = {
    "num_classes",
    "n_classes",
    "classes",
    "num_labels",
    "n_labels",
}

OPTIMIZER_MAP = {
    "adam": "Adam",
    "adamw": "AdamW",
    "sgd": "SGD",
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
    "adamax": "Adamax",
    "nadam": "NAdam",
    "lbfgs": "LBFGS"
}

LOSS_FUNCTION_MAP = {
    "crossentropyloss": "CrossEntropyLoss",
    "cross_entropy": "CrossEntropyLoss",
    "bcewithlogitsloss": "BCEWithLogitsLoss",
    "bceloss": "BCELoss",
    "binary_crossentropy": "BinaryCrossentropy",
    "categoricalcrossentropy": "CategoricalCrossentropy",
    "categorical_crossentropy": "CategoricalCrossentropy",
    "sparsecategoricalcrossentropy": "SparseCategoricalCrossentropy",
    "mseloss": "MSELoss",
    "mse_loss": "MSELoss",
    "l1loss": "L1Loss",
    "smoothl1loss": "SmoothL1Loss",
    "nllloss": "NLLLoss",
    "ctcloss": "CTCLoss"
}

OPTIMIZER_PARAMETER_NAMES = {
    "optimizer",
}

LOSS_PARAMETER_NAMES = {
    "loss",
    "loss_fn",
    "loss_function",
    "criterion"
}

INPUT_PATTERNS = {
    "imagefolder": "Image",
    "image.open": "Image",
    "pil.image.open": "Image",
    "cv2.imread": "Image",
    "transforms": "Image",
    "albumentations": "Image",
    "imagedatagenerator": "Image",
    "image_dataset_from_directory": "Image",

    "tokenizer": "Text",
    "berttokenizer": "Text",
    "autotokenizer": "Text",

    "read_csv": "Tabular",
    "read_excel": "Tabular",

    "nib.load": "3D Medical Image",
    "pydicom.dcmread": "DICOM Image"
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

TABULAR_EXTENSIONS = {
    ".csv",
    ".xls",
    ".xlsx"
}

TEXT_EXTENSIONS = {
    ".txt",
    ".json"
}

MEDICAL_EXTENSIONS = {
    ".nii",
    ".nii.gz",
    ".dcm"
}

GENERIC_ARCHITECTURE_MAP = {
    "conv1d": "Custom CNN",
    "conv2d": "Custom CNN",
    "conv3d": "Custom CNN",

    "maxpool1d": "Custom CNN",
    "maxpool2d": "Custom CNN",
    "maxpool3d": "Custom CNN",

    "avgpool2d": "Custom CNN",
    "adaptiveavgpool2d": "Custom CNN",

    "lstm": "Custom RNN",
    "gru": "Custom RNN",
    "simplernn": "Custom RNN",
    "rnn": "Custom RNN",

    "transformer": "Custom Transformer",
    "transformerencoder": "Custom Transformer",
    "transformerdecoder": "Custom Transformer",
    "multiheadattention": "Custom Transformer",

    "gcnconv": "Graph Neural Network",
    "graphconv": "Graph Neural Network",
    "gatconv": "Graph Neural Network",
    "sageconv": "Graph Neural Network"
}

CLASSIFIER_MODELS = {
    "xgbclassifier": "Classification",
    "randomforestclassifier": "Classification",
    "decisiontreeclassifier": "Classification",
    "logisticregression": "Classification",
    "svc": "Classification",
    "gaussiannb": "Classification",
    "kneighborsclassifier": "Classification",
    "mlpclassifier": "Classification"
}

REGRESSOR_MODELS = {
    "xgbregressor": "Regression",
    "randomforestregressor": "Regression",
    "decisiontreeregressor": "Regression",
    "linearregression": "Regression",
    "lasso": "Regression",
    "ridge": "Regression",
    "elasticnet": "Regression",
    "svr": "Regression"
}

CLASSIFICATION_LOSSES = {
    "crossentropyloss",
    "bceloss",
    "bcewithlogitsloss",
    "nllloss",
    "focalloss",
    "binary_crossentropy",
    "binarycrossentropy",
    "sparsecategoricalcrossentropy"
}

REGRESSION_LOSSES = {
    "mseloss",
    "mean_squared_error",
    "meansquarederror",
    "l1loss",
    "smoothl1loss",
    "huberloss"
}

CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc_score",
    "confusion_matrix"
}

REGRESSION_METRICS = {
    "mean_squared_error",
    "mse",
    "mae",
    "rmse",
    "r2_score"
}

CLASSIFICATION_TARGET_NAMES = {
    "label",
    "labels",
    "class",
    "classes",
    "category",
    "categories"
}