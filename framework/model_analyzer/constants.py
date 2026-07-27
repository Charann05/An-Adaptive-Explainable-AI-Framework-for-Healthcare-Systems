
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