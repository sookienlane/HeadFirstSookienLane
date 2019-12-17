import torchvision
import torch.nn as nn

def get_model(model_name):
    if model_name == "mobilenet":
        mobilenet_raw = torchvision.models.mobilenet_v2()
        mobilenet_raw.classifier[-1] = nn.Linear(in_features=mobilenet_raw.classifier[-1].in_features,out_features=200)
        model = mobilenet_raw
    elif model_name == "densenet":
        densenet_raw = torchvision.models.densenet201()
        densenet_raw.classifier = nn.Linear(in_features=densenet_raw.classifier.in_features,out_features=200)
        model = densenet_raw

    return model