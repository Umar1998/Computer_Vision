import os
import torchvision
from utils import object_detection_api


# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference

def load_model():
    """
    load faster rcnn model with resnet 50 backbone 
    from torchvision

    returns model object ready for inference
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    model.eval()
    return model


def main():
    data_path = './faster-rcnn/data'
    out_path_base = './faster-rcnn/outputs'
    if not os.path.exists(out_path_base):
        os.mkdir(out_path_base)

    model = load_model()
    images = os.listdir(data_path)

    for each_image in images:
        image_path = os.path.join(data_path, each_image)
        out_path = os.path.join(out_path_base, each_image)
        object_detection_api(image_path, model, out_path)


if __name__ == '__main__':
    main()
