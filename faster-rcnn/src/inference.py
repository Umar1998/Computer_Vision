import os
import torchvision
import argparse
from utils import object_detection_api


def load_model():
    """
    load faster rcnn model with resnet 50 backbone 
    from torchvision

    get the pretrained model from torchvision.models

    returns model object ready for inference
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)
    model.eval()  # model.eval() to use the model for inference
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./faster-rcnn/data')
    parser.add_argument('--out-path-base', type=str,
                        default='./faster-rcnn/outputs')
    args = parser.parse_args()

    data_path = args.data_path
    out_path_base = args.out_path_base

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
