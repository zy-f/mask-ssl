from models import resnet
ARCHITECTURES = {
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50
}