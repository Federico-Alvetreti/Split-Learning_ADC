import datasets
import timm
import torchvision

# DATASETS
torchvision.datasets.CIFAR100('./data', train=True)
torchvision.datasets.CIFAR100('./data', train=False)

torchvision.datasets.Food101('./data', split='train')
torchvision.datasets.Food101('./data', split='test')

# MODELS
timm.create_model(model_name='deit_small_patch16_224.fb_in1k', pretrained=True)
timm.create_model(model_name='deit_tiny_patch16_224.fb_in1k', pretrained=True)
