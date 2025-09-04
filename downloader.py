import datasets
import timm
import torchvision

# DATASETS
torchvision.datasets.CIFAR100('./data', train=True, download=True)
torchvision.datasets.CIFAR100('./data', train=False, download=True)

torchvision.datasets.Food101('./data', split='train', download=True)
torchvision.datasets.Food101('./data', split='test', download=True)

torchvision.datasets.Imagenette('./data', split='train', download=True, size='full')
torchvision.datasets.Imagenette('./data', split='test', download=True, size='full')

# MODELS
timm.create_model(model_name='deit_small_patch16_224.fb_in1k', pretrained=True)
timm.create_model(model_name='deit_tiny_patch16_224.fb_in1k', pretrained=True)
