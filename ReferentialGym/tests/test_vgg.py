import torch
import torchvision
import ReferentialGym as RG

def test_vgg():
    model = torchvision.models.vgg.vgg16(pretrained=True, progress=True)

    inputs = torch.zeros(32, 3, 256, 256)
    outputs = model(inputs)

    assert(outputs.shape[-1] == 4096)

def test_ModelVGG16():
    model = RG.networks.ModelVGG16(input_shape=[3, 256, 256], feature_dim=256, pretrained=True)

    inputs = torch.zeros(32, 3, 256, 256)
    outputs = model(inputs)

    assert(outputs.shape[-1] == 256)


def test_one_channel_ModelVGG16():
    model = RG.networks.ModelVGG16(input_shape=[1, 256, 256], feature_dim=256, pretrained=True)

    inputs = torch.zeros(32, 1, 256, 256)
    outputs = model(inputs)

    assert(outputs.shape[-1] == 256)

if __name__ == "__main__":
    test_vgg()
    test_ModelVGG16()