import torch
import torchvision
import ReferentialGym as RG

def test_ExtractorResNet18():
    model = RG.networks.ExtractorResNet18(input_shape=[3, 64, 64], final_layer_idx=3, pretrained=True)

    inputs = torch.zeros(32, 3, 64, 64)
    outputs = model(inputs)

    print(outputs.shape)

    assert(outputs.shape[-1] == 4)


if __name__ == "__main__":
    test_ExtractorResNet18()