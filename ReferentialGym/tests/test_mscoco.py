import torch
import torchvision
import torchvision.transforms as transforms
import ReferentialGym as RG

from pycocotools.coco import COCO

def test_mscoco_api():
    #initialization for instances:
    dataDir = '../../examples'
    dataYear = '2014'
    dataType = 'train'
    annFile = '{}/datasets/MSCOCO{}/{}_ann/annotations/instances_{}{}.json'.format(dataDir, dataYear, dataType, dataType, dataYear)
    dataset = COCO(annFile)

def test_mscoco():
    transform = transforms.ToTensor()
    
    #initialization for instances:
    dataDir = '../../examples'
    dataYear = '2014'
    dataType = 'val'
    annFile = '{}/datasets/MSCOCO{}/{}_ann/annotations/instances_{}{}.json'.format(dataDir, dataYear, dataType, dataType, dataYear)
    
    root = '{}/datasets/MSCOCO{}/{}_imgs/'.format(dataDir, dataYear, dataType)
    
    dataset = RG.datasets.MSCOCODataset(root=root, annFile=annFile, transform=transform)

    output = dataset[0]

    assert(len(output) == 3)
    print(output[1])
    

if __name__ == "__main__":
    test_mscoco()
    