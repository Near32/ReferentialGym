from .dataset import Dataset
from .labeled_dataset import LabeledDataset

from .CLEVR_dataset import CLEVRDataset
from .sort_of_CLEVR_dataset import SortOfCLEVRDataset

try:
	import minerl
	from .MineRL_dataset import MineRLDataset
except Exception as e:
	print(f"During importation of MineRLDataset:{e}")
	print("Please install minerl if you want to use the MineRLDataset.")
 
from .dSprites_dataset import dSpritesDataset
from .MSCOCO_dataset import MSCOCODataset 

from .utils import collate_dict_wrapper, ResizeNormalize, RescaleNormalize
