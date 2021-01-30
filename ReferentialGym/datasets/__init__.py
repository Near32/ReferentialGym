from .dataset import Dataset, shuffle
from .dict_dataset_wrapper import DictDatasetWrapper 
from .labeled_dataset import LabeledDataset
from .dual_labeled_dataset import DualLabeledDataset

from .CLEVR_dataset import CLEVRDataset
from .sort_of_CLEVR_dataset import SortOfCLEVRDataset
from .extended_sort_of_CLEVR_dataset import XSortOfCLEVRDataset
#from .ah_so_CLEVR_dataset import AhSoCLEVRDataset
from .spatial_queries_on_object_tuples_dataset import SQOOTDataset 

try:
	import pybullet 
	from ._3d_shapes_pybullet_dataset import _3DShapesPyBulletDataset 
except Exception as e:
	print(f"During importation of 3DShapesPyBulletDataset:{e}")
	print("Please install pybullet if you want to use the 3DShapesPyBulletDataset.")

try:
	import minerl
	from .MineRL_dataset import MineRLDataset
except Exception as e:
	print(f"During importation of MineRLDataset:{e}")
	print("Please install minerl if you want to use the MineRLDataset.")
 
from .dSprites_dataset import dSpritesDataset
from .MSCOCO_dataset import MSCOCODataset 

from .utils import collate_dict_wrapper, ResizeNormalize, RescaleNormalize
