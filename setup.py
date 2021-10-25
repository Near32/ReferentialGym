from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

test_requirements = ['pytest']

setup(
    name='ReferentialGym',
    version='0.0.1',
    description='Framework to study the emergence of artificial languages using deep learning. Developed by Kevin Denamganaï at the University of York.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Near32/ReferentialGym',
    author='Kevin Denamganaï',
    author_email='denamganai.kevin@gmail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Deep Learning :: Language Emergence / Grounding',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=[
	'tqdm',
        #'Cython',
	'numpy',
	'scipy',
	'scikit-image',
        # previously: 'scikit-learn==0.23',
        # now:
        'scikit-learn==1.0',
	'h5py',
	'opencv-python',
	#'torch==1.4',
        #'torchvision==0.5.0',
	'torch>=1.4',
	'torchvision>=0.5.0',
        'tensorboardX',
	'matplotlib',
	'docopt',
	'pycocotools',
        'pybullet==3.1.7',
	] + test_requirements,

    python_requires=">=3.6",
)
