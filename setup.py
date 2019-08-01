from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

test_requirements = ['tqdm', 'pytest']

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
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Language Emergence / Grounding',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=['pip',
                      'numpy',
                      'gym',
                      'torch',
                      'torchvision',
                      'seaborn',
                      'matplotlib',
                      'docopt'
                      ] + test_requirements,

    python_requires=">=3.6",
)
