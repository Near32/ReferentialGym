# ReferentialGym: Framework for Language Emergence/Grounding using Referential Games

This framework provides out-of-the-box implementations of Referential Games variants in order to study the emergence of artificial languages using deep learning, relying on [PyTorch](https://www.pytorch.org). 
This framework has been in constant development since July 2019.
The following paper details its main features:


> [ReferentialGym: A Nomenclature and Framework for Language Emergence & Grounding in (Visual) Referential Games](https://arxiv.org/abs/2012.09486) (or [workshop link 2](https://drive.google.com/file/d/1p6rgNcvtug7UuVHrEgDIZNI6nYILkqZR/view?usp=sharing))
> Kevin Denamganaï and James Alfred Walker.
> [4th NeurIPS Workshop on Emergent Communication: "Talking with Strangers: Zero-Shot Emergent Communication", 2020](https://sites.google.com/view/emecom2020).

## Features

<!--
+ PyTorch implementation of: [A2C](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752),[REINFORCE](https://danielhp95.github.io/policy-gradient-algorithms-a-review),[PPO](https://arxiv.org/abs/1707.06347)...
-->
+ Provides an interface for dataset to be used in the context of referential games. 
<!--
See [Adding a new dataset](docs/adding-a-new-dataset.md).
-->
+ Provides state-of-the-art language emergence algorithms based on referential game variants that can be configured at will by the users.
+ Provides common implementations of various metrics, e.g. topographic similarity as a [compositionality metric](https://arxiv.org/abs/1804.03984), [causal influence of communication](https://arxiv.org/abs/1903.05168) metric, [FactorVAE](https://arxiv.org/abs/1802.05983)'s disentanglement metric ...

## Documentation 

Tutorials: 
* Getting Started: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Near32/ReferentialGym/blob/master/examples/tutorial-getting-started.ipynb) Learn how to use the framework's features out-of-the-box with different agent architectures and referential game variants.
* Creating New Modules:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Near32/ReferentialGym/blob/master/examples/tutorial-creating-new-modules.ipynb) Learn how to create new modules either as part of the agents' architecture or as a new metric.

All relevant documentation can be found [here](https://near32.github.io/ReferentialGym/html/index.html) and in the above-mentioned paper. Refer to source code for more specific documentation.

## Installation

### Using `pip` (**SOON**)

This project has not yet been uploaded to PyPi.

<!--
This project can be found in [PyPi](LINK TO PYPI project) (Python Package Index). It can be installed via
`pip`:

`pip install referentialgym`
-->

### Installing from source

Clone this repository:

```
git clone https://github.com/Near32/ReferentialGym
```

And, install it locally:
```
cd ReferentialGym/
pip install -e .
```

### Dependencies

This package enforces Python version `3.6` or higher. 
Python dependencies are listed in the file [`setup.py`](./setup.py). 

### License

Read [License](LICENSE).
