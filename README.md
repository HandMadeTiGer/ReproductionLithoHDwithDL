### Description

This is an end-to-end reproduction of https://github.com/unnir/lithography_hotspot_detection
```
@inproceedings{borisov2018lithography,
  title={Lithography Hotspots Detection Using Deep Learning},
  author={Borisov, Vadim and Scheible, J{\"u}rgen},
  booktitle={2018 15th International Conference on Synthesis, Modeling, Analysis and Simulation Methods and Applications to Circuit Design (SMACD)},
  pages={145--148},
  year={2018},
  organization={IEEE}
}
```

Some hyper parameters are taken from the original paper, which ara:
- learning rate: 0.001
- variance of GaussianBlur: 3.3
- batch size: 32

The other are determine by cross validation and comparison with original paper:
- size of GaussianBlur: 55
- epochs: 15

For conciseness, the OAS files from iccad2019 benchmarks are converted to GDS with kLayout.

The GDS parser 

### Installation

###### Dependencies
- Python3
- torch
- torch-vision
- tqdm
- scikit-learn
- pybind11
- boost
- cairo


##### Install Feature Extraction Package

In the project directory:
```
pip install .
```

### Training and test

```
python train.py {'iccad2012', 'iccad2019-1', 'iccad2019-2'} isPretrain
```

## License Notice

This project includes code and/or assets from the following open source project(s) that are licensed under the MIT License:

- **[Limbo](https://github.com/limbo018/Limbo)** -  An Library for VLSI CAD Design

The MIT License applies to the included portions of this project. See the LICENSE file in the respective project repository for the full terms of the MIT License.

## Acknowledgments

- Thanks to [Limbo](https://github.com/limbo018/Limbo) for providing the open-source code/asset that was used in this project.