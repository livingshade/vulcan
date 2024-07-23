Adpated from [PyTorch](https://github.com/pytorch/examples/tree/main/imagenet)

# ImageNet

## Setup

```
pip install -r requirements.txt
```

Download dataset from [ImageNet](http://www.image-net.org/), then extract the dataset using `bash ./extract_ILSVRC.sh`

## Usage


```
python main.py -a alexnet /data/imagenet

usage: main.py [-h] [-a ARCH] [-j N] [-b N] [-p N] [-e] [--pretrained]  [--seed SEED] [--gpu GPU] [--dummy]   [DIR]

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset (default: imagenet)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny | densenet121 | densenet161 | densenet169 | densenet201 | efficientnet_b0 |
                        efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 | efficientnet_b7 | googlenet | inception_v3 | mnasnet0_5 | mnasnet0_75 |
                        mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large | mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf |
                        regnet_x_8gf | regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 | resnet152 | resnet18 |
                        resnet34 | resnet50 | resnext101_32x8d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 | vit_b_32 | vit_l_16 | vit_l_32 | wide_resnet101_2 | wide_resnet50_2 (default: resnet18)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel
  -j N, --workers N     number of data loading workers (default: 4)
  -p N, --print-freq N  print frequency (default: 10)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --dummy               use fake data to benchmark

```

## Knob

- resnet18
- resnet34
- resnet50
- resnet101
- resnet152