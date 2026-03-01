## Requirements

- matplotlib
- tqdm
- torch
- tensorboard

## Dataset

The directory structure for datasets looks like:
```
datasets
├── cifar-10
├── cifar-100
├── stl-10
├── imagenet32
└── imagenet64
```


## Usage
if you want to run the code only , fixmatch based method then run fixmatchSCAD.py
if you want to run the code, SCAD based method with ACR then run acrwithSCAD.py

For example:

For consistent:
```
python acrwithSCAD.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --ema-u 0.99 --out out/cifar-10/N500_M4000/consistent
```

For uniform:
```
python acrwithSCAD.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 1 --ema-u 0.99 --out out/cifar-10/N500_M4000/uniform
```

For reversed:
```
python acrwithSCAD.py --dataset cifar10 --num-max 500 --num-max-u 4000 --arch wideresnet --batch-size 64 --lr 0.03 --seed 0 --imb-ratio-label 100 --imb-ratio-unlabel 100 --ema-u 0.99 --flag-reverse-LT 1 --out out/cifar-10/N500_M4000/reversed
```

## acknowlegement
this code is based on ACR and fixmatch-pytorch.

