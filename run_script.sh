#!/bin/bash

# 환경 활성화
source /home/your_username/anaconda3/bin/activate sunga2

# 스크립트 실행
python C:/MST-Compression/ResNet18/main_layer_sensitivity.py --dataset CIFAR10 --algorithm kruskal --epochs 100
python C:/MST-Compression/ResNet18/main_layer_sensitivity.py --dataset CIFAR10 --algorithm prim --epochs 100
python C:/MST-Compression/ResNet18/main_layer_sensitivity.py --dataset CIFAR10 --algorithm kruskal --prune_method filters --epochs 100
python C:/MST-Compression/ResNet18/main_layer_sensitivity.py --dataset CIFAR10 --algorithm prim --prune_method filters --epochs 100
python C:/MST-Compression/ResNet18/main_layer_percentage.py --dataset CIFAR10 --algorithm kruskal --epochs 100
python C:/MST-Compression/ResNet18/main_layer_percentage.py --dataset CIFAR10 --algorithm prim --epochs 100
python C:/MST-Compression/ResNet18/main_layer_percentage.py --dataset CIFAR10 --algorithm kruskal --pruning_method prune_model_filters_by_importance --epochs 100
python C:/MST-Compression/ResNet18/main_layer_percentage.py --dataset CIFAR10 --algorithm prim --pruning_method prune_model_filters_by_importance --epochs 100
python C:/MST-Compression/VGG16/main_layer_sensitivity.py --dataset CIFAR10 --algorithm kruskal --epochs 100
python C:/MST-Compression/VGG16/main_layer_sensitivity.py --dataset CIFAR10 --algorithm prim --epochs 100
python C:/MST-Compression/VGG16/main_layer_sensitivity.py --dataset CIFAR10 --algorithm kruskal --prune_method filters --epochs 100
python C:/MST-Compression/VGG16/main_layer_sensitivity.py --dataset CIFAR10 --algorithm prim --prune_method filters --epochs 100

# 환경 비활성화
conda deactivate
