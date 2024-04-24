#!/bin/bash

# 환경 활성화
source /home/your_username/anaconda3/bin/activate sunga2

# 스크립트 실행
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset CIFAR10 --algorithm kruskal --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset CIFAR10 --algorithm kruskal --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset CIFAR10 --algorithm prim --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset CIFAR10 --algorithm prim --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset MNIST --algorithm kruskal --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset MNIST --algorithm kruskal --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset MNIST --algorithm prim --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset MNIST --algorithm prim --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset ImageNet --algorithm kruskal --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 150
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset ImageNet --algorithm kruskal --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 150
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset ImageNet --algorithm prim --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 150
python /path/to/MST-Compression/ResNet18/main_layer.py --dataset ImageNet --algorithm prim --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 150

# 환경 비활성화
conda deactivate
