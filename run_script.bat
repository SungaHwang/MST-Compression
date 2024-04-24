@echo On

call C:\Users\sunga\anaconda3\Scripts\activate.bat C:\Users\sunga\anaconda3\envs\sunga2

python C:\MST-Compression\ResNet18\main_layer.py --dataset CIFAR10 --algorithm kruskal --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python C:\MST-Compression\ResNet18\main_layer.py --dataset CIFAR10 --algorithm kruskal --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python C:\MST-Compression\ResNet18\main_layer.py --dataset CIFAR10 --algorithm prim --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python C:\MST-Compression\ResNet18\main_layer.py --dataset CIFAR10 --algorithm prim --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 100
python C:\MST-Compression\ResNet18\main_layer.py --dataset MNIST --algorithm kruskal --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20
python C:\MST-Compression\ResNet18\main_layer.py --dataset MNIST --algorithm kruskal --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20 
python C:\MST-Compression\ResNet18\main_layer.py --dataset MNIST --algorithm prim --pruning_method prune_model_weights --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20
python C:\MST-Compression\ResNet18\main_layer.py --dataset MNIST --algorithm prim --pruning_method prune_model_filters_by_importance --target_layers layer1.0.conv1 layer2.0.conv1 layer3.0.conv1 layer4.0.conv1 --epochs 20

call conda deactivate
