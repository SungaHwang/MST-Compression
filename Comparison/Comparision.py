import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import os
import logging
import random
import argparse
import time
import torch.nn.utils.prune as prune
import pandas as pd

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def prune_model(model, method, pruning_dict):
    model.cpu()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if name in pruning_dict:
                amount = pruning_dict[name]
                if method == "l1_unstructured":
                    prune.l1_unstructured(module, name='weight', amount=amount)
                elif method == "random_unstructured":
                    prune.random_unstructured(module, name='weight', amount=amount)
                elif method == "l2_structured":
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                elif method == "global_l1_unstructured":
                    prune.global_unstructured(
                        [(module, 'weight')],
                        pruning_method=prune.L1Unstructured,
                        amount=amount,
                    )
                elif method == "global_random_unstructured":
                    prune.global_unstructured(
                        [(module, 'weight')],
                        pruning_method=prune.RandomUnstructured,
                        amount=amount,
                    )
                elif method == "l1_structured":
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                elif method == "random_structured":
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                elif method == "smallest_weights_unstructured":
                    prune.ln_structured(module, name='weight', amount=amount, n=0, dim=0)
                elif method == "largest_weights_unstructured":
                    prune.ln_structured(module, name='weight', amount=amount, n=float('inf'), dim=0)
                elif method == "norm_structured":
                    prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
                else:
                    raise ValueError(f"Unknown pruning method: {method}")

                # Make pruning permanent by removing pruned weights
                prune.remove(module, 'weight')

    model.to(device)
    return model

def compute_flops(model, input_size=(1, 3, 32, 32)):
    model = model.to(device)
    model.eval()
    input = torch.randn(input_size).to(device)

    total_flops = 0

    def conv_flops_hook(module, input, output):
        weight = module.weight
        active_elements_count = torch.sum(weight != 0).item()

        output_dims = output.shape[2:]
        kernel_dims = module.kernel_size
        in_channels = module.in_channels / module.groups
        flops = torch.prod(torch.tensor(kernel_dims)).item() * in_channels * active_elements_count * torch.prod(torch.tensor(output_dims)).item()

        nonlocal total_flops
        total_flops += flops

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hook = layer.register_forward_hook(conv_flops_hook)
            hooks.append(hook)

    with torch.no_grad():
        model(input)

    for hook in hooks:
        hook.remove()

    return total_flops

def count_nonzero_parameters(model):
    total_params = 0
    zero_count = 0
    for param in model.parameters():
        param_count = param.numel()
        zero_count += torch.sum(param == 0).item()
        total_params += param_count

    nonzero_count = total_params - zero_count
    return nonzero_count

def evaluate_model_full(model, test_loader):
    model = model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = end_time - start_time

    flops = compute_flops(model)
    nonzero_params = count_nonzero_parameters(model)

    return accuracy, inference_time, flops, nonzero_params

def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    logging.info("Starting the program")
    logging.info("Arguments: %s", args)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    save_directory = 'saved_models/Resnet18'
    filename = f"original_model_CIFAR10.pth"
    save_path = os.path.join(save_directory, filename)

    pruning_methods = [
        "l1_unstructured", "random_unstructured", "l2_structured",
        "global_l1_unstructured", "global_random_unstructured", "l1_structured",
        "random_structured", "smallest_weights_unstructured", "largest_weights_unstructured",
        "norm_structured"
    ]

    results = []

    # Load the base model and evaluate
    model = models.resnet18(weights=None)
    model.load_state_dict(torch.load(save_path, map_location=device))
    base_accuracy, base_inference_time, base_flops, base_nonzero_params = evaluate_model_full(model, test_loader)

    # Example of pruning dictionary
    pruning_dict = {
        "conv1": 0,
        "layer1.0.conv1": 0,
        "layer1.0.conv2": 0,
        "layer1.1.conv1": 0,
        "layer1.1.conv2": 0,
        "layer2.0.conv1": 0,
        "layer2.0.conv2": 0,
        "layer2.1.conv1": 0,
        "layer2.1.conv2": 0.05,
        "layer3.0.conv1": 0.05,
        "layer3.0.conv2": 0.05,
        "layer3.1.conv1": 0.4,
        "layer3.1.conv2": 0.3,
        "layer4.0.conv1": 0.3,
        "layer4.0.conv2": 0.4,
        "layer4.1.conv1": 0.7,
        "layer4.1.conv2": 0.7,
    }

    for method in pruning_methods:
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)

        pruned_model = prune_model(model, method, pruning_dict)
        
        accuracy, inference_time, flops, nonzero_params = evaluate_model_full(pruned_model, test_loader)
        
        accuracy_diff = accuracy - base_accuracy
        speedup = base_inference_time / inference_time
        
        logging.info(f"Pruning method: {method}, Base Accuracy: {base_accuracy:.2f}%, Pruned Accuracy: {accuracy:.2f}%, Δ Acc: {accuracy_diff:.2f}%, Speedup: {speedup:.2f}x, Base FLOPs: {base_flops}, Pruned FLOPs: {flops}, Base Non-zero Params: {base_nonzero_params}, Pruned Non-zero Params: {nonzero_params}")
        
        results.append({
            "Method": method,
            #"Base Accuracy (%)": base_accuracy,
            "Pruned Accuracy (%)": accuracy,
            "Δ Acc (%)": accuracy_diff,
            #"Speedup (x)": speedup,
            #"Base FLOPs": base_flops,
            #"Pruned FLOPs": flops,
            #"Base Non-zero Params": base_nonzero_params,
            "Pruned Non-zero Params": nonzero_params
        })

    # Convert results to DataFrame and print
    df_results = pd.DataFrame(results)
    print(df_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Pruning")
    args = parser.parse_args()
    main(args)
