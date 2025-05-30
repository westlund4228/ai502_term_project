import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.data import get_split_dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--split_seed', default=None, type=int, help='random seed for train/valid split')
    parser.add_argument('--model_path', default=None, type=str, help='checkpoint path to resume from')

    return parser.parse_args()


def evaluate(model, data_loader, name=""):
    model.eval()
    correct = 0
    total = 0

    print(f"\nEvaluating on {name} ({len(data_loader)} batches)...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(data_loader):
                acc = 100. * correct / total
                print(f"[{batch_idx + 1:3d}/{len(data_loader)}] Current Accuracy: {acc:.2f}%")

    final_acc = 100. * correct / total
    print("=" * 40)
    print(f"{name} Final Accuracy: {final_acc:.2f}%")
    print("=" * 40 + "\n")
    return final_acc


if __name__ == '__main__':
    args = parse_args()

    split_seed = args.split_seed
    model_name = args.model
    model_path = args.model_path

    batch_size = 256
    val_size = 5000
    data_root = './data'

    train_loader, val_loader, n_class = get_split_dataset(
        'cifar10', batch_size, 2, val_size, data_root=data_root, split_seed=split_seed)

    total_indices = np.arange(50000)
    np.random.seed(split_seed)
    np.random.shuffle(total_indices)
    train_indices = total_indices[val_size:] 

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    ])

    trainset_clean = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_test)
    train_loader_eval = torch.utils.data.DataLoader(
        trainset_clean,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=2,
        pin_memory=True
    )

    testset_final = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset_final, batch_size=batch_size,
                                                    shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'mobilenet':
        from models.mobilenet_cifar import MobileNet_CIFAR
        model = MobileNet_CIFAR(n_class=10).to(device)
    elif model_name == 'resnet':
        from models.resnet_cifar import ResNet, BasicBlock
        model = ResNet(BasicBlock, [2, 2, 2, 2], n_class=10).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Model Path: {model_path}")

    model.eval()

    train_acc = evaluate(model, train_loader_eval, name="Train")
    val_acc = evaluate(model, val_loader, name="Validation")
    test_acc = evaluate(model, test_loader, name="Test")

    t1 = total_indices

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")