import torch
from data import DatasetName, DATASETS, BASE_DIR
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from models import MODEL_SPECS, get_network
from utils import DEVICE


def get_loader(dataset_name: DatasetName, batch_size: int, target_label: int | None=None, num_batches: int | None=None) -> DataLoader:
    """Load dataset and return a DataLoader"""
    images, labels = torch.load(DATASETS[dataset_name].pt_file, map_location="cpu")
    if target_label is not None:
        non_targeted_indices = (labels != target_label).nonzero().squeeze(1)
        images, labels = images[non_targeted_indices], labels[non_targeted_indices]
    if num_batches is not None:
        num_samples = min(num_batches * batch_size, len(images))
        images, labels = images[:num_samples], labels[:num_samples]
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def filter_testset(dataset_name: DatasetName, batch_size: int=256, device: str="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Load and filter correctly classified test samples"""
    testset = DATASETS[dataset_name].dataset_class(BASE_DIR, train=False, transform=ToTensor(), download=True)
    print(f"Loaded {len(testset)} {dataset_name} test samples.")
    nets: list[torch.nn.Sequential] = []
    for name, model_spec in MODEL_SPECS.items():
        if model_spec.dataset_name == dataset_name and not model_spec.debug:
            nets.append(get_network(name, device=device))
    print(f"Loaded {len(nets)} {dataset_name} networks for filtering.")
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    filtered_images, filtered_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            correct_mask = torch.ones(labels.size(), dtype=torch.bool, device=device)
            for net in nets:
                preds = net(images).argmax(dim=1)
                correct_mask &= (preds == labels)

            if correct_mask.any():
                filtered_images.append(images[correct_mask])
                filtered_labels.append(labels[correct_mask])
    return torch.cat(filtered_images), torch.cat(filtered_labels)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Filter and collect correctly classified samples from vison testset"
    )
    parser.add_argument("--save", action="store_true", help="Save the filtered dataset")
    parser.add_argument("-bs", "--batch-size", type=int, default=256, help="Batch size for filtering")
    args = parser.parse_args()

    for dataset_name in DatasetName:
        filtered_images, filtered_labels = filter_testset(dataset_name, args.batch_size, DEVICE)
        print(f"Filtered {filtered_images.shape} images and {filtered_labels.shape} labels loaded successfully.")
        if args.save:
            torch.save((filtered_images, filtered_labels), DATASETS[dataset_name].pt_file)
            print(f"Filtered dataset saved to {DATASETS[dataset_name].pt_file}")
        else:
            print(f"Filtered dataset not saved. Use --save to save the dataset.")