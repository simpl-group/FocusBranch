import torch
from data import DatasetName, DATASETS, BASE_DIR
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from models import MODEL_SPECS, get_network_and_dataset_utils


def build_loader(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    start_idx: int=0,
    num_batches: int=1
) -> DataLoader:
    num_samples = min(num_batches * batch_size, len(images))
    images, labels = images[start_idx:start_idx+num_samples], labels[start_idx:start_idx+num_samples]
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def filter_testset(dataset_name: DatasetName, testset: torch.utils.data.Dataset, batch_size: int=256, device: str="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Load and filter correctly classified test samples"""
    nets: list[torch.nn.Sequential] = []
    for name, model_spec in MODEL_SPECS.items():
        if model_spec.dataset_name == dataset_name and not model_spec.debug:
            nets.append(get_network_and_dataset_utils(name, device=device)[0])
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
        description="Filter and collect correctly classified samples from vision testset"
    )
    parser.add_argument("--filter-all", action="store_true", help="Enable filtering of all datasets")
    parser.add_argument("--save", action="store_true", help="Save the filtered dataset")
    parser.add_argument("-bs", "--batch-size", type=int, default=256, help="Batch size for filtering")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to use for filtering")
    args = parser.parse_args()

    for dataset_name in DatasetName:
        testset = DATASETS[dataset_name].dataset_class(BASE_DIR, train=False, transform=ToTensor(), download=True)
        print(f"Loaded {len(testset)} {dataset_name} test samples.")

        if args.filter_all:
            images, labels = filter_testset(dataset_name, testset, args.batch_size, args.device)
            print(f"Filtered {images.shape} images and {labels.shape} labels loaded successfully.")
        else:
            test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            images_list, labels_list = [], []
            for batch_images, batch_labels in test_loader:
                images_list.append(batch_images)
                labels_list.append(batch_labels)
            images, labels = torch.cat(images_list), torch.cat(labels_list)
            print(f"Loaded {images.shape} images and {labels.shape} labels without filtering.")

        if args.save:
            torch.save((images, labels), DATASETS[dataset_name].pt_file)
            print(f"Dataset saved to {DATASETS[dataset_name].pt_file}")
        else:
            print(f"Dataset not saved. Use --save to save the dataset.")