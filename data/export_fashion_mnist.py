import numpy as np
import pandas as pd
from torchvision.datasets import FashionMNIST
from data import BASE_DIR, DATASETS, DatasetName


def export_fashion_mnist_to_csv(num_samples: int = 100, output_file: str = DATASETS[DatasetName.FASHIONMNIST].csv_file):
    """Export first N samples from Fashion MNIST test set to CSV"""
    # Load Fashion MNIST test set
    testset = FashionMNIST(BASE_DIR, train=False, download=True)

    labels = []
    images = []

    for i in range(min(num_samples, len(testset))):
        image, label = testset[i]
        # Convert PIL image to [0, 255] uint8
        image = np.array(image, dtype=np.uint8).flatten()
        labels.append(label)
        images.append(image)

    # Create DataFrame with label as first column
    df = pd.DataFrame(images)
    df.insert(0, 'label', labels)

    # Save to CSV
    df.to_csv(output_file, index=False, header=False)
    print(f"Saved {len(df)} Fashion MNIST test samples to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export Fashion MNIST test set to CSV")
    parser.add_argument("-n", "--num-samples", type=int, default=100, help="Number of samples to export")
    parser.add_argument("-o", "--output", type=str, default=DATASETS[DatasetName.FASHIONMNIST].csv_file, help="Output CSV file path")
    args = parser.parse_args()

    export_fashion_mnist_to_csv(args.num_samples, args.output)
