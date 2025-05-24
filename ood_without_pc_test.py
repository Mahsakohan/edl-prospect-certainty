import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

from helpers import get_device
from resnet import ResNet18

def load_imagenet_subset(data_dir="./data/imagenet_subset", common_classes=None, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    if common_classes:
        dataset.classes = common_classes
        dataset.class_to_idx = {name: i for i, name in enumerate(common_classes)}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def test_model_on_ood():
    device = get_device()
    model = ResNet18(num_classes=10, dropout=False).to(device)

    checkpoint = torch.load("./results/model_uncertainty_mse.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    imagenet_loader = load_imagenet_subset(common_classes=cifar10_classes)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in imagenet_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)

    predicted_labels = torch.cat(all_preds, dim=0)
    true_labels = torch.cat(all_labels, dim=0)

    f1 = f1_score(true_labels, predicted_labels, average='macro')
    print(f"✅ OOD Test F1-score (Raw Model): {f1:.4f}")

    num_samples = len(predicted_labels)
    data = {
        "True Label": true_labels.numpy(),
        "Predicted Label": predicted_labels.numpy(),
    }
    df = pd.DataFrame(data)
    print("\n📊 OOD Prediction Table:")
    print(df)

    fig, ax = plt.subplots(figsize=(8, 0.4 * num_samples))
    ax.axis("off")
    ax.table(cellText=df.values,
             colLabels=df.columns,
             cellLoc='center',
             loc='center',
             colColours=["#cccccc"] * df.shape[1])
    plt.tight_layout()
    plt.savefig("ood_prediction_table_raw.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    test_model_on_ood()
