import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import f1_score

from helpers import get_device
from resnet import ResNet18
from prospect_certainty import refine_logits_with_prospect_certainty


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


def test_model_with_prospect_certainty():
    device = get_device()
    model = ResNet18(num_classes=10, dropout=False).to(device)

    checkpoint = torch.load("./results/model_uncertainty_mse.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open("./results/source_distribution.json", "r") as f:
        source_distribution = json.load(f)

    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    imagenet_loader = load_imagenet_subset(common_classes=cifar10_classes)

    refined_outputs, certainty_scores = refine_logits_with_prospect_certainty(
        model, imagenet_loader, source_distribution, device=device
    )

    predicted_labels = torch.argmax(refined_outputs, dim=1)
    true_labels = torch.cat([labels for _, labels in imagenet_loader], dim=0).to(device)

    f1 = f1_score(true_labels.cpu(), predicted_labels.cpu(), average='macro')
    print(f"✅ OOD Test F1-score (Prospect Certainty): {f1:.4f}")

    # Show prediction table
    num_samples = len(predicted_labels)
    data = {
        "True Label": true_labels.cpu().numpy(),
        "Predicted Label": predicted_labels.cpu().numpy(),
        "Certainty Score": [round(c.item(), 4) for c in certainty_scores]
    }
    df = pd.DataFrame(data)
    print("\n📊 OOD Prediction Table:")
    print(df)

    # Plot table
    fig, ax = plt.subplots(figsize=(8, 0.4 * num_samples))
    ax.axis("off")
    ax.table(cellText=df.values,
             colLabels=df.columns,
             cellLoc='center',
             loc='center',
             colColours=["#cccccc"] * df.shape[1])
    plt.tight_layout()
    plt.savefig("ood_prediction_table.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    test_model_with_prospect_certainty()
