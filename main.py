import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from resnet import ResNet18
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import argparse
from matplotlib import pyplot as plt 
from PIL import Image

from helpers import get_device
from data import dataloaders
from train import train_model
from losses import edl_mse_loss, edl_mse_loss_with_prospect
from prospect_certainty import refine_logits_with_prospect_certainty
import json


def main():

    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train", action="store_true", help="To train the network."
    )
    mode_group.add_argument("--test", action="store_true", help="To test the network.")
    mode_group.add_argument(
        "--examples", action="store_true", help="To example MNIST data."
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="Desired number of epochs."
    )
    parser.add_argument(
        "--dropout", action="store_true", help="Whether to use dropout or not."
    )
    parser.add_argument(
        "--uncertainty", action="store_true", help="Use uncertainty or not."
    )

    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument(
        "--mse",
        action="store_true",
        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.",
    )
  
    args = parser.parse_args()

    if args.examples:
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig("./images/examples.jpg")

    elif args.train:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = 10

        
        model = ResNet18(num_classes=10, dropout=args.dropout)

        if use_uncertainty:
            if args.mse:
                # criterion = edl_mse_loss
                criterion = edl_mse_loss_with_prospect
            else:
                parser.error("--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


        device = get_device()
        model = model.to(device)

        print("Computing source_distribution from training set...")
        model.eval()
        softmax_outputs = []

        print("Computing source_distribution from training set...")

        label_counts = torch.zeros(num_classes)
        for _, labels in dataloaders["train"]:
            for label in labels:
                label_counts[label] += 1

        source_distribution = (label_counts / label_counts.sum()).tolist()

        # Save it for use during test time
        with open("./results/source_distribution.json", "w") as f:
            json.dump(source_distribution, f)
        print("Saved source_distribution to ./results/source_distribution.json")

        model, metrics = train_model(
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
            source_distribution=source_distribution,
        )

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if use_uncertainty:
            if args.mse:
                torch.save(state, "./results/model_uncertainty_mse.pt")
                print("Saved: ./results/model_uncertainty_mse.pt")

        else:
            torch.save(state, "./results/model.pt")
            print("Saved: ./results/model.pt")

    # elif args.test:

    #     use_uncertainty = args.uncertainty
    #     device = get_device()
    #     model = ResNet18(num_classes=10, dropout=args.dropout)
    #     model = model.to(device)
    #     optimizer = optim.Adam(model.parameters())

    #     if use_uncertainty:
    #         if args.mse:
    #             checkpoint = torch.load("./results/model_uncertainty_mse.pt")
    #             filename = "./results/rotate_uncertainty_mse.jpg"

    #     else:
    #         checkpoint = torch.load("./results/model.pt")
    #         filename = "./results/rotate.jpg"

    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    #     model.eval()    

    #     with open("./results/source_distribution.json", "r") as f:
    #         source_distribution = json.load(f)

    #     # Evaluate model on test samples using prospect certainty
    #     refined_outputs, certainty_scores = refine_logits_with_prospect_certainty(
    #         model, dataloaders["val"], source_distribution, device=device
    #     )


if __name__ == "__main__":
    main()
