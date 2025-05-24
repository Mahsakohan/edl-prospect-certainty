import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence, edl_mse_loss_with_prospect
from sklearn.metrics import f1_score


def train_model(
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    uncertainty=False,
    source_distribution=None,
):

    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            all_preds = []
            all_labels = []

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        # y = one_hot_embedding(labels, num_classes)
                        y = one_hot_embedding(labels, num_classes, smoothing=0.1)

                        y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.detach().cpu().numpy())
                        all_labels.extend(labels.detach().cpu().numpy())

                        # loss = criterion(
                        #     outputs, y.float(), epoch, num_classes, 10, device
                        # )
                        # loss, normalized_edl_loss, normalized_certainty_penalty = criterion(
                        #         outputs, y.float(), epoch, num_classes, 10, device, source_distribution, num_epochs=num_epochs, target_alpha=0.1
                        #     )

                        loss, normalized_edl_loss, normalized_certainty_penalty = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device,
                            source_distribution, lambda_certainty=0.5
                        )


                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                        acc = torch.mean(match)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')
            epoch_acc = epoch_f1  # use F1 instead of accuracy


            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc)
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            print(
                "{} Final Loss: {:.4f} | Normalized EDL Loss: {:.4f} | Certainty Penalty: {:.4f} | F1-score: {:.4f}".format(
                    phase.capitalize(), epoch_loss, normalized_edl_loss.item(), normalized_certainty_penalty.item(), epoch_f1
                )
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        if uncertainty and phase == "val":
            print(f"Evidence | Mean: {mean_evidence:.2f}, Success: {mean_evidence_succ:.2f}, Fail: {mean_evidence_fail:.2f}")

        print()
        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val F1-score: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics
