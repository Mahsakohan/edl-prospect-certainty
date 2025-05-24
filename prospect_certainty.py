import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
import random

EPSILON = 1e-5
S = 1e-3
E = np.e
gamma_b = 2.25
eps1 = eps2 = 0.88
gamma_w = 0.61


# Step 1: Create logit masks (random sparse connections)
def apply_logit_masking(model_output, nm=3, ratio_range=(0.4, 0.7)):
    B, C = model_output.shape
    masks = []

    for _ in range(nm):
        ratio = np.random.uniform(*ratio_range)
        k = int(C * ratio)
        indices = np.sort(np.random.choice(C, k, replace=False))

        # Create sparse masked version of logits
        mask = torch.zeros_like(model_output)
        mask[:, indices] = model_output[:, indices]
        masks.append(mask)

    return masks  # list of Tensors


# Step 2: Weighted probability
def compute_weighted_probs(logits_and_masks):
    all_logits = torch.stack(logits_and_masks)  # shape: [nm+1, B, C]
    mean_logits = all_logits.mean(dim=0)
    weighted_probs = []

    for logit in logits_and_masks:
        diff = torch.abs(logit - mean_logits)
        w = 1.0 / torch.log(diff / diff.mean() + E + EPSILON + S)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.ones_like(w) / len(logits_and_masks)  # equal count
        weighted = w * probs
        weighted_probs.append(weighted)

    weighted_probs = torch.stack(weighted_probs)  # [nm+1, B, C]
    normed_probs = weighted_probs / weighted_probs.sum(dim=0, keepdim=True)
    return normed_probs


# Step 3: Behavior function using Wasserstein distance
def compute_behavior_scores(logits_and_masks, source_distribution):
    """
    logits_and_masks: list of Tensors [original logits + mask logits]
    source_distribution: list or tensor [distribution of training labels]
    """
    batch_size = logits_and_masks[0].shape[0]
    device = logits_and_masks[0].device
    
    behaviors = []

    # Assume the starting point (before any mask) is just the original model output

    previous_outputs = F.softmax(logits_and_masks[0], dim=1).detach()  # Shape [B, C]

    # Precompute source_distribution as tensor
    source_distribution = torch.tensor(source_distribution, device=device, dtype=torch.float32)
    source_distribution = source_distribution / source_distribution.sum()  # normalize
    source_distribution = source_distribution.unsqueeze(0).expand(batch_size, -1)  # [B, C]

    for logit in logits_and_masks:
        current_outputs = F.softmax(logit, dim=1)

        # --- Compute Wasserstein distances ---
        # Previous distance (before updating with this mask/logit)
        wasserstein_before = torch.tensor([
            wasserstein_distance(previous_outputs[i].cpu().numpy(), source_distribution[i].cpu().numpy())
            for i in range(batch_size)
        ], device=device)

        # New distance (after adding this mask/logit)
        wasserstein_after = torch.tensor([
            wasserstein_distance(current_outputs[i].detach().cpu().numpy(), source_distribution[i].detach().cpu().numpy())
            for i in range(batch_size)
        ], device=device)

        # --- Behavior score ---
        behavior = wasserstein_before - wasserstein_after  # Positive means improvement
        behaviors.append(behavior.unsqueeze(1))  # [B, 1]

    behaviors = torch.stack(behaviors)  # Shape: [nm+1, B, 1]

    return behaviors


# Step 4: Prospect Certainty calculation
def compute_prospect_certainty(weighted_probs, behavior_scores):
    certainties = []
    for w_prob, b_score in zip(weighted_probs, behavior_scores.squeeze(2)):
        # Value function Ω_b
        mask1 = b_score >= 0
        omega_b = torch.where(mask1, b_score ** eps1, -gamma_b * (-b_score) ** eps2)

        # Weighting function Ω_w
        omega_w = torch.exp(-(-torch.log(w_prob + EPSILON)) ** gamma_w)
        omega = omega_b * omega_w.sum(dim=1)  # sum across classes
        certainties.append(omega)  # [B]

    return torch.stack(certainties)  # [nm+1, B]

# Step 5: Select refined output based on max Ω
def refine_logits_with_prospect_certainty(model, dataloader, source_distribution, device):
    model.eval()
    refined_outputs = []
    certainty_scores = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # [B, C]
            original = outputs

            masks = apply_logit_masking(outputs)
            all_variants = [original] + masks

            weighted_probs = compute_weighted_probs(all_variants)  # [nm+1, B, C]
            behavior_scores = compute_behavior_scores(all_variants, source_distribution)
            certainties = compute_prospect_certainty(weighted_probs, behavior_scores)  # [nm+1, B]

            best_indices = certainties.argmax(dim=0)  # [B]
            batch_refined = torch.stack(all_variants)  # [nm+1, B, C]
            refined = batch_refined[best_indices, torch.arange(outputs.shape[0])]  # [B, C]

            refined_outputs.append(refined)
            certainty_scores.append(certainties.max(dim=0).values)

    refined_outputs = torch.cat(refined_outputs, dim=0)
    certainty_scores = torch.cat(certainty_scores, dim=0)

    min_certainty = certainty_scores.min()
    max_certainty = certainty_scores.max()
    certainty_scores = (certainty_scores - min_certainty) / (max_certainty - min_certainty + 1e-8)  # tiny epsilon to avoid division by zero

    return refined_outputs, certainty_scores

