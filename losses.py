import torch
import torch.nn.functional as F
from helpers import get_device
import numpy as np
from scipy.stats import wasserstein_distance
from prospect_certainty import apply_logit_masking, compute_weighted_probs, compute_behavior_scores, compute_prospect_certainty

def relu_evidence(y):
    return F.relu(y)

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood



def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)

    # Original loglikelihood
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    # Normalize loglikelihood per sample
    loglikelihood = loglikelihood / (loglikelihood.max().detach() + 1e-8)

    # KL divergence
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = kl_divergence(kl_alpha, num_classes, device=device)

    # Normalize KL per sample
    kl_div = kl_div / (kl_div.max().detach() + 1e-8)

    # Combine and bound the final loss to [0, 1]
    total_loss = 0.5 * loglikelihood + 0.5 * annealing_coef * kl_div
    total_loss = (total_loss - total_loss.min().detach()) / (total_loss.max().detach() - total_loss.min().detach() + 1e-8)
    return total_loss


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = torch.clamp(relu_evidence(output), min=0, max=30)
    alpha = evidence + 1
    per_sample_loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    return per_sample_loss.mean()


# --- Prospect Certainty additions ---
def edl_mse_loss_with_prospect(output, target, epoch_num, num_classes, annealing_step, device, source_distribution, lambda_certainty=0.5):
    if not device:
        device = get_device()
    evidence = torch.clamp(relu_evidence(output), min=0, max=30)
    alpha = evidence + 1

    # EDL loss (already ∈ [0,1])
    per_sample_edl = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device).squeeze()
    

    # Prospect Certainty
    original_logits = output.detach()
    masked_logits = apply_logit_masking(original_logits)
    all_logits = [original_logits] + masked_logits

    weighted_probs = compute_weighted_probs(all_logits)
    behavior_scores = compute_behavior_scores(all_logits, source_distribution)
    certainties = compute_prospect_certainty(weighted_probs, behavior_scores)
    max_certainty = certainties.max(dim=0).values  # [B]
    max_certainty = (max_certainty - max_certainty.min()) / (max_certainty.max() - max_certainty.min() + 1e-8)
    
    # print(f"[DEBUG] EDL loss per sample range: min={per_sample_edl.min().item():.4f}, max={per_sample_edl.max().item():.4f}")
    # print(f"[DEBUG] Certainty (all variants) range: min={certainties.min().item():.4f}, max={certainties.max().item():.4f}")
    # print(f"[DEBUG] Max certainty per sample range: min={max_certainty.min().item():.4f}, max={max_certainty.max().item():.4f}")
    # print()

    # Certainty Penalty
    certainty_penalty = 1.0 - max_certainty  # [B]


    # Gradually increase weight on certainty penalty
    # lambda_certainty = min(lambda_certainty, epoch_num / (annealing_step * 2))
    lambda_certainty = min(0.5, epoch_num / (annealing_step * 3))


    combined_loss = per_sample_edl + lambda_certainty * certainty_penalty
    return combined_loss.mean(), per_sample_edl.mean(), certainty_penalty.mean()
