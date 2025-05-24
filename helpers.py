import torch
import scipy.ndimage as nd


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


# def one_hot_embedding(labels, num_classes=10):
#     # Convert to One Hot Encoding
#     y = torch.eye(num_classes)
#     return y[labels]

def one_hot_embedding(labels, num_classes=10, smoothing=0.1):
    """
    Convert class indices to smoothed one-hot encoded labels.
    smoothing: float between 0 and 1. 0 = no smoothing (standard one-hot).
    """
    confidence = 1.0 - smoothing
    label_shape = (labels.size(0), num_classes)
    with torch.no_grad():
        true_dist = torch.full(label_shape, smoothing / (num_classes - 1)).to(labels.device)
        true_dist.scatter_(1, labels.unsqueeze(1), confidence)
    return true_dist
