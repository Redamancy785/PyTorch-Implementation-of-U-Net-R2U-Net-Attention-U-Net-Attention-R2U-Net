import torch

def binary_threshold(tensor, threshold=0.5):
    return torch.where(tensor > threshold, torch.tensor(1), torch.tensor(0))


def get_accuracy(SR, GT, threshold=0.5):
    SR_binary = binary_threshold(SR, threshold)
    GT_binary = GT == torch.max(GT)

    correct_pixels = torch.sum(SR_binary == GT_binary)
    total_pixels = SR.numel()
    accuracy = float(correct_pixels) / float(total_pixels)

    return accuracy

def get_sensitivity(SR, GT, threshold=0.5):
    SR_binary = binary_threshold(SR, threshold)
    GT_binary = GT == torch.max(GT)

    TP = ((SR_binary == 1) & (GT_binary == 1)).float()
    FN = ((SR_binary == 0) & (GT_binary == 1)).float()

    sensitivity = torch.sum(TP) / (torch.sum(TP + FN) + 1e-6)
    return sensitivity

def get_specificity(SR, GT, threshold=0.5):
    SR_binary = binary_threshold(SR, threshold)
    GT_binary = GT == torch.max(GT)

    TN = ((SR_binary == 0) & (GT_binary == 0)).float()
    FP = ((SR_binary == 1) & (GT_binary == 0)).float()

    specificity = torch.sum(TN) / (torch.sum(TN + FP) + 1e-6)
    return specificity

def get_precision(SR, GT, threshold=0.5):
    SR_binary = binary_threshold(SR, threshold)
    GT_binary = GT == torch.max(GT)

    TP = ((SR_binary == 1) & (GT_binary == 1)).float()
    FP = ((SR_binary == 1) & (GT_binary == 0)).float()

    precision = torch.sum(TP) / (torch.sum(TP + FP) + 1e-6)
    return precision

def get_F1(SR, GT, threshold=0.5):
    sensitivity = get_sensitivity(SR, GT, threshold=threshold)
    precision = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-6)
    return F1

def get_JS(SR, GT, threshold=0.5):
    SR_binary = binary_threshold(SR, threshold)
    GT_binary = GT == torch.max(GT)

    intersection = torch.sum((SR_binary + GT_binary) == 2)
    union = torch.sum((SR_binary + GT_binary) >= 1)

    JS = float(intersection) / (float(union) + 1e-6)
    return JS

def get_DC(SR, GT, threshold=0.5):
    SR_binary = binary_threshold(SR, threshold)
    GT_binary = GT == torch.max(GT)

    intersection = torch.sum((SR_binary + GT_binary) == 2)
    dice_coefficient = float(2 * intersection) / (float(torch.sum(SR_binary) + torch.sum(GT_binary)) + 1e-6)

    return dice_coefficient
