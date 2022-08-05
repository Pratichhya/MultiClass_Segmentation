import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1).type(torch.float)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets)
        return BCE


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        BCE = F.binary_cross_entropy_with_logits(inputs, targets.type(torch.float), reduction="mean")
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, smooth=1e-7):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection + self.smooth) / (union + self.smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    """Focal Loss was introduced by Lin et al of Facebook AI Research in 2017
    as a means of combatting extremely imbalanced datasets where positive cases
    were relatively rare. Their paper "Focal Loss for Dense Object Detection" is
    retrievable here: https://arxiv.org/abs/1708.02002. In practice, the researchers
    used an alpha-modified version of the function so I have included it in this
    implementation.
    """

    def __init__(self, smooth=1e-7, alpha=0.8, gamma=0.2):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE

        return focal_loss


class TverskyLoss(nn.Module):
    """
    This loss was introduced in "Tversky loss function for image segmentation using
    3D fully convolutional deep networks", retrievable here:
    https://arxiv.org/abs/1706.05721. It was designed to optimise segmentation
    on imbalanced medical datasets by utilising constants that can adjust how
    harshly different types of error are penalised in the loss function. From the
    paper:

    ```
    in the case of α=β=0.5 the Tversky index simplifies to be the same as the Dice
    coefficient, which is also equal to the F1 score. With α=β=1, Equation 2 produces
    Tanimoto coefficient, and setting α+β=1 produces the set of Fβ scores. Larger βs
    weigh recall higher than precision (by placing more emphasis on false negatives).
    ```
    To summarise, this loss function is weighted by the constants 'alpha' and 'beta'
    that penalise false positives and false negatives respectively to a higher degree
    in the loss function as their value is increased. The beta constant in particular has
    applications in situations where models can obtain misleadingly positive performance
    via highly conservative prediction. You may want to experiment with different values
    to find the optimum. With alpha==beta==0.5, this loss becomes equivalent to Dice Loss.
    """

    def __init__(self, smooth=1e-7, alpha=0.8, beta=0.2):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
       

    def forward(self, inputs, targets):
        
        #changing the inputs to 0 and 1
        inputs = F.sigmoid(inputs)  
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - Tversky
    
class TverskyBCELoss(nn.Module):
    """
    This is a modification of the Tversky Loss which combines it with the Binary Cross
    Entropy Loss.
    
    """

    def __init__(self, smooth=1e-7, alpha=0.7, beta=0.3, ratios=[0.5,0.5]):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.ratios = ratios
       

    def forward(self, inputs, targets):
        
        #changing the inputs to 0 and 1
        sig_inputs = F.sigmoid(inputs)  
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        sig_inputs = sig_inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (sig_inputs * targets).sum()
        FP = ((1 - targets) * sig_inputs).sum()
        FN = (targets * (1 - sig_inputs)).sum()

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        
        BCE = F.binary_cross_entropy_with_logits(inputs, targets.type(torch.float),reduction="mean")
        
        Tversky_BCE = self.ratios[0]*BCE + self.ratios[1]*(1 - Tversky)

        return Tversky_BCE



class FocalTverskyLoss(nn.Module):
    """A variant on the Tversky loss that also includes the gamma modifier from Focal Loss."""

    def __init__(self, smooth=1e-7, alpha=0.5, beta=0.5, gamma=1):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky


class LovaszHingeLoss(nn.Module):
    """
    This complex loss function was introduced by Berman, Triki and Blaschko in their paper
    "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union
    measure in neural networks", retrievable here: https://arxiv.org/abs/1705.08790. It is
    designed to optimise the Intersection over Union score for semantic segmentation, particularly
    for multi-class instances. Specifically, it sorts predictions by their error before calculating
    cumulatively how each error affects the IoU score. This gradient vector is then multiplied with
    the initial error vector to penalise most strongly the predictions that decreased the IoU score
    the most. This procedure is detailed by jeandebleu in his excellent summary here.

    This code is taken directly from the author's github repo here: https://github.com/bermanmaxim/LovaszSoftmax
    and all credit is to them.

    In this kernel I have implemented the flat variant that uses reshaped rank-1 tensors as inputs for
    PyTorch. You can modify it accordingly with the dimensions and class number of your data as needed.
    This code takes raw logits so ensure your model does not contain an activation layer prior to the
    loss calculation.

    I have hidden the researchers' own code below for brevity; simply load it into your kernel for the
    losses to function. In the case of their tensorflow implementation.
    """

    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, inputs, targets):
        Lovasz = self.lovasz_hinge(
            inputs, targets, per_image=self.per_image, ignore=self.ignore
        )
        return Lovasz

    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
        """
        if per_image:
            loss = self.mean(
                self.lovasz_hinge_flat(
                    *self.flatten_binary_scores(
                        log.unsqueeze(0), lab.unsqueeze(0), ignore
                    )
                )
                for log, lab in zip(logits, labels)
            )
        else:
            loss = self.lovasz_hinge_flat(
                *self.flatten_binary_scores(logits, labels, ignore)
            )
        return loss

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
        logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * Variable(signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = labels != ignore
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == "raise":
                raise ValueError("Empty mean")
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n


class ComboLoss(nn.Module):
    """
    This loss was introduced by Taghanaki et al in their paper "Combo loss:
    Handling input and output imbalance in multi-organ segmentation", retrievable
    here: https://arxiv.org/abs/1805.02798. Combo loss is a combination of Dice
    Loss and a modified Cross-Entropy function that, like Tversky loss, has additional
    constants which penalise either false positives or false negatives more respectively.
    """

    def __init__(self, smooth=1e-7, alpha=0.5, ce_ratio=0.5, eps=1e-9):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.eps = eps

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)
        out = -(
            self.alpha
            * (
                (targets * torch.log(inputs))
                + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs))
            )
        )
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)

        return combo


