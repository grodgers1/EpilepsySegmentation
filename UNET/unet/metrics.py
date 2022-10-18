import torch
from torch.autograd import Variable
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import torchmetrics

EPSILON = 1e-32


# New metrics
class CrossEntropyLoss(torch.nn.Module):
    def __init__(self,weights=None):
        super(CrossEntropyLoss,self).__init__()
        self.weights = weights

    def forward(self, preds, target):
        return my_crossentropy(preds, target, self.weights, as_loss=True)


class JaccardLoss(torch.nn.Module):
    def __init__(self, num_classes=4, as_loss=True):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.as_loss = as_loss

    def forward(self, preds, target):
        return Variable(my_jaccard(preds, target, num_classes=self.num_classes, as_loss=self.as_loss),
                        requires_grad=True)


class DiceLoss(torch.nn.Module):
    def __init__(self, as_loss=True):
        super(DiceLoss, self).__init__()
        self.as_loss = as_loss

    def forward(self, preds, target):
        return Variable(my_dice(preds, target, as_loss=self.as_loss), requires_grad=True)


def my_crossentropy(preds, target, weights=None, as_loss=False):
    ce_loss = F.cross_entropy(
        preds.float(),
        target.long(),
        weight=weights,
    )

    if not as_loss:
        ce_loss = ce_loss.item()

    return ce_loss


def my_dice(preds, target, as_loss=False):
    dice_loss = torchmetrics.functional.dice_score(preds, target)
    if as_loss:
        dice_loss = 1 - dice_loss
    else:
        dice_loss = dice_loss.item()

    return dice_loss


def my_dice_labels(preds, target, as_loss=False):
    preds = torch.argmax(preds, dim=1).squeeze(dim=1)
    dice_loss = torchmetrics.functional.dice_score(preds, target)
    if as_loss:
        dice_loss = 1 - dice_loss
    else:
        dice_loss = dice_loss.item()

    return dice_loss


def my_F1(preds, target, as_loss=False):
    f1_loss = torchmetrics.functional.f1_score(preds, target, average='micro', mdmc_average='samplewise', num_classes=4)
    if as_loss:
        f1_loss = 1 - f1_loss
    else:
        f1_loss = f1_loss.item()

    return f1_loss


def my_F1_labels(preds, target, as_loss=False):
    preds = torch.argmax(preds, dim=1).squeeze(dim=1)
    f1_loss = torchmetrics.functional.f1_score(preds, target, average='micro', mdmc_average='samplewise', num_classes=4)
    if as_loss:
        f1_loss = 1 - f1_loss
    else:
        f1_loss = f1_loss.item()

    return f1_loss


def my_jaccard(preds, target, num_classes=4, as_loss=False):
    jaccard_loss = torchmetrics.functional.jaccard_index(preds, target, num_classes)
    if as_loss:
        jaccard_loss = 1 - jaccard_loss
    else:
        jaccard_loss = jaccard_loss.item()

    return jaccard_loss


def my_jaccard_labels(preds, target, num_classes=4, as_loss=False):
    preds = torch.argmax(preds, dim=1).squeeze(dim=1)
    jaccard_loss = torchmetrics.functional.jaccard_index(preds, target, num_classes)
    if as_loss:
        jaccard_loss = 1 - jaccard_loss
    else:
        jaccard_loss = jaccard_loss.item()

    return jaccard_loss


# metrics I previously added
def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weights: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits.float(),
        true.long(),
        ignore_index=ignore,
        weight=weights,
    )
    return ce_loss


def dice_score(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return dice_loss


def jaccard_score(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)

    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return jacc_loss


# metrics in original code
class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    dims = (0, *range(2, len(output.shape)))
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output * gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return (classwise_scores * weights).sum().item()

    return weighted_metric


jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)

if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))
