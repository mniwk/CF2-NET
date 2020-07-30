import keras.backend as K
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from keras.utils.generic_utils import get_custom_objects

from .metrics import jaccard_score, f_score, mydice, sigmoid_cross_entropy_balanced, cross_entropy_balanced

SMOOTH = 1e-12

__all__ = [
    'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
    'dice_loss', 'bce_dice_loss', 'cce_dice_loss',
]


# ============================== Jaccard Losses ==============================

def jaccard_loss(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
    r"""Jaccard loss function for imbalanced datasets:

    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        Jaccard loss in range [0, 1]

    """
    return 1 - jaccard_score(y_true, y_pred, class_weights=class_weights, smooth=smooth, per_image=per_image)


def bce_jaccard_loss(y_true, y_pred, bce_weight=1., smooth=SMOOTH, per_image=True):
    bce = K.mean(binary_crossentropy(y_true, y_pred))
    loss = bce_weight * bce + jaccard_loss(y_true, y_pred, smooth=smooth, per_image=per_image)
    return loss


def cce_jaccard_loss(y_true, y_pred, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    cce = categorical_crossentropy(y_true, y_pred) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + jaccard_loss(y_true, y_pred, smooth=smooth, class_weights=class_weights, per_image=per_image)


# Update custom objects
get_custom_objects().update({
    'jaccard_loss': jaccard_loss,
    'bce_jaccard_loss': bce_jaccard_loss,
    'cce_jaccard_loss': cce_jaccard_loss,
})


# ============================== Dice Losses ================================

def dice_loss(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        Dice loss in range [0, 1]

    """
    return 1 - f_score(y_true, y_pred, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=1.)


def dice_loss2(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        Dice loss in range [0, 1]

    """
    return 1 - mydice(y_true, y_pred, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=1.)


def bce_dice_loss(y_true, y_pred, bce_weight=1., smooth=SMOOTH, per_image=True):
    bce = K.mean(binary_crossentropy(y_true, y_pred))
    loss = bce_weight * bce + dice_loss2(y_true, y_pred, smooth=smooth, per_image=per_image)
    return loss


def cce_dice_loss(y_true, y_pred, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    cce = categorical_crossentropy(y_true, y_pred) * class_weights
    cce = K.mean(cce)
    return cce_weight * cce + dice_loss(y_true, y_pred, smooth=smooth, class_weights=class_weights, per_image=per_image)


# Update custom objects
get_custom_objects().update({
    'dice_loss': dice_loss,
    'bce_dice_loss': bce_dice_loss,
    'cce_dice_loss': cce_dice_loss,
})



def edge_loss2(y_true, y_pred, per_image=True):
    return sigmoid_cross_entropy_balanced(y_true, y_pred, per_image=True)

def edge_loss(y_true, y_pred, per_image=True):
    return cross_entropy_balanced(y_true, y_pred)  