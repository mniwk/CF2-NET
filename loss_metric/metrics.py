import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Flatten

__all__ = [
    'iou_score', 'jaccard_score', 'f1_score', 'f2_score', 'dice_score',
    'get_f_score', 'get_iou_score', 'get_jaccard_score', 'binary_focal_loss_fixed'
]

SMOOTH = 1


# ============================ Jaccard/IoU score ============================
import tensorflow as tf 

def iou_score(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
    (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
    similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
    and is defined as the size of the intersection divided by the size of the union of the sample sets:

    .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        IoU/Jaccard score in range [0, 1]

    .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

    """
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    # pr_2 = K.cast(K.greater(pr, 0), dtype=pr.dtype)
    # K.print_tensor(pr_2, message="pr_2 is: ")
    intersection = K.sum(gt * pr, axis=axes)
    union = K.sum(gt + pr, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou


def get_iou_score(class_weights=1., smooth=SMOOTH, per_image=True):
    """Change default parameters of IoU/Jaccard score

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        ``callable``: IoU/Jaccard score
    """
    def score(gt, pr):
        return iou_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)

    return score


jaccard_score = iou_score
get_jaccard_score = get_iou_score

# Update custom objects
get_custom_objects().update({
    'iou_score': iou_score,
    'jaccard_score': jaccard_score,
})


# ============================== F/Dice - score ==============================

def f_score(gt, pr, class_weights=1, beta=1, smooth=SMOOTH, per_image=True):
    r"""The F-score (Dice coefficient) can be interpreted as a weighted average of the precision and recall,
    where an F-score reaches its best value at 1 and worst score at 0.
    The relative contribution of ``precision`` and ``recall`` to the F1-score are equal.
    The formula for the F score is:

    .. math:: F_\beta(precision, recall) = (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


    where:
        TP - true positive;
        FP - false positive;
        FN - false negative;

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        beta: f-score coefficient
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        F-score in range [0, 1]

    """
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    tp = K.sum(gt * tf.round(pr), axis=axes)
    fp = K.sum(tf.round(pr), axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    # score = 2*tp/(fp+fn+2*tp)
    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score


def get_f_score(class_weights=1, beta=1, smooth=SMOOTH, per_image=True):
    """Change default parameters of F-score score

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        beta: f-score coefficient
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        ``callable``: F-score
    """
    def score(gt, pr):
        return f_score(gt, pr, class_weights=class_weights, beta=beta, smooth=smooth, per_image=per_image)

    return score


f1_score = get_f_score(beta=1)
f2_score = get_f_score(beta=2)
dice_score = f1_score




def mydice(gt, pr, class_weights=1, beta=1, smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    # 加入权重
    count_neg = tf.reduce_sum(1. - gt, axis=axes)
    count_neg = tf.cast(count_neg, dtype=tf.float32)
    count_pos = tf.reduce_sum(gt, axis=axes)
    count_pos = tf.cast(count_pos, dtype=tf.float32)
    weights = tf.div(count_neg, count_pos+count_neg)
    tp = K.sum(gt * pr, axis=axes)
    fp = K.sum(pr, axis=axes) - tp
    fn = K.sum(gt, axis=axes) - tp
    tp2 = K.sum((1-gt)*(1-pr), axis=axes)
    fp2 = K.sum((1-pr), axis=axes)
    fn2 = K.sum((1-gt), axis=axes)

    score = (weights*((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)+\
                (1-weights)*((1 + beta ** 2) * tp2 + smooth)\
                / ((1 + beta ** 2) * tp2 + beta ** 2 * fn2 + fp2 + smooth))
    # score = 2*tp/(fp+fn+2*tp)
    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score

# Update custom objects
get_custom_objects().update({
    'f1_score': f1_score,
    'f2_score': f2_score,
    'dice_score': dice_score,
})

def binary_focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25, per_image=True):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """

    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    fc1 = K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis=axes)
    fc0 = K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=axes)

    return -K.mean(fc1)-K.mean(fc0)

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def sigmoid_cross_entropy_balanced(y_true, y_pred, per_image=True):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    """

    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    y = tf.cast(y_true, tf.float32)

    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    logits   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    # logits   = tf.log(logits/ (1 - logits))

    count_neg = tf.reduce_sum(1. - y, axis=axes)
    count_neg = tf.cast(count_neg, dtype=tf.float32)
    count_pos = tf.reduce_sum(y, axis=axes)
    count_pos = tf.cast(count_pos, dtype=tf.float32)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    beta = tf.reshape(beta, (-1,1,1,1))
    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    cost = pos_weight*y*(-K.log(logits))+(1.-y)*(-K.log(1-logits))

    # cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
    # Multiply by 1 - beta
    cost = tf.reduce_sum(cost * (1 - beta), axis=axes)
    cost = tf.reduce_mean(cost, axis=0)

    # check if image has no edge pixels return 0 else return complete error function tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
    return cost


def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def ofuse_pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')