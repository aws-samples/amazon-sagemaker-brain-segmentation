import mxnet as mx
from mxnet import nd
import numpy as np

# Loss and metric derived from Sørensen–Dice_coefficient as described at below: 
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

def avg_dice_coef_loss(y_true, y_pred, class_weights):
    """
    Symbol for computing weighted average dice coefficient loss

    Parameters
    ----------
    y_true : symbol
        Symbol representing ground truth mask.
    y_pred : symbol
        Symbol representing predicted mask.
    class_weights : symbol
        Symbol of class weights.
    """
    intersection = mx.sym.sum(y_true * y_pred, axis=(2, 3))
    numerator = 2. * intersection
    denominator = mx.sym.broadcast_add(mx.sym.sum(y_true, axis=(2, 3)),
                                       mx.sym.sum(y_pred, axis=(2, 3)))
    scores = 1 - mx.sym.broadcast_div(numerator + 1., denominator + 1.)
    
    # Take weighted average to ensure normalization
    average = mx.sym.broadcast_div(mx.sym.sum(mx.sym.broadcast_mul(scores, class_weights), axis=1),
                                   mx.sym.sum(class_weights))
    return average


def avg_dice_coef_metric(y_true, y_pred, num_classes=4):
    """
    Method for computing average dice coefficient metric,
    used to create a Custom Metric.

    Parameters
    ----------
    y_true : array
        Array representing ground truth mask.
    y_pred : array
        Array representing predicted mask.
    num_classes : int
        Number of classes.
    """
    intersection = np.sum(y_true * y_pred, axis=(2, 3))
    numerator = 2. * intersection
    denominator = np.sum(y_true, axis=(2, 3)) + np.sum(y_pred, axis=(2, 3))
    scores = 1 - (numerator + 1.) / (denominator + 1.)
    batch_average = np.mean(np.sum(scores, axis=1) / num_classes)
    return batch_average
