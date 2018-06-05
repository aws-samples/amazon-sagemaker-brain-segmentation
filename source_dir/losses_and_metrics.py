import mxnet as mx
import mxnet.ndarray as F
import numpy as np

def avg_dice_coef_loss(y_true, y_pred, class_weights):    
    intersection = mx.sym.sum(y_true * y_pred, axis=(2, 3))
    scores =  1 - mx.sym.broadcast_div((2. * intersection + 1.),
                                 (mx.sym.broadcast_add(mx.sym.sum(y_true, axis=(2, 3)),
                                                       mx.sym.sum(y_pred, axis=(2, 3))) + 1.))
    average = mx.sym.broadcast_div(mx.sym.sum(mx.sym.broadcast_mul(scores, class_weights), axis=1),
                                   mx.sym.sum(class_weights))
    return average

def channel_wise_softmax(decoded):
    denominator = mx.sym.sum(mx.sym.exp(decoded), axis=1, keepdims=True)
    return mx.sym.broadcast_div(mx.sym.exp(decoded), denominator)

def avg_dice_coef_metric(y_true, y_pred, num_classes=4):
    intersection = np.sum(y_true*y_pred, axis=(2, 3))
    scores =  1 - (2. * intersection + 1.)/(np.sum(y_true, axis=(2, 3))+np.sum(y_pred, axis=(2, 3)) + 1.)
    batch_average = np.mean(np.sum(scores, axis=1)/ num_classes)
    return batch_average