#
#   Copyright EAVISE
#   Author: Maarten Vandersteegen
#   Author: Tanguy Ophoff
#
#   Functions for generating PR-curve values and calculating average precision
#

import math
from statistics import mean
import numpy as np
import scipy.interpolate

from .util import *

__all__ = ['pr', 'ap']


def pr(detections, ground_truth, overlap_threshold=0.5):
    """ Compute a list of precision recall values that can be plotted into a graph.

    Args:
        detections (dict): Detection objects per image
        ground_truth (dict): Annotation objects per image
        overlap_threshold (Number, optional): Minimum iou threshold for true positive; Default **0.5**

    Returns:
        tuple: **[precision_values]**, **[recall_values]**
    """
    tps, fps, num_annotations = match_detections(detections, ground_truth, overlap_threshold)

    precision = []
    recall = []
    for tp, fp in zip(tps, fps):
        recall.append(tp / num_annotations)
        precision.append(tp / (fp + tp))

    return precision, recall


def ap(precision, recall, num_of_samples=100):
    """ Compute the average precision from a given pr-curve.
    The average precision is defined as the area under the curve.

    Args:
        precision (list): Precision values
        recall (list): Recall values
        num_of_samples (int, optional): Number of samples to take from the curve to measure the average precision; Default **100**

    Returns:
        Number: average precision
    """
    if len(precision) > 1 and len(recall) > 1:
        p = np.array(precision)
        r = np.array(recall)
        p_start = p[np.argmin(r)]
        samples = np.arange(0., 1., 1.0/num_of_samples)
        interpolated = scipy.interpolate.interp1d(r, p, fill_value=(p_start, 0.), bounds_error=False)(samples)
        avg = sum(interpolated) / len(interpolated)
    elif len(precision) > 0 and len(recall) > 0:
        # 1 point on PR: AP is box between (0,0) and (p,r)
        avg = precision[0] * recall[0]
    else:
        avg = float('nan')

    return avg
