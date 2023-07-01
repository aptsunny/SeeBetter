# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform

from mmagic.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomMirrorSequence(BaseTransform):
    """Extend short sequences (e.g. Vimeo-90K) by mirroring the sequences.

    Given a sequence with N frames (x1, ..., xN), extend the sequence to
    (x1, ..., xN, xN, ..., x1).

    Required Keys:

    - [KEYS]

    Modified Keys:

    - [KEYS]

    Args:
        keys (list[str]): The frame lists to be extended.


    CFG:
        dict(type='RandomMirrorSequence', keys=['img'], fix_center='gt', mirror_ratio=0.5, shuffle=True)

        fix the center frame as gt after shuffle the framelist.
    """

    def __init__(self, keys, fix_center=None, mirror_ratio=0.5, shuffle=False):
        self.keys = keys
        self.fix_center = fix_center
        self.mirror_ratio = mirror_ratio
        self.shuffle = shuffle

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        mirror = np.random.random() < self.mirror_ratio

        for key in self.keys:
            if isinstance(results[key], list):
                if self.shuffle:
                    np.random.shuffle(results[key])
                    if self.fix_center is not None:
                        results[self.fix_center] = [results[key][2]]
                if mirror:
                    results[key] = results[key][::-1]
            else:
                raise TypeError('The input must be of class list[nparray]. '
                                f'Got {type(results[key])}.')

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')

        return repr_str


@TRANSFORMS.register_module()
class MirrorSequence(BaseTransform):
    """Extend short sequences (e.g. Vimeo-90K) by mirroring the sequences.

    Given a sequence with N frames (x1, ..., xN), extend the sequence to
    (x1, ..., xN, xN, ..., x1).

    Required Keys:

    - [KEYS]

    Modified Keys:

    - [KEYS]

    Args:
        keys (list[str]): The frame lists to be extended.
    """

    def __init__(self, keys):

        self.keys = keys

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        for key in self.keys:
            if isinstance(results[key], list):
                results[key] = results[key] + results[key][::-1]
            else:
                raise TypeError('The input must be of class list[nparray]. '
                                f'Got {type(results[key])}.')

        return results

    def __repr__(self):

        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys})')

        return repr_str


@TRANSFORMS.register_module()
class TemporalReverse(BaseTransform):
    """Reverse frame lists for temporal augmentation.

    Required keys are the keys in attributes "lq" and "gt",
    added or modified keys are "lq", "gt" and "reverse".

    Args:
        keys (list[str]): The frame lists to be reversed.
        reverse_ratio (float): The probability to reverse the frame lists.
            Default: 0.5.
    """

    def __init__(self, keys, reverse_ratio=0.5):

        self.keys = keys
        self.reverse_ratio = reverse_ratio

    def transform(self, results):
        """transform function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        reverse = np.random.random() < self.reverse_ratio

        if reverse:
            for key in self.keys:
                results[key].reverse()

        results['reverse'] = reverse

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, reverse_ratio={self.reverse_ratio})'
        return repr_str
