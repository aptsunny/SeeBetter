# Copyright (c) OpenMMLab. All rights reserved.
from typing import Mapping

import mmengine
import torch.distributed as dist
from mmengine.registry import DATA_SAMPLERS
from torch.utils.data import Sampler


@DATA_SAMPLERS.register_module()
class MultiSourceSampler(Sampler):
    """Multi-source sampler.

    It is designed to extract correct data in proportion from multi-source
    datasets for multi model training. It is dedicated to
    ``MultiSourceDataset``. It can be used for `IterationBased`` and
    `EpochBased` runner.

    Note:
        When used in `EpochBased` runner, at least one `EpochBased` sampler
        should be included.

    Args:
        dataset (object): Dataset to be sampled. It must be
            ``MultiSourceDataset``.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        sub_samples_per_gpu (list[int]): The amount of data that each dataset
            needs to sample in one iteration. Their sum must be equal to
            ``samples_per_gpu``. Default [samples_per_gpu/N, samples_per_gpu/N,
            ...]
        sub_samplers (list[dict, optional]): Subsamplers corresponding to each
            dataset. When it is None or no input, the default subsampler is set
            to ``InfiniteSampler``.
        num_replicas (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu,
                 sub_samples_per_gpu=None,
                 sub_samplers=None,
                 num_replicas=None,
                 rank=None,
                 seed=None,
                 shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        # set default sub_samples_per_gpu, default [samples_per_gpu/N, ...]
        if sub_samples_per_gpu is None:
            sub_samples_per_gpu = [
                samples_per_gpu // len(dataset.datasets)
                for _ in range(len(dataset.datasets))
            ]
        assert sum(sub_samples_per_gpu) == samples_per_gpu, \
            'The sum of sub_samples_per_gpu must be equal to ' \
            f'samples_per_gpu, but get {sub_samples_per_gpu}'
        self.sub_samples_per_gpu = sub_samples_per_gpu
        self.samples_per_gpu = samples_per_gpu

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle

        self.indices_offset = [0] + self.dataset.cumulative_sizes[:-1]
        self.sub_samplers = self._init_sub_samplers(sub_samplers)
        self.size = self._get_aligned_len()

    def __iter__(self):
        sub_iters = [iter(sampler) for sampler in self.sub_samplers]
        try:
            while True:
                indices = []
                for i, sub_iter in enumerate(sub_iters):
                    for _ in range(self.sub_samples_per_gpu[i]):
                        indices.append(next(sub_iter) + self.indices_offset[i])
                yield from indices
        except StopIteration:
            pass

    def __len__(self):
        return self.size

    def _init_sub_samplers(self, sub_samplers):
        """Initialize sub samplers of multi-source dataset."""
        if sub_samplers is None:
            sub_samplers = [None] * len(self.sub_samples_per_gpu)
        assert len(sub_samplers) == len(self.sub_samples_per_gpu), \
            'The length of sub_samplers must be equal to the number of ' \
            f'datasets, but get {len(sub_samplers)} != '\
            f'{len(self.sub_samples_per_gpu)}'
        samplers = []
        for i, sub_sampler in enumerate(sub_samplers):
            if sub_sampler is None:
                cfg = dict(
                    type='InfiniteSampler',
                    seed=self.seed,
                    shuffle=self.shuffle)
                samplers.append(
                    DATA_SAMPLERS.build(
                        cfg,
                        default_args=dict(dataset=self.dataset.datasets[i])))
            elif isinstance(sub_sampler, Sampler):
                samplers.append(sub_sampler)
            elif isinstance(sub_sampler, Mapping):
                samplers.append(
                    DATA_SAMPLERS.build(
                        sub_sampler,
                        default_args=dict(dataset=self.dataset.datasets[i])))
            else:
                raise ValueError(
                    'sub_sampler must be None or Sampler or Mapping, '
                    f'but get {type(samplers)}')
        return samplers

    def _get_aligned_len(self):
        """Calculate the alignment length when the `EpochBased` sampler is
        included, otherwise it is the actual dataset length."""
        has_epoch = False
        epoch_len = []
        for i, sampler in enumerate(self.sub_samplers):
            if hasattr(sampler, 'epoch'):
                epoch_len.append(
                    len(sampler) // self.sub_samples_per_gpu[i] *
                    self.samples_per_gpu)
                has_epoch = True
        if has_epoch:
            aligned_len = min(epoch_len)
        else:
            aligned_len = sum([len(sampler) for sampler in self.sub_samplers])
        return aligned_len

    def set_epoch(self, epoch):
        """Set the epoch for sub samplers when it is `EpochBased` sampler.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            if mmengine.has_method(sampler, 'set_epoch'):
                try:
                    sampler.set_epoch(epoch)
                except NotImplementedError:
                    pass
