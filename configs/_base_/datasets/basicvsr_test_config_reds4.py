# configs for REDS4
reds_data_root = 'data/REDS'

reds_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

reds_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds_reds4', task_name='vsr'),
        data_root=reds_data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        ann_file='meta_info_reds4_val.txt',
        depth=1,
        num_input_frames=100,
        fixed_seq_len=100,
        pipeline=reds_pipeline))

reds_evaluator = [
    dict(type='PSNR', prefix='REDS4-BIx4-RGB'),
    dict(type='SSIM', prefix='REDS4-BIx4-RGB')
]