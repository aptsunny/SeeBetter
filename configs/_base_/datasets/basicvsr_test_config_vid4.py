# config for vid4
vid4_data_root = 'data/Vid4'

vid4_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]
vid4_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        pipeline=vid4_pipeline))

vid4_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vid4', task_name='vsr'),
        data_root=vid4_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_Vid4_GT.txt',
        depth=1,
        pipeline=vid4_pipeline))

vid4_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='VID4-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='VID4-BDx4-Y'),
]
vid4_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='VID4-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='VID4-BIx4-Y'),
]