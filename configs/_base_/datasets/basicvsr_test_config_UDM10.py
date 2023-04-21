# config for UDM10 (BDx4)
udm10_data_root = 'data/UDM10'

udm10_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:04d}.png'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='PackEditInputs')
]

udm10_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='udm10', task_name='vsr'),
        data_root=udm10_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        pipeline=udm10_pipeline))

udm10_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='UDM10-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='UDM10-BDx4-Y')
]