dnd_data_root = 'data/DND'
dnd_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='DND', task_name='denoising'),
        data_root=dnd_data_root,
        data_prefix=dict(img='input', gt='groundtruth'),
        pipeline=test_pipeline))
dnd_evaluator = [
    dict(type='PSNR', prefix='DND'),
    dict(type='SSIM', prefix='DND'),
]