sidd_data_root = 'data/SIDD'
sidd_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='SIDD', task_name='denoising'),
        data_root=sidd_data_root,
        data_prefix=dict(img='test/noisy', gt='test/gt'),
        pipeline=test_pipeline))
sidd_evaluator = [
    dict(type='PSNR', prefix='SIDD'),
    dict(type='SSIM', prefix='SIDD'),
]
