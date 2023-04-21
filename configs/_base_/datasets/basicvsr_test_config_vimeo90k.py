# configs for vimeo90k-bd and vimeo90k-bi
vimeo_90k_data_root = 'data/vimeo90k'
vimeo_90k_file_list = [
    'im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png', 'im7.png'
]

vimeo_90k_pipeline = [
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='MirrorSequence', keys=['img']),
    dict(type='PackEditInputs')
]

vimeo_90k_bd_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=vimeo_90k_data_root,
        data_prefix=dict(img='BDx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_test_GT.txt',
        depth=2,
        num_input_frames=7,
        fixed_seq_len=7,
        load_frames_list=dict(img=vimeo_90k_file_list, gt=['im4.png']),
        pipeline=vimeo_90k_pipeline))

vimeo_90k_bi_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='vimeo90k_seq', task_name='vsr'),
        data_root=vimeo_90k_data_root,
        data_prefix=dict(img='BIx4', gt='GT'),
        ann_file='meta_info_Vimeo90K_test_GT.txt',
        depth=2,
        num_input_frames=7,
        fixed_seq_len=7,
        load_frames_list=dict(img=vimeo_90k_file_list, gt=['im4.png']),
        pipeline=vimeo_90k_pipeline))

vimeo_90k_bd_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='Vimeo-90K-T-BDx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='Vimeo-90K-T-BDx4-Y'),
]

vimeo_90k_bi_evaluator = [
    dict(type='PSNR', convert_to='Y', prefix='Vimeo-90K-T-BIx4-Y'),
    dict(type='SSIM', convert_to='Y', prefix='Vimeo-90K-T-BIx4-Y'),
]