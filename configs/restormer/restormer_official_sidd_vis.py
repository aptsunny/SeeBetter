_base_ = [
    './restormer_official_sidd.py'
]

_base_.default_hooks.logger.interval=5
_base_.custom_hooks[0].interval=1000

# normal vis
_base_.visualizer.vis_backends = [
    dict(type='GenVisBackend'),  # vis_backend for saving images to file system
]

# wandb vis
"""
_base_.visualizer.vis_backends = [
    dict(type='WandbGenVisBackend',  # vis_backend for uploading images to Wandb
        init_kwargs=dict(
            project='MMEditing',   # project name for Wandb
            name='GAN-Visualization-Demo'  # name of the experiment for Wandb
        ))
]
"""

# full wandb vis config
"""
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=[
        dict(type='GenVisBackend'),
        dict(
            type='WandbGenVisBackend',
            init_kwargs=dict(
                project='MMEditing', name='GAN-Visualization-Demo'))
    ],
    fn_key='gt_path',
    img_keys=['gt_img', 'input', 'pred_img'],
    bgr2rgb=False)
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]
"""