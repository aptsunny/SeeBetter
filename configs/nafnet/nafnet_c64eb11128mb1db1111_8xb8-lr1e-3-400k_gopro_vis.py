_base_ = [
    './nafnet_c64eb11128mb1db1111_8xb8-lr1e-3-400k_gopro.py'
]

# _base_.visualizer.vis_backends = [
#     dict(type='GenVisBackend'),  # vis_backend for saving images to file system
#     dict(type='WandbGenVisBackend',  # vis_backend for uploading images to Wandb
#         init_kwargs=dict(
#             project='MMEditing',   # project name for Wandb
#             name='GAN-Visualization-Demo'  # name of the experiment for Wandb
#         ))
# ]

_base_.default_hooks.logger.interval=10