_base_ = [
    './denoising-real_test_config_sidd.py'
    # './denoising-real_test_config_dnn.py'
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackEditInputs')
]

# test config
test_cfg = dict(type='EditTestLoop')
test_dataloader = [
    _base_.sidd_dataloader,
    # _base_.dnd_dataloader,
]
test_evaluator = [
    _base_.sidd_evaluator,
    # _base_.dnd_dataloader,
]