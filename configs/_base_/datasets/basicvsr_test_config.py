_base_ = [
    # './basicvsr_test_config_reds4.py',
    # './basicvsr_test_config_vimeo90k.py',
    # './basicvsr_test_config_UDM10.py',
    './basicvsr_test_config_vid4.py'
]

# config for test
test_cfg = dict(type='EditTestLoop')
test_dataloader = [
    # _base_.reds_dataloader,
    # _base_.vimeo_90k_bd_dataloader,
    # _base_.vimeo_90k_bi_dataloader,
    # _base_.udm10_dataloader,
    _base_.vid4_bd_dataloader,
    _base_.vid4_bi_dataloader,
]
test_evaluator = [
    # _base_.reds_evaluator,
    # _base_.vimeo_90k_bd_evaluator,
    # _base_.vimeo_90k_bi_evaluator,
    # _base_.udm10_evaluator,
    _base_.vid4_bd_evaluator,
    _base_.vid4_bi_evaluator,
]
