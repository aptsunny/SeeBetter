custom_imports = dict(imports='projects.isp_pipeline')

height = 3000
width = 4000
frame_dir = 'work_dirs/Data/S0/0/'
lsc_grid_file = 'work_dirs/Data/lsc_grid.txt'

# setting
# is_save_img = True
use_c_plugin = False
is_save_img = dict(
    save_dirs='work_dirs/results/',
    input=False,
    blc=False,
    lsc=False,
    awb=False,
    dem=False,
    cmc=False,
    gamma=False,
    final=True,
    concat=False)

# exp3
# pattern = 'RGGB'
# pattern = 'BGGR'
pattern = 'GRBG'
# pattern = 'GBRG'

# 1 exp blc
# blc
blacklevel = 64
whitelevel = 2**10 - 1
# python ISP_implement.py isp_cfg.py --cfg-options blacklevel=32

# 2 exp wbc
# (1.35449219, 1.0, 2.48828125)
wbc = dict(fr_now=1.35449219, fg_now=1.0, fb_now=2.48828125)

# wbc = dict(
#     fr_now=2.7,
#     fg_now=1.0,
#     fb_now=2.48828125)

# wbc = dict(
#     fr_now=1.35449219,
#     fg_now=2.0,
#     fb_now=2.48828125)

# wbc = dict(
#     fr_now=1.35449219,
#     fg_now=1.0,
#     fb_now=4.88828125)

# (1.35449219, 1.0, 2.48828125)

# exp4 cam

cam = dict(diagonal=0, extra=0)

# cam = dict(
#     diagonal=32,
#     extra=-16
# )

# cam = dict(
#     diagonal=-32,
#     extra=16
# )

# exp5 gamma
gamma = dict(value=2.22222, k0=0.081, phi=4.5, alpha=0.099)

# gamma = dict(
#     value=2.22222,
#     k0 = 0.081,
#     phi = 2.0,
#     alpha = 0.099
# )

# gamma = dict(
#     value=2.22222,
#     k0 = 0.081,
#     phi = 18.0,
#     alpha = 0.099
# )
# gamma = 2.22222

# gamma = dict(
#     value=4,
#     k0 = 0.081,
#     phi = 4.5,
#     alpha = 0.099
# )

# gamma = dict(
#     value=1.53,
#     k0 = 0.081,
#     phi = 4.5,
#     alpha = 0.099
# )
