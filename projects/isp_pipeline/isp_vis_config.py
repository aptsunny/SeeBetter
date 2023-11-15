custom_imports = dict(imports='projects.isp_pipeline')

# setting
frame_dir = 'work_dirs/Data/S0/0/'
save_dirs = 'work_dirs/results/'
is_save_img = dict(
    input=False,
    blc=False,
    lsc=False,
    awb=False,
    dem=False,
    cmc=False,
    gamma=False,
    final=True,
    concat=False)

# 1. Bayer Domain
# Black Level Compensation
blacklevel = 64
whitelevel = 2**10 - 1

# Dead Pixel Correction
# Noise Reduction (Bayer Domain)

# Lens Shading Correction
# Mesh Shading Correction file
lsc_grid_file = 'work_dirs/Data/lsc_grid.txt'

# Anti-aliasing Noise Filter

# AWB Gain Control
wbc = dict(fr_now=1.35449219, fg_now=1.0, fb_now=2.48828125)
# wbc = dict(fr_now=2.7, fg_now=1.0, fb_now=2.48828125)
# wbc = dict(fr_now=1.35449219, fg_now=2.0, fb_now=2.48828125)
# wbc = dict(fr_now=1.35449219, fg_now=1.0, fb_now=4.88828125)

# Demosaicing
# pattern = 'RGGB'
# pattern = 'BGGR'
pattern = 'GRBG'
# pattern = 'GBRG'

# 2. RGB Domain
# Color Correction Matrix
cam = dict(diagonal=0, extra=0)
# cam = dict(diagonal=32, extra=-16)
# cam = dict(diagonal=-32, extra=16)

# Gamma Correction
gamma = dict(value=2.22222, k0=0.081, phi=4.5, alpha=0.099)
# gamma = dict(value=2.22222, k0=0.081, phi=2.0, alpha=0.099)
# gamma = dict(value=2.22222, k0=0.081, phi=18.0, alpha=0.099)
# gamma = dict(value=4, k0=0.081, phi=4.5, alpha=0.099)
# gamma = dict(value=1.53, k0= 0.081, phi=4.5, alpha=0.099)

# Color Space Conversion

# 3. YUV Domain
# Noise Filter for Luma/Chroma (YUVNR)
