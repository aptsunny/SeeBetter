import numpy as np

from mmagic.registry import MODELS
from .bayer_domain.demosaicing import demosaicing_CFA_Bayer_Malvar2004


@MODELS.register_module()
class ISP:
    pattern_mapping = {
        'GRBG': {
            0: 1,
            1: 0,
            2: 2,
            3: 1
        },
        'RGGB': {
            0: 0,
            1: 1,
            2: 1,
            3: 2
        },
        'BGGR': {
            0: 2,
            1: 1,
            2: 1,
            3: 0
        },
        'GBRG': {
            0: 1,
            1: 2,
            2: 0,
            3: 1
        }
    }

    def __init__(self, pattern, whitelevel, blacklevel, img_width, img_height,
                 lsc, gains, cam, gamma):
        self.pattern = pattern
        self.whitelevel = whitelevel
        self.blacklevel = blacklevel
        self.img_width = img_width
        self.img_height = img_height
        self.lsc = lsc
        self.gains = gains
        self.cam = cam
        self.gamma = gamma

    def black_level_correction(self, img, blacklevel, whitelevel):
        """black level correction.

        Args:
            blacklevel (int): default to 64.
            whitelevel (int): default to 1023, if 10 bit.
        """

        # After subtracting blacklevel, ensure that the maximum value is 1
        img = (img - blacklevel) / (whitelevel - blacklevel)

        img = np.clip(img, 0, 1)
        return img

    def lens_shading_correction(self, img, pattern, lsc):
        """lens shading correction."""
        lsc_map = lsc['lsc_map']
        lsc_grid_x = lsc['lsc_grid_x']
        lsc_grid_y = lsc['lsc_grid_y']

        lsc_height, lsc_width, _ = lsc_map.shape

        img_height, img_width = img.shape
        lsc_grid_x = lsc_grid_x.copy()
        lsc_grid_x.insert(0, 0)
        lsc_grid_x.append(self.img_width)
        lsc_grid_x = np.array(lsc_grid_x)
        lsc_grid_y = lsc_grid_y.copy()
        lsc_grid_y.insert(0, 0)
        lsc_grid_y.append(self.img_height)
        lsc_grid_y = np.array(lsc_grid_y)

        y_index = 0
        for img_y in range(img_height):
            x_index = 0
            if img_y > lsc_grid_y[y_index + 1] and y_index < lsc_height - 2:
                y_index += 1
            for img_x in range(img_width):
                if img_x > lsc_grid_x[x_index + 1] and x_index < lsc_width - 2:
                    x_index += 1
                # 根据pattern选择正确的通道
                raw_index = (img_y % 2) << 1 | (img_x % 2)
                channel = self.pattern_mapping[pattern][raw_index]
                # 计算插值权重
                float_x_k = lsc_grid_x[x_index + 1] - lsc_grid_x[x_index]
                float_x = (img_x - lsc_grid_x[x_index]) / float_x_k
                float_y_k = lsc_grid_y[y_index + 1] - lsc_grid_y[y_index]
                float_y = (img_y - lsc_grid_y[y_index]) / float_y_k
                # 四个相邻grid的坐标
                bottom_left = lsc_map[y_index, x_index, channel]
                bottom_right = lsc_map[y_index, x_index + 1, channel]
                top_left = lsc_map[y_index + 1, x_index, channel]
                top_right = lsc_map[y_index + 1, x_index + 1, channel]
                # 执行双线性插值
                bottom_value = bottom_left * (1.0 -
                                              float_x) + bottom_right * float_x
                top_value = top_left * (1.0 - float_x) + top_right * float_x
                result = bottom_value * (1.0 - float_y) + top_value * float_y
                # 对图像进行增益处理
                img[img_y, img_x] *= result
        img = np.clip(img, 0, 1)
        return img

    def white_balance(self, img, gains):
        """
        White Balance 3000 * 4000
        """
        fr_now = gains.fr_now
        fg_now = gains.fg_now
        fb_now = gains.fb_now
        wb_mask = np.ones(img.shape) * fg_now
        if self.pattern == 'RGGB':
            wb_mask[0::2, 0::2] = fr_now / fg_now
            wb_mask[1::2, 1::2] = fb_now / fg_now
        elif self.pattern == 'BGGR':
            wb_mask[1::2, 1::2] = fr_now / fg_now
            wb_mask[0::2, 0::2] = fb_now / fg_now
        elif self.pattern == 'GRBG':
            wb_mask[0::2, 1::2] = fr_now / fg_now
            wb_mask[1::2, 0::2] = fb_now / fg_now
        elif self.pattern == 'GBRG':
            wb_mask[1::2, 0::2] = fr_now / fg_now
            wb_mask[0::2, 1::2] = fb_now / fg_now

        img = img * wb_mask
        img = np.clip(img, 0, 1)
        return img

    def demosaic(self, img, pattern):
        """# Demosaic (3000, 4000, 3)"""
        img = demosaicing_CFA_Bayer_Malvar2004(img, pattern)
        img = np.clip(img, 0, 1)
        return img

    def color_matrix_correction(self, img, cam):
        cam = np.reshape(cam, (3, 3))
        img = np.matmul(img, cam.T)
        img = np.clip(img, 0, 1)
        return img

    def gamma_correction(self, img, gamma):
        """# BT.601标准的gamma变换，ref https://www.nmm-
        hd.org/newbbs/viewtopic.php?t=1286."""
        img = np.where(
            img <= np.ones_like(img) * gamma.k0 / gamma.phi,
            img * gamma.phi,
            (img**(1 / gamma.value)) * (1 + gamma.alpha) - gamma.alpha,
        )

        img = np.clip(img, 0, 1)
        return img

    def raw2rgb(self, img_raw):
        img_mosaic = self.black_level_correction(img_raw, self.blacklevel,
                                                 self.whitelevel)
        img_lsc = self.lens_shading_correction(img_mosaic, self.pattern,
                                               self.lsc)
        img_wb = self.white_balance(img_lsc, self.gains)
        img_demosaic = self.demosaic(img_wb, self.pattern)
        img_IL = self.color_matrix_correction(img_demosaic, self.cam)
        img_Irgb = self.gamma_correction(img_IL, self.gamma)
        return img_mosaic, img_lsc, img_wb, img_demosaic, img_IL, img_Irgb
