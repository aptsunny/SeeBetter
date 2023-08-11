import ctypes

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
                 lsc_grid_x, lsc_grid_y, lsc_map_files, fr_now, fg_now, fb_now,
                 M_cam, use_c_plugin, gamma, k0, phi, alpha):
        self.pattern = pattern
        self.whitelevel, self.blacklevel = whitelevel, blacklevel
        self.lsc_grid_x = lsc_grid_x.copy()
        self.lsc_grid_x.insert(0, 0)
        self.lsc_grid_x.append(img_width)
        self.lsc_grid_y = lsc_grid_y.copy()
        self.lsc_grid_y.insert(0, 0)
        self.lsc_grid_y.append(img_height)

        lsc_map_list = []
        for lsc_map_file in lsc_map_files:
            with open(lsc_map_file, 'r') as tf:
                lines = tf.readlines()

                lsc_map_channel = []
                for line in lines:
                    lsc_map_channel.append(
                        list(map(int,
                                 line.strip().split('\t'))))

            lsc_map_channel = np.array(lsc_map_channel)
            lsc_map_channel_pad = np.pad(lsc_map_channel, 1, mode='reflect')

            lsc_map_list.append(lsc_map_channel_pad)

        self.lsc_map = np.stack(lsc_map_list, axis=2).astype(np.uint16)

        self.fr_now = fr_now
        self.fg_now = fg_now
        self.fb_now = fb_now

        self.M_cam = M_cam

        self.use_c_plugin = use_c_plugin

        self.gamma = gamma
        self.k0 = k0
        self.phi = phi
        self.alpha = alpha

    def black_level_correction(self, img, blacklevel, whitelevel):
        """black level correction."""
        img = (img - blacklevel) / (whitelevel - blacklevel)

        # 每一步进行clip可以方便数据处理，但是可能会有信息丢失的风险
        img = np.clip(img, 0, 1)
        return img

    def lens_shading_correction(self, img, use_c_plugin):
        """lens shading correction."""
        if use_c_plugin:
            LSC = ctypes.CDLL('./modules/libs/lens_shading_correction.dll')

            # 定点化实现，重新把数据变回uint16
            img = np.round(img * (self.whitelevel - self.blacklevel)).astype(
                np.uint16)
            img_c = img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            lsc_map_c = self.lsc_map.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16))

            lsc_grid_x = np.array(self.lsc_grid_x)
            lsc_grid_y = np.array(self.lsc_grid_y)
            lsc_grid_x_c = lsc_grid_x.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32))
            lsc_grid_y_c = lsc_grid_y.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int32))

            LSC.lens_shading_correction(
                img_c,
                lsc_map_c,
                lsc_grid_x_c,
                lsc_grid_y_c,
                img.shape[1],
                img.shape[0],
                len(self.lsc_grid_x),
                len(self.lsc_grid_y),
                ['GRBG', 'RGGB', 'BGGR', 'GBRG'].index(self.pattern),
            )

            # 定点化实现，重新归一化
            img = img / (self.whitelevel - self.blacklevel)

        else:
            lsc_map = self.lsc_map.astype(np.float32) / self.lsc_map.min()

            lsc_grid_x = np.array(self.lsc_grid_x)
            lsc_grid_y = np.array(self.lsc_grid_y)

            img_height, img_width = img.shape
            lsc_height, lsc_width, _ = lsc_map.shape

            y_index = 0

            for img_y in range(img_height):
                x_index = 0

                if img_y > lsc_grid_y[y_index +
                                      1] and y_index < lsc_height - 2:
                    y_index += 1

                for img_x in range(img_width):
                    if img_x > lsc_grid_x[x_index +
                                          1] and x_index < lsc_width - 2:
                        x_index += 1

                    # 根据pattern选择正确的通道
                    raw_index = (img_y % 2) << 1 | (img_x % 2)
                    channel = self.pattern_mapping[self.pattern][raw_index]

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
                    bottom_value = bottom_left * (
                        1.0 - float_x) + bottom_right * float_x
                    top_value = top_left * (1.0 -
                                            float_x) + top_right * float_x
                    result = bottom_value * (1.0 -
                                             float_y) + top_value * float_y

                    # 对图像进行增益处理
                    img[img_y, img_x] *= result

        img = np.clip(img, 0, 1)
        return img

    def white_balance(self, img, fr_now, fg_now, fb_now):
        """
        White Balance 3000 * 4000
        """
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

    def color_matrix_correction(self, img, M_cam):
        M_cam = np.reshape(M_cam, (3, 3))
        img = np.matmul(img, M_cam.T)
        img = np.clip(img, 0, 1)
        return img

    def gamma_correction(self, img, gamma, k0, phi, alpha):
        """# BT.601标准的gamma变换，ref https://www.nmm-
        hd.org/newbbs/viewtopic.php?t=1286."""
        img = np.where(
            img <= np.ones_like(img) * k0 / phi,
            img * phi,
            (img**(1 / gamma)) * (1 + alpha) - alpha,
        )

        img = np.clip(img, 0, 1)
        return img

    def raw2rgb(self, img_raw):
        img_mosaic = self.black_level_correction(img_raw, self.blacklevel,
                                                 self.whitelevel)
        img_lsc = self.lens_shading_correction(img_mosaic, self.use_c_plugin)
        img_wb = self.white_balance(img_lsc, self.fr_now, self.fg_now,
                                    self.fb_now)
        img_demosaic = self.demosaic(img_wb, self.pattern)
        img_IL = self.color_matrix_correction(img_demosaic, self.M_cam)
        img_Irgb = self.gamma_correction(img_IL, self.gamma, self.k0, self.phi,
                                         self.alpha)
        return img_mosaic, img_lsc, img_wb, img_demosaic, img_IL, img_Irgb
