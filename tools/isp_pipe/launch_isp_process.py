# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import re
from glob import glob

import numpy as np
from mmengine import Config, DictAction
from PIL import Image

from mmagic.registry import MODELS


def unpack_mipi_raw10(byte_buf):
    # 转换为numpy数组，每个元素为1个字节
    data = np.frombuffer(byte_buf, dtype=np.uint8)  # (15000000,)

    # mipi压缩存储，5字节对应4个像素
    # 参考开源实现（https://github.com/taotaoli/mipi2raw/），避免循环切分和合并数据导致的低效率
    b1, b2, b3, b4, b5 = np.reshape(data, (data.shape[0] // 5,
                                           5)).astype(np.uint16).T
    p1 = (b1 << 2) + ((b5) & 0x3)
    p2 = (b2 << 2) + ((b5 >> 2) & 0x3)
    p3 = (b3 << 2) + ((b5 >> 4) & 0x3)
    p4 = (b4 << 2) + ((b5 >> 6) & 0x3)
    unpacked = np.reshape(
        np.concatenate((p1[:, None], p2[:, None], p3[:, None], p4[:, None]),
                       axis=1),
        4 * p1.shape[0],
    )

    return unpacked  # (12000000,)


def Mipi2Raw(frame_dir, imgWidth, imgHeight):
    mipiFile = glob(frame_dir + '*.RAW')[0]

    # 读取二进制文件，data以单字节保存
    mipiData = np.fromfile(mipiFile, dtype='uint8')

    bayerData = unpack_mipi_raw10(mipiData)  # (12000000,)

    # 保存为uint16，防止数据溢出
    img = bayerData.astype(np.uint16).reshape(imgHeight, imgWidth)

    # 读取mipi标准的压缩raw格式图像
    # 将数据转换至float32，防止后续步骤（主要是BLC）数据溢出
    return img.astype(np.float32)


def save_img(img, filename):
    # ensure images are normalized images
    Image.fromarray(np.uint8(np.round(np.clip(img, 0, 1) *
                                      255))).save(filename)


def load_lsc_params(frame_dir, lsc_grid_file):
    # Mesh Shading Correction
    with open(lsc_grid_file, 'r') as tf:
        lines = tf.readlines()
        # position_x:
        lsc_grid_x = list(map(int, lines[0][11:].strip().split(',')))
        # position_y:
        lsc_grid_y = list(map(int, lines[1][11:].strip().split(',')))

    lsc_map_files = [
        frame_dir + 'lsc_R_map.txt',
        frame_dir + 'lsc_G_map.txt',
        frame_dir + 'lsc_B_map.txt',
    ]

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
    lsc_map = np.stack(lsc_map_list, axis=2).astype(np.uint16)
    lsc_map = lsc_map.astype(np.float32) / lsc_map.min()

    lsc = dict(lsc_grid_x=lsc_grid_x, lsc_grid_y=lsc_grid_y, lsc_map=lsc_map)
    return lsc


def load_gains_ccm_params(frame_dir):
    with open(glob(frame_dir[:-2] + '*.txt')[0], 'r') as tf:
        text = tf.read()
        re_pattern = r'MainColorCorrectionGains\n\t \*\*\* Start Payload\*\*\*\*\n\t(.*?)\n\t \*\*\*'  # noqa: E501
        gains = list(
            map(float,
                re.search(re_pattern, text).group(1).strip().split(' ')))
        # 一般来说，白平衡中的G gain设置为1，所以此处排列应为RGGB
        fr_now, fg_now, fb_now = gains[0], (gains[1] + gains[2]) / 2, gains[3]

        re_pattern = r'MainColorCorrectionTransform\n\t \*\*\* Start Payload\*\*\*\*\n\t(.*?)\n\t \*\*\*'  # noqa: E501
        # 多行抽取，需要加上re.DOTALL
        matched_text = (
            re.search(re_pattern, text,
                      re.DOTALL).group(1).replace('\n\t', ''))
        M_cam = list(map(eval, re.findall(r'\((.*?)\)', matched_text)))
    wbc = dict(
        fr_now=fr_now,
        fg_now=fg_now,
        fb_now=fb_now,
    )
    return wbc, M_cam


def run_pipeline(cfg):
    frame_dir = cfg.frame_dir
    lsc_grid_file = cfg.lsc_grid_file
    width = cfg.get('width', 4000)
    height = cfg.get('height', 3000)
    gamma = cfg.gamma

    # load params from file
    img_raw = Mipi2Raw(frame_dir, width, height)
    wbc, M_cam = load_gains_ccm_params(frame_dir)

    if cfg.lsc_grid_file:
        lsc = load_lsc_params(frame_dir, lsc_grid_file)

    cfg.isp_pipe = dict(
        type='ISP',
        pattern=cfg.get('pattern', 'GRBG'),
        whitelevel=cfg.get('whitelevel', 64),
        blacklevel=cfg.get('blacklevel', 1023),
        img_width=width,
        img_height=height,
        lsc=lsc,
        gains=cfg.get('wbc', wbc),
        cam=cfg.get('cam', M_cam),
        gamma=gamma)

    isp_pipeline = MODELS.build(cfg.isp_pipe)

    img_mosaic, img_lsc, img_wb, img_demosaic, img_IL, img_Irgb = \
        isp_pipeline.raw2rgb(img_raw)

    if cfg.is_save_img:
        os.makedirs(cfg.save_dirs, exist_ok=True)
        filename_prefix = cfg.save_dirs + frame_dir.replace('/', '_')

        if cfg.is_save_img.input:
            save_img(img_raw / isp_pipeline.whitelevel,
                     filename_prefix + '0_img_raw.png')

        if cfg.is_save_img.blc:
            save_img(img_mosaic, filename_prefix + '1_img_mosaic.png')

        if cfg.is_save_img.lsc:
            save_img(img_lsc, filename_prefix + '2_img_lsc.png')

        if cfg.is_save_img.awb:
            save_img(img_wb, filename_prefix + '3_img_wb.png')

        if cfg.is_save_img.dem:
            save_img(img_demosaic, filename_prefix + '4_img_demosaic.png')

        if cfg.is_save_img.cmc:
            save_img(img_IL, filename_prefix + '5_img_IL.png')

        if cfg.is_save_img.gamma:
            save_img(img_Irgb, filename_prefix + '6_img_Irgb_gamma.png')

        if cfg.is_save_img.final:
            save_img(
                img_Irgb, filename_prefix +
                'Img_final_p_{}_g_{}_{}_{}_b_{}_g_{}_{}_{}_{}.png'.format(
                    isp_pipeline.pattern, isp_pipeline.gains.fr_now,
                    isp_pipeline.gains.fg_now, isp_pipeline.gains.fb_now,
                    isp_pipeline.blacklevel, isp_pipeline.gamma.value,
                    isp_pipeline.gamma.k0, isp_pipeline.gamma.phi,
                    isp_pipeline.gamma.alpha))

        if cfg.is_save_img.concat:
            img_join = np.concatenate(
                [
                    np.expand_dims(img_raw / isp_pipeline.whitelevel,
                                   axis=2).repeat(3, axis=2),
                    np.expand_dims(img_mosaic, axis=2).repeat(3, axis=2),
                    np.expand_dims(img_lsc, axis=2).repeat(3, axis=2),
                    np.expand_dims(img_wb, axis=2).repeat(3, axis=2),
                    img_demosaic,
                    img_IL,
                    img_Irgb,
                ],
                axis=0,
            )

            save_img(
                img_join, filename_prefix + 'img_pipeline_gamma_{}.png'.format(
                    str(isp_pipeline.gamma.value)))


def parse_args():
    parser = argparse.ArgumentParser(description='Print the whole config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    run_pipeline(cfg)


if __name__ == '__main__':
    main()
