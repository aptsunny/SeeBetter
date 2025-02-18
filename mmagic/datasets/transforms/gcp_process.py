# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


def apply_gains_jdd(bayer_images, red_gains, blue_gains):
    """Applies white balance gains to a batch of Bayer images."""
    if red_gains.clone().detach().shape[0] == bayer_images.shape[0]:
        red_gains = red_gains.clone().detach()
        blue_gains = blue_gains.clone().detach()
    else:
        red_gains = torch.tensor(([[red_gains]]))
        blue_gains = torch.tensor(([[blue_gains]]))
    # Permute the image tensor to BxHxWxC format from BxCxHxW format
    bayer_images = bayer_images.permute(0, 2, 3, 1)
    green_gains = torch.ones_like(red_gains)
    gains = torch.cat([red_gains, green_gains, blue_gains], dim=-1)
    gains = gains[:, None, None, :]
    # outs  = bayer_images * gains
    outs = bayer_images
    outs[:, :, :, 0] = outs[:, :, :, 0] * gains[:, :, :, 0]
    outs[:, :, :, 1] = outs[:, :, :, 1] * gains[:, :, :, 1]
    outs[:, :, :, 2] = outs[:, :, :, 2] * gains[:, :, :, 2]
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images.permute(0, 2, 3, 1)
    images = images[:, :, :, None, :]
    if ccms.shape != (3, 3):
        ccms = ccms[:, None, None, :, :]
    else:
        ccms = ccms[None, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images.permute(0, 2, 3, 1)
    outs = torch.clamp(images, min=1e-8)**(1.0 / gamma)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def process_train(bayer_images, red_gains, blue_gains, cam2rgbs):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    bayer_images = apply_gains_jdd(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = bayer_images
    # images = demosaic(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images)
    # images = smoothstep(images)
    return images


def process(bayer_images, red_gains, blue_gains, cam2rgbs):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = demosaic(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images)
    return images


def process_test(bayer_images, red_gains, blue_gains, cam2rgbs):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    bayer_images = apply_gains_jdd(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = bayer_images
    # images = demosaic(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images)
    images_show = smoothstep(images)
    return images, images_show


def apply_gains(bayer_images, red_gains, blue_gains):
    """Applies white balance gains to a batch of Bayer images."""
    red_gains = red_gains.squeeze(1)
    blue_gains = blue_gains.squeeze(1)
    # Permute the image tensor to BxHxWxC format from BxCxHxW format
    bayer_images = bayer_images.permute(0, 2, 3, 1)
    green_gains = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, green_gains, blue_gains],
                        dim=-1)
    gains = gains[:, None, None, :]
    outs = bayer_images * gains
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def demosaic(bayer_images):

    def SpaceToDepth_fact2(x):
        bs = 2
        N, C, H, W = x.size()
        # (N, C, H//bs, bs, W//bs, bs)
        x = x.view(N, C, H // bs, bs, W // bs, bs)
        # (N, bs, bs, C, H//bs, W//bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        # (N, C*bs^2, H//bs, W//bs)
        x = x.view(N, C * (bs**2), H // bs, W // bs)
        return x

    def DepthToSpace_fact2(x):
        bs = 2
        N, C, H, W = x.size()
        # (N, bs, bs, C//bs^2, H, W)
        x = x.view(N, bs, bs, C // (bs**2), H, W)
        # (N, C//bs^2, H, bs, W, bs)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        # (N, C//bs^2, H * bs, W * bs)
        x = x.view(N, C // (bs**2), H * bs, W * bs)
        return x

    """Bilinearly demosaics a batch of RGGB Bayer images."""
    # Permute the image tensor to BxHxWxC format from BxCxHxW format
    bayer_images = bayer_images.permute(0, 2, 3, 1)

    shape = bayer_images.size()
    shape = [shape[1] * 2, shape[2] * 2]

    red = bayer_images[Ellipsis, 0:1]
    upsamplebyX = nn.Upsample(size=shape, mode='bilinear', align_corners=False)
    red = upsamplebyX(red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_red = bayer_images[Ellipsis, 1:2]
    green_red = torch.flip(green_red, dims=[1])  # Flip left-right
    green_red = upsamplebyX(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    green_red = torch.flip(green_red, dims=[1])  # Flip left-right
    green_red = SpaceToDepth_fact2(green_red.permute(0, 3, 1,
                                                     2)).permute(0, 2, 3, 1)

    green_blue = bayer_images[Ellipsis, 2:3]
    green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down
    green_blue = upsamplebyX(green_blue.permute(0, 3, 1,
                                                2)).permute(0, 2, 3, 1)
    green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down
    green_blue = SpaceToDepth_fact2(green_blue.permute(0, 3, 1,
                                                       2)).permute(0, 2, 3, 1)

    green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
    green_at_green_red = green_red[Ellipsis, 1]
    green_at_green_blue = green_blue[Ellipsis, 2]
    green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = DepthToSpace_fact2(
        torch.stack(green_planes, dim=-1).permute(0, 3, 1,
                                                  2)).permute(0, 2, 3, 1)

    blue = bayer_images[Ellipsis, 3:4]
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
    blue = upsamplebyX(blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

    rgb_images = torch.cat([red, green, blue], dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    rgb_images = rgb_images.permute(0, 3, 1, 2)
    return rgb_images


def smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    temp = torch.mul(image, image)
    out = 3.0 * temp - 2.0 * torch.mul(temp, image)
    return out
