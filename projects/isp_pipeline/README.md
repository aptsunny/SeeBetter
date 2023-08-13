# Image Signal Processor Pipeline

作为 `SeeBetter` 工具链 的一个 ISP 项目，尽可能将 ISP 各个算法组件拆解，由于在进行 Noise Reduction 相关的算法开发时，也会遇到 ISP 的处理，因此后续可以更多地整合进 `Transform` 中去，便于 `Raw2RGB` 相关算法的开发。

现阶段都是在 `projects/isp_pipeline/isp.py` 的 `ISP` 类中简单的实现。

## Implementations

1. Bayer Domain

- [x] Black Level Compensation
- [ ] Dead Pixel Correction
- [ ] Noise Reduction (Bayer Domain)
  - [ ] Luma Denoising
  - [ ] Chroma Denoising
- [x] Lens Shading Correction
- [ ] Anti-aliasing Noise Filter
- [x] AWB Gain Control
- [x] Demosaicing

2. RGB Domain

- [x] Color Correction Matrix
- [x] Gamma Correction
- [ ] Color Space Conversion

3. YUV Domain

- [ ] Noise Filter for Luma/Chroma (YUVNR)
  - [ ] Luma Noise Reduction
    - [ ] Bilateral Filtering
    - [ ] Non-local Means Denoising
  - [ ] Chroma Noise Reduction
- [ ] Edge Enhancement
- [ ] False Color Suppression
- [ ] Hue/Saturation Control
- [ ] Brightness/Contrast Control

Except:

- Anti-aliasing Noise Filter

## Usage

```shell
./tools/check_isp.sh projects/isp_pipeline/isp_vis_config.py --cfg-options blacklevel=32
```

为了分析各个模块的参数对于出图造成的影响，将各个模型的超参数可以通过命令修改;

- `blacklevel`
- `wbc`
- `cam`
