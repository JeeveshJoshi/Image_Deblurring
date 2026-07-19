<div align="center">

# Hybrid Vision-Transformer GAN for Single-Image Deblurring

**A Transformer-generator / PatchGAN-discriminator hybrid that removes motion blur — outperforming DeblurGAN-v2, Restormer, and the FFT-ReLU sparsity prior on average, under a matched 4&nbsp;GB-VRAM training budget.**

MSc Data Science Dissertation · Heriot-Watt University · 2025

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E?logoColor=black)
![Task](https://img.shields.io/badge/Task-Image%20Deblurring-8A2BE2)

</div>

---

## Overview

Motion blur destroys the high-frequency detail that downstream vision tasks depend on. This project introduces a **hybrid deblurring model** that pairs a **Vision-Transformer generator** — for global context and long-range structure — with a lightweight **PatchGAN CNN discriminator** for local realism, trained under a three-part objective: a **Charbonnier** pixel loss, a **ViT-based perceptual loss**, and an **adversarial** loss.

Crucially, every model in this study — the proposed one *and* all three baselines — was trained under an **identical 4&nbsp;GB-VRAM budget** on synthetically blurred **BSD300**. The comparison therefore isolates *architecture*, not compute. Under that controlled setting, the proposed model records the best average PSNR, SSIM, and LPIPS of the four.

## Results

<table>
  <tr>
    <th>Blurred input</th>
    <th>Deblurred&nbsp;(Ours)</th>
    <th>Ground truth</th>
  </tr>
  <tr>
    <td><img src="Additional%20Testing%20Outputs/snow_blur.png"   width="240" alt="Snow — blurred"></td>
    <td><img src="Additional%20Testing%20Outputs/snow_deblur.png" width="240" alt="Snow — deblurred"></td>
    <td><img src="Additional%20Testing%20Outputs/snow_sharp.png"  width="240" alt="Snow — ground truth"></td>
  </tr>
  <tr>
    <td><img src="Additional%20Testing%20Outputs/tree_blur.png"   width="240" alt="Tree — blurred"></td>
    <td><img src="Additional%20Testing%20Outputs/tree_deblur.png" width="240" alt="Tree — deblurred"></td>
    <td><img src="Additional%20Testing%20Outputs/tree_sharp.png"  width="240" alt="Tree — ground truth"></td>
  </tr>
  <tr>
    <td><img src="Additional%20Testing%20Outputs/boat_blur.png"   width="240" alt="Boat — blurred"></td>
    <td><img src="Additional%20Testing%20Outputs/boat_deblur.png" width="240" alt="Boat — deblurred"></td>
    <td><img src="Additional%20Testing%20Outputs/boat_sharp.png"  width="240" alt="Boat — ground truth"></td>
  </tr>
  <tr>
    <td><img src="Additional%20Testing%20Outputs/eagle_blur.png"   width="240" alt="Eagle — blurred"></td>
    <td><img src="Additional%20Testing%20Outputs/eagle_deblur.png" width="240" alt="Eagle — deblurred"></td>
    <td><img src="Additional%20Testing%20Outputs/eagle_sharp.png"  width="240" alt="Eagle — ground truth"></td>
  </tr>
</table>

<sub>More qualitative examples in <a href="Additional%20Testing%20Outputs">Additional Testing Outputs/</a>.</sub>

### Quantitative comparison

Averaged over the BSD300 test split — PSNR and SSIM: higher is better; LPIPS: lower is better.

| Method | PSNR&nbsp;↑ | SSIM&nbsp;↑ | LPIPS&nbsp;↓ |
|---|:---:|:---:|:---:|
| DeblurGAN-v2 <sub>(Kupyn et al., ICCV 2019)</sub> | 20.52 | 0.2891 | 0.7081 |
| FFT-ReLU Sparsity Prior <sub>(Al Radi et al., WACV 2025)</sub> | 21.10 | 0.3648 | 0.5389 |
| Restormer <sub>(Zamir et al., CVPR 2022)</sub> | 20.64 | 0.3143 | 0.7265 |
| **Proposed Hybrid ViT-GAN** | **22.03** | **0.5248** | **0.5357** |

The proposed model leads on all three metrics on average, with the largest margin in **SSIM** (0.52 vs 0.29–0.36). Absolute PSNR is modest relative to published figures because every model shares the constrained 4&nbsp;GB / BSD300 budget; individual images vary (some baselines win on particular scenes), so the accurate claim is **best average performance under matched constraints** rather than universal superiority.

## Method

### Generator — Vision Transformer
- **Overlapping patch embedding** (16×16 patches, stride 8) → 768-d tokens + positional encoding
- **3 Transformer blocks** (8 attention heads, MLP dim 3072) capture global structure
- Tokens are reshaped to a spatial feature map and passed through a **convolutional decoder** (stride-2 down-sampling ×2, transpose-conv up-sampling ×3) with a `tanh` output head

### Discriminator — PatchGAN
- 5-layer strided CNN (64 → 128 → 256 → 512 → 1), LeakyReLU + BatchNorm, producing a patch-wise real/fake map for local texture realism

### Objective

```
L_G  =  L_Charbonnier  +  L_ViT-perceptual  +  0.01 · L_adversarial
```

- **Charbonnier** — a robust, L1-style pixel-fidelity loss
- **ViT perceptual** — feature distance computed from a frozen `google/vit-base-patch16-224` (encoder layer 5); using a Vision Transformer here in place of the conventional VGG perceptual loss is the core design contribution
- **Adversarial** — BCE-with-logits signal from the PatchGAN discriminator

### Ablations
The dissertation ablates each component (Transformer depth, ViT-perceptual loss, adversarial loss) and measures its contribution. Notably, removing the adversarial term *raises* PSNR but *degrades* LPIPS — the expected pixel-vs-perceptual trade-off, and evidence that the components pull in complementary directions.

## Dataset
Sharp source images come from **BSD300**. Synthetic **motion blur** (varied kernel size and angle) is applied via [`Scripts/deblurScript.ipynb`](Scripts/deblurScript.ipynb) to produce paired `BlurredData` / `CleanData` under `Dataset/ProcessedDataset/`, split 90/10 (seed 42).

BSD300 can be downloaded from the [Berkeley Segmentation Dataset page](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/); run the data-prep notebook to regenerate the blurred/clean pairs.

## Getting started

```bash
git clone https://github.com/JeeveshJoshi/Image_Deblurring.git
cd Image_Deblurring/Code/ProposedHybridModel
pip install torch torchvision transformers pillow tqdm
```

**1 · Prepare data** — run [`Scripts/deblurScript.ipynb`](Scripts/deblurScript.ipynb) to synthesise blurred/clean pairs from BSD300 (or point the loader at your own paired data).

**2 · Train**

```bash
python main.py --blurred-dir <path/to/BlurredData> --clean-dir <path/to/CleanData>
```
Defaults: 256×256 inputs, batch 20, 25 epochs, Adam @ 2e-4. Checkpoints are written to `checkpoints/` every 10 epochs.

**3 · Evaluate** — [`Code/testMetricCalculation.ipynb`](Code/testMetricCalculation.ipynb) computes PSNR / SSIM / LPIPS against the baselines.

## Repository structure

```
Code/
├─ ProposedHybridModel/     ← the proposed model
│  ├─ models/               ← generator.py, discriminator.py
│  ├─ losses/               ← charbonnier_loss.py, vit_perceptual_loss.py
│  ├─ components/           ← vit_modules.py (patch embedding, transformer blocks)
│  └─ main.py               ← training entry point
├─ DeblurGANv2/             ← baseline notebook
└─ testMetricCalculation.ipynb
Scripts/deblurScript.ipynb  ← synthetic-blur data generation
Additional Testing Outputs/ ← qualitative results
comparison.xlsx             ← quantitative results
```

## Acknowledgements
This project benchmarks against, and is indebted to, three prior works:
- **DeblurGAN-v2** — Kupyn et al., *ICCV 2019*
- **Restormer** — Zamir et al., *CVPR 2022*
- **FFT-ReLU Sparsity Prior** — Al Radi et al., *WACV 2025*

## Author
**Jeevesh Joshi** — MSc Data Science, Heriot-Watt University, Edinburgh (2024–2025)

<!-- Add your links once ready: -->
Connect on LinkedIn · more work on my portfolio *(links to be added)*

## License
Released under the MIT License — see [`LICENSE`](LICENSE).
