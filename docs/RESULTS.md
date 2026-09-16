# Recorded results and limitations

These values come from saved experiments, not new training performed when writing this documentation. Run identifiers keep scores tied to the relevant configuration. Follow each folder link for configuration, history and figures.

| Experiment | Recorded result | Evidence |
|---|---|---|
| LFW PCA + Random Forest | 62.42% test accuracy | [Notebook](../lab2.ipynb) saved output |
| LFW CNN | 83.85% saved test accuracy | [Notebook](../lab2.ipynb); rerun to verify current source |
| CIFAR-10 ResNet-18 | 94.06% test accuracy; 192.67 s total training; selected epoch 90 | [Run](../part3_results/cifar10_20260916_114411_236247/) |
| VAE | Test MSE 0.00507945; combined loss 0.00535722; selected epoch 17 | [Run](../part4_results/vae_train_20260915_004541_802297/) |
| U-Net | Pooled Dice 0.99952 / 0.96754 / 0.96499 / 0.97857 (raw labels 0/85/170/255) | [Run](../part4_results/unet_train_20260915_005104_064230/) |
| WGAN-GP | 64 generated images; mean pixel std 0.05491; pairwise MSE at 32×32 0.008226 | [Run](../part4_results/gan_wgan_gp_train_20260916_122956_496407/) |

## Interpretation

ResNet timing includes GPU cache setup, training, validation and checkpoint saves; it excludes downloads, model setup, final testing and plots. The A100 result meets the local 94%/360 s comparison, but is not a certified DAWNBench submission. Validation-based selection and final test evaluation must not be confused with per-epoch test benchmark timing.

VAE evaluation uses posterior-mean reconstructions. It is not a sampled ELBO estimate; KL is divided by pixel count in the actual objective. Inspect the [reconstructions](../part4_results/vae_train_20260915_004541_802297/reconstructions.png) for smoothing.

U-Net also reports mean-subject Dice approximately 0.99952 / 0.96142 / 0.96444 / 0.97837. Background is label 0; verify the anatomical meanings of the other codes against the course dataset. The held-out results are strong, but [example slices](../part4_results/unet_train_20260915_005104_064230/segmentations.png) are a visual sample, not the entire test set. The script selects checkpoints using validation foreground Dice, not test Dice.

The old BCE GAN produced near-identical images. The current [WGAN-GP grid](../part4_results/gan_wgan_gp_train_20260916_122956_496407/generated_final.png) is visibly more varied, with residual blur/artifacts. Its diagnostics do not prove realism, complete distribution coverage or lack of memorisation. Old `gan_train_...` folders are retained as historical evidence, not presented as WGAN-GP results.

Training has not been rerun for this documentation update. CPU smoke checks validate selected computational properties only. During the demo, open the exact run configuration and checkpoint alongside the figures, and explain preprocessing, architecture, objective, selection and limitations.
