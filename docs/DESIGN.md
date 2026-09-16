# Design and workflows

## Conventions used in the scripts

Each executable has a main guard so importing model classes does not start training. Command-line parsers expose settings; paths resolve relative to the script or an explicit argument. Model, loss, data, training and reporting functions have separate responsibilities. Comments explain tensor shapes, mathematical choices and gradient flow. Data checks fail early for invalid masks, overlapping subjects or non-finite losses.

Keeping one file per MRI task makes independent copying and assessment straightforward. The trade-off is duplicated data/reporting helpers; changes to those helpers must be checked in all three files. This is deliberate, rather than an implicit dependency between tasks.

## Notebook: Parts 1, 2 and 3.1

1. **Fourier:** sample time without repeating the endpoint → sum odd harmonics → run direct DFT and FFT → compare numerical outputs → inspect the spectrum and benchmark. CPU/GPU timings use warm-up and synchronisation in the benchmark. Fifty series terms reach harmonic 99; the initial supplied spectrum plot only extends to 50 Hz and therefore hides higher harmonics.
2. **PCA:** load LFW → split → subtract the training mean → obtain principal axes with SVD → project onto 150 components → train Random Forest → evaluate held-out identities. Centring does not reduce variance. The saved absolute-variance calculation uses the full sample count in its denominator; this should be the training count for absolute variance, although the explained-variance ratio is unchanged because the factor cancels. The split is random, not stratified cross-validation despite an old comment.
3. **Face CNN:** retain 2D images → add the channel dimension → batch → two convolution/ReLU/pooling blocks → flatten → dense layers → cross-entropy/Adam → held-out predictions. Seven output classes come from the filtered LFW identities. Do not apply softmax before CrossEntropyLoss. Re-run from a fresh kernel before presenting source and saved outputs as a matching execution.

## Part 3.2: ResNet-18

`load_data` creates 45,000/5,000/10,000 train/validation/test splits. `CachedCifarBatches` optionally keeps uint8 data on GPU and applies training crops, flips and normalisation per batch. `BasicBlock` adds a residual shortcut; `ResNet18` stacks eight blocks and classifies globally averaged features. The CIFAR stem is 3×3/stride 1.

`run_epoch` switches between optimisation and evaluation, using cross entropy. SGD with momentum, cosine scheduling, AMP and optional GPU caching improve throughput. Best validation accuracy selects weights; final test results are produced afterwards. `evaluation_metrics`, `save_history` and `save_evaluation` generate reports/figures. Timing definitions must accompany accuracy claims.

## Task 1: VAE

`find_data_folder` and `make_loaders` locate case-separated data. `PngSlices` maps grayscale MRI to [0,1]. `VAE.encode` uses four stride-2 convolutions and separate mean/log-variance heads. `forward` samples z = mean + exp(0.5 log variance) × noise during training; `decode` reconstructs via transposed convolutions and sigmoid.

`vae_loss` combines mean pixel MSE with beta × KL / pixel count. `epoch` trains with a KL warm-up; validation uses fixed beta and posterior means. `main` selects the lowest validation objective. `visualise` saves reconstructions, a latent manifold and embeddings. Blurry reconstructions can reflect the two-dimensional bottleneck and MSE objective; low loss is not proof that every anatomical detail is preserved.

## Task 2: U-Net

`PngSlices` pairs image and mask, scales images and maps mask codes 0/85/170/255 to classes 0/1/2/3. Image resizing is bilinear; mask resizing is nearest-neighbour. `double_conv` applies convolution/GroupNorm/ReLU twice. `UNet.forward` stores encoder features, pools, upsamples and concatenates matching skips, then emits four per-pixel logits through a 1×1 convolution.

`segmentation_loss` adds categorical cross entropy and one minus mean soft Dice, including background. `epoch` applies training intensity augmentation, updates weights and aggregates confusion matrices. `dice_scores` computes hard-label Dice; absent classes return NaN rather than a perfect score. `main` selects the highest validation foreground Dice. `report_dice` distinguishes pooled and mean-subject results. `visualise` shows image/ground-truth/prediction triplets. These are 2D predictions; per-subject aggregation does not turn the model into a 3D U-Net.

## Task 3: WGAN-GP

Scale MRI into [-1,1]. `Generator` maps Gaussian noise through transposed convolutions, BatchNorm and ReLU to a tanh image. `Critic` downsamples images to unrestricted scalar scores without BatchNorm or sigmoid. Neither network uses pretrained weights.

`gradient_penalty` interpolates detached real/fake inputs, differentiates critic scores with respect to input pixels and penalises (gradient norm − 1)². `create_graph=True` allows the penalty to train critic weights. `train_epoch` minimises fake mean − real mean + 10 × penalty for the critic and −fake mean for the generator. Five critic batches precede a generator update; the final partial group also updates it. Generator BatchNorm is frozen during critic-only phases. Critic parameters are frozen during generator updates while gradients still flow through its input.

`samples` writes fixed-noise grids; `diagnostics` measures diversity and a small nearest-training preview. `main` saves the last epoch, not a validation-selected best-realism model. WGAN-GP mitigates training instability but cannot guarantee no mode collapse. Diversity metrics are neither FID nor medical validity measures.
