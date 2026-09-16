# Validation record

16 September 2026, Windows CPU checks.

`python -m unittest discover -s tests -v`: **6 checks passed**.

Python 3.13.7; torch 2.9.0+cu128; NumPy 2.2.6; Pillow 11.3.0; scikit-learn 1.7.2; matplotlib 3.10.7

Checks cover ResNet and U-Net output/gradient contracts, VAE zero-KL reference and gradients, GAN output/gradient contracts, known Dice values and absent classes, and an analytic WGAN-GP input gradient including endpoint detachment. Small synthetic inputs are used; no training dataset is downloaded.

All relative documentation links were checked against the project files. These checks do not establish full-training convergence, GPU performance, cluster availability or marks. Existing experiment results are documented separately in RESULTS.md.
