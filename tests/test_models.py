"""Small CPU checks; no downloads, datasets or full training are required."""
import unittest

import numpy as np
import torch
from torch import nn

from part3 import ResNet18
from task1_vae import VAE, vae_loss
from task2_unet import UNet, dice_scores, segmentation_loss
from task3_gan import Generator, Critic, gradient_penalty


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def assert_finite_backward(self, model, loss):
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))

    def test_resnet_classification_backward(self):
        model = ResNet18()
        output = model(torch.rand(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assert_finite_backward(model, nn.functional.cross_entropy(output, torch.tensor([0, 9])))

    def test_vae_reconstruction_and_prior_kl(self):
        model = VAE(image_size=32, latent_dim=2)
        images = torch.rand(2, 1, 32, 32)
        output, mean, logvar = model(images)
        self.assertEqual(output.shape, images.shape)
        self.assertTrue(((output >= 0) & (output <= 1)).all())
        self.assert_finite_backward(model, vae_loss(output, images, mean, logvar, 1)[0])
        # A posterior exactly equal to N(0,I) has zero KL; perfect reconstruction has zero MSE.
        loss, mse, kl = vae_loss(images, images, torch.zeros(2, 2), torch.zeros(2, 2), 1)
        self.assertEqual((loss.item(), mse.item(), kl.item()), (0, 0, 0))

    def test_unet_segmentation_backward(self):
        model = UNet(classes=4, base_channels=8)
        output = model(torch.rand(2, 1, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 4, 32, 32))
        self.assert_finite_backward(model, segmentation_loss(output, torch.randint(4, (2, 32, 32))))

    def test_dice_known_overlap_and_absent_class(self):
        scores = dice_scores(np.array([[8, 2, 0], [1, 9, 0], [0, 0, 0]]))
        np.testing.assert_allclose(scores[:2], [16 / 19, 18 / 21])
        self.assertTrue(np.isnan(scores[2]))

    def test_gan_shapes_and_generator_gradient(self):
        generator, critic = Generator(image_size=32), Critic(image_size=32)
        fake = generator(torch.randn(2, 100, 1, 1))
        self.assertEqual(tuple(fake.shape), (2, 1, 32, 32))
        self.assertTrue(((fake >= -1) & (fake <= 1)).all())
        scores = critic(fake)
        self.assertEqual(tuple(scores.shape), (2,))
        self.assert_finite_backward(generator, -scores.mean())

    def test_gradient_penalty_analytic_norm_and_detached_endpoints(self):
        # C(x) = weight * sum(x)/sqrt(pixel count) has input norm abs(weight).
        class LinearCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(2.0))

            def forward(self, x):
                return self.weight * x.flatten(1).sum(1) / x[0].numel() ** 0.5

        critic = LinearCritic()
        real = torch.randn(2, 1, 4, 4, requires_grad=True)
        fake = torch.randn_like(real, requires_grad=True)
        penalty, norm = gradient_penalty(critic, real, fake)
        self.assertAlmostEqual(norm.item(), 2.0, places=5)
        self.assertAlmostEqual(penalty.item(), 1.0, places=5)
        penalty.backward()
        self.assertAlmostEqual(critic.weight.grad.item(), 2.0, places=5)
        self.assertIsNone(real.grad)
        self.assertIsNone(fake.grad)


if __name__ == '__main__':
    unittest.main()
