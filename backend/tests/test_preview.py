"""The per-step preview's maths (aries_worker/stages/preview.py), in numpy: the
convolution weights must give what the fitted matrix gives directly (studio/src/preview.rs
applies it that way), and the clean-image estimate must undo an Euler step."""
import numpy as np

from aries_worker.stages.preview import CHANNELS, INPUTS, OUTPUTS, SCALE, clean_estimate, conv_weights


def direct(matrix, z):
    """Studio's way: each latent pixel's 3x3 neighbourhood times the matrix."""
    c, h, w = z.shape
    zp = np.pad(z, ((0, 0), (1, 1), (1, 1)), mode="edge")
    feats = np.concatenate(
        [zp[:, dy:dy + h, dx:dx + w].reshape(c, -1).T for dy in range(3) for dx in range(3)]
        + [np.ones((h * w, 1))],
        1,
    )
    out = feats @ matrix
    return out.reshape(h, w, SCALE, SCALE, 3).transpose(0, 2, 1, 3, 4).reshape(h * SCALE, w * SCALE, 3)


def as_conv(matrix, z):
    """The worker's way, with numpy for torch: conv2d on the padded latent, then the same
    reorder as StepPreview.render."""
    weight, bias = conv_weights(matrix)
    c, h, w = z.shape
    zp = np.pad(z, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros((OUTPUTS, h, w))
    for dy in range(3):
        for dx in range(3):
            out += np.einsum("oc,chw->ohw", weight[:, :, dy, dx], zp[:, dy:dy + h, dx:dx + w])
    out += bias[:, None, None]
    return out.reshape(1, SCALE, SCALE, 3, h, w).transpose(0, 4, 1, 5, 2, 3).reshape(h * SCALE, w * SCALE, 3)


def test_conv_weights_match_the_matrix():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(INPUTS, OUTPUTS))
    z = rng.normal(size=(CHANNELS, 3, 5))
    np.testing.assert_allclose(as_conv(matrix, z), direct(matrix, z), rtol=1e-9, atol=1e-9)


def test_clean_estimate_undoes_an_euler_step():
    rng = np.random.default_rng(1)
    x0, noise = rng.normal(size=8), rng.normal(size=8)
    v = noise - x0  # flow matching: x_sigma = (1 - sigma) x0 + sigma noise
    x_prev = 0.7 * noise + 0.3 * x0
    x_next = x_prev + (0.4 - 0.7) * v
    np.testing.assert_allclose(clean_estimate(x_next, x_prev, 0.7, 0.4), x0)
