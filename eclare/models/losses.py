import torch
import numpy as np
import torch.nn.functional as F


def unwrap(x):
    pi = torch.pi
    y = x % (2 * pi)
    return torch.where(y > pi, 2 * pi - y, y)


def power_of_2_pad(x):
    """
    In order to compute FFT in half-precision, all dimensions
    must be powers of 2. This function takes `x` and returns
    a reflect-padded `x_pad` where all axis are powers of 2.
    """
    # find the next power of two
    target_shape = [
        2 ** ((d - 1).bit_length()) if i > 1 else 0 for i, d in enumerate(x.shape)
    ]

    # pads
    padding = [new - old for new, old in zip(target_shape, x.shape)]

    x_pad = F.pad(x, (0, padding[-2], 0, padding[-1]), mode="reflect")
    return x_pad


def mag_pha(x, log=False):
    omega = torch.fft.rfftn(power_of_2_pad(x))

    mag = torch.abs(omega).float()
    if log:
        mag = torch.log(mag + 1e-8)

    pha = torch.angle(omega).float()
    pha = unwrap(pha)

    return mag, pha


class GANLoss(torch.nn.Module):
    r"""Loss of the original GAN with cross entropy.

    For the discriminator :math:`D`, this loss minimizes the binary cross
    entropy with logits:

    .. math::

        l = - \mathrm{mean}_x (y \ln(\sigma(D(x)))
          + (1 - y) \ln(1 - \sigma(D(x)))),

    where :math:`\sigma` is the sigmoid function. If
    :math:`x \in \mathrm{\{truth\}}`, we have :math:`y = 1` and

    .. math::

        l = - \mathrm{mean}_x \ln(\sigma(D(x))).

    If :math:`x \in \mathrm{\{generated\}}`, i.e. :math:`\exists z` s.t.
    :math:`x = G(z)` where `G` is the generator, we have :math:`y = 0` and

    .. math::

        l = - \mathrm{mean}_x \ln(1 - \sigma(D(G(z)))).

    Combine the above two terms together, we can get the loss for the
    discriminator.

    For the generator :math:`G`, this loss minimizes the binary cross entropy
    with the same form and :math:`y = 1`, which is to minimize

    .. math::

        l = - \mathrm{mean}_x \ln(\sigma(D(G(z)))).

    This is the modified GAN loss which minimizes
    :math:`l = \mathrm{mean}_x \ln(1 - \sigma(D(G(z))))`.

    """

    def __init__(self):
        super().__init__()

    def forward(self, x, is_real):
        """Calculates the loss.

        Args:
            x (torch.Tensor): The image.
            is_real (bool): Whether the image is real.

        Returns:
            The calculated loss.

        """
        target = torch.ones_like(x) - 0.1 if is_real else torch.zeros_like(x) + 0.1
        loss = F.binary_cross_entropy_with_logits(x, target)
        return loss


class SmoothnessLoss(torch.nn.Module):
    r"""L2 norm of derivative.

    This loss minimizes

    .. math::

        l = \lVert \nabla k \rVert_2,

    where :math:`k` is the kernel, to encourage smoothness.

    """

    def forward(self, kernel):
        """Calculates the loss for this kernel."""
        device = kernel.device
        operator = torch.tensor([1, -1], dtype=torch.float32, device=device)
        operator = operator[None, None, ..., None]
        derivative = F.conv2d(kernel, operator)
        loss = torch.sqrt(torch.sum(derivative**2))
        return loss


class CenterLoss(torch.nn.Module):
    r"""Penalizes off-center.

    This loss minimizes the differences between the center of the kernel and the
    center of the vector:

    .. math::

        l = \left(\sum_x k(x) x - C \right) ^ 2,

    where :math:`k(x)` is the kernel, :math:`x` is vector indices, and :math:`C`
    is the center of the vector. Assume the sum of the kernel equals 1.

    """

    def __init__(self, kernel_length):
        super().__init__()
        self.kernel_length = kernel_length
        center = torch.tensor(self.kernel_length // 2, dtype=torch.float32)
        self.register_buffer("center", center)
        locs = torch.arange(self.kernel_length, dtype=torch.float32)
        self.register_buffer("locs", locs)

    def forward(self, kernel):
        """Calculates the loss for this kernel."""
        kernel_center = torch.sum(kernel.squeeze() * self.locs)
        loss = F.mse_loss(kernel_center, self.center)
        return loss


class BoundaryLoss(torch.nn.Module):
    r"""Penalizes non-zero values at kernel boundary.

    This loss minimizes the weighted sum:

    .. math::

        l = \sum_x | m(x) k(x) |

    where :math:`m` is 0, 0, 1, 1, ..., 1, 1, 0, 0.

    """

    def __init__(self, kernel_length):
        super().__init__()
        self.kernel_length = kernel_length
        mask = self._create_penalty_mask().clone().detach().float()
        self.register_buffer("mask", mask[None, None, ..., None])

    def _create_penalty_mask(self):
        mask = torch.ones(self.kernel_length).float()
        mask[2:-2] = 0
        return mask

    def forward(self, kernel):
        """Calculates the loss for this kernel."""
        return torch.sum(torch.abs(kernel * self.mask))


class PeakLoss(torch.nn.Module):
    def forward(self, kernel):
        device = kernel.device
        op_f = torch.tensor([1, -1], dtype=torch.float32, device=device)
        op_b = torch.tensor([-1, 1], dtype=torch.float32, device=device)
        op_f = op_f[None, None, ..., None]
        op_b = op_b[None, None, ..., None]
        diff_f = F.conv2d(F.pad(kernel, (0, 0, 0, 1)), op_f).squeeze()
        diff_b = F.conv2d(F.pad(kernel, (0, 0, 1, 0)), op_b).squeeze()
        mid_ind = kernel.shape[2] // 2
        result = torch.sum(F.relu(diff_f[:mid_ind])) + torch.sum(
            F.relu(-diff_f[mid_ind:])
        )
        # result = torch.tensor(0).float().cuda()
        # diff = F.relu(diff_f * diff_b).squeeze()
        # result = torch.sum(diff[:mid_ind]) + torch.sum(diff[mid_ind+1:])
        return result
