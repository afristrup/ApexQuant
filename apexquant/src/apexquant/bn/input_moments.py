"""Per-input-channel first and second moments for every correctable convolution."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .activation_moments import activation_moments
from .case_letters import CORRECTABLE_CASES
from .conv_bn_site import ConvBnSite
from .pre_activation_moments import pre_activation_moments


def input_moments(model: nn.Module, sites: Sequence[ConvBnSite]
                  ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return ``conv name -> (e, s)`` per input channel, for the correctable sites.

    This is section 4.4's input-moment oracle, dispatched on the case letter.

    * **Case A**, input is ``act(BN_p(.))``: read the producer BatchNorm through
      :func:`bn_output_moments` and push ``(m, sigma)`` through the activation.
    * **Case B**, input is ``act(sum_r B_r)``: sum the branch means and the
      branch *variances* first, then apply the same activation formula to the
      sum. Summing after the activation would be a different and wrong model.
    * **Case C**, input is the image: asserted, not implemented. Section 4.4 says
      so because both falsification models leave their stem dense under the
      harness group rule, so no case-C layer is ever quantized and no case-C
      moments are ever needed. The assertion that this holds belongs where the
      quantized layers are known, which is :func:`apply_qbn`, not here; a case-C
      site simply gets no entry.
    * **Case D**, pooling and squeeze-excitation gates: the layer is uncorrected.
      It gets no entry either, and :func:`apply_qbn` counts it.

    Under spatial stationarity (Nagel et al. 2019 App. B, Eqs. 29 to 30) every
    column ``j`` of the reshaped weight inherits the moments of input channel
    ``in(j) = j // (kh kw)``, which is why one pair per *channel* is enough and
    the expansion to columns is left to the caller.

    Output moments are recorded per node as they are computed, so a later
    case-B site whose identity branch copies an earlier node finds them waiting.
    Sites arrive in forward order, which makes that always true.

    Args:
        model: The model the sites' dotted names address, in eval mode. Only
            BatchNorm parameters and buffers are read; no forward pass is run.
        sites: The records from :func:`conv_bn_sites`, in forward order.

    Returns:
        Dict from convolution name to ``(e, s)``, both shape ``(C_in,)``, for
        every site in a correctable case. ``s`` is the raw second moment.

    Raises:
        ValueError: A site's moments do not have one entry per input channel of
            its convolution, which means the topology adapter and the model
            disagree.
    """
    recorded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    moments: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for site in sites:
        if site.case not in CORRECTABLE_CASES:
            continue
        m, sigma = pre_activation_moments(model, site, recorded)
        e, s = activation_moments(site.activation, m, sigma)
        conv = model.get_submodule(site.conv)
        if e.shape != (conv.in_channels,):
            raise ValueError(
                f"{site.conv} takes {conv.in_channels} input channels but its "
                f"producers {site.producers} supply {tuple(e.shape)}")
        recorded[site.source] = (e, s)
        moments[site.conv] = (e, s)
    return moments
