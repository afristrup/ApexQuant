"""Quantization-aware BatchNorm: topology, moment oracle, row algebra, correction."""

from __future__ import annotations

from .activation_kind import activation_kind
from .activation_moments import activation_moments
from .apply_qbn import apply_qbn
from .arm_model import arm_model
from .arm_result import ArmResult
from .bn_output_moments import bn_output_moments
from .bundled_activation import bundled_activation
from .case_fractions import case_fractions
from .case_letters import (CASE_A, CASE_B, CASE_C, CASE_D, CASE_LETTERS,
                           CORRECTABLE_CASES)
from .clipped_linear_moments import clipped_linear_moments
from .conv_bn_site import ConvBnSite
from .conv_bn_sites import conv_bn_sites
from .group_tau import group_tau
from .harness_group_size import harness_group_size
from .input_moments import input_moments
from .input_source import InputSource
from .layer_residual import LayerResidual
from .perp_energy import perp_energy
from .pre_activation_moments import pre_activation_moments
from .qbn_arms import (ARM_DATA_FREE, ARM_FOLDED, ARM_FP32, ARM_MODES,
                       ARM_NAMES, ARM_NUMBERS, ARM_QBN_MEAN,
                       ARM_QBN_MEAN_VAR, ARM_QBN_VAR, ARM_QUANTIZED,
                       ARM_REESTIMATED, FP32_PASSES)
from .qbn_correction import qbn_correction
from .qbn_correction_result import QbnCorrection
from .qbn_criteria import qbn_criteria
from .qbn_guards import (MAX_MEAN_SHIFT, SKIP_LAYOUT, SKIP_NO_BATCHNORM,
                         SKIP_NO_RESIDUAL, SKIP_NONE, SKIP_UNCORRECTABLE_CASE,
                         VAR_FLOOR_RATIO)
from .qbn_modes import MODE_MEAN, MODE_MEAN_VAR, MODE_VAR, QBN_MODES
from .qbn_report import QbnReport
from .quadrature_moments import quadrature_moments
from .quantize_with_residuals import quantize_with_residuals
from .relu_moments import relu_moments
from .residual_input_source import residual_input_source
from .row_alpha import row_alpha
from .site_from_source import site_from_source

__all__ = ["ARM_DATA_FREE", "ARM_FOLDED", "ARM_FP32", "ARM_MODES",
           "ARM_NAMES", "ARM_NUMBERS", "ARM_QBN_MEAN", "ARM_QBN_MEAN_VAR",
           "ARM_QBN_VAR", "ARM_QUANTIZED", "ARM_REESTIMATED", "ArmResult",
           "CASE_A", "CASE_B", "CASE_C", "CASE_D", "CASE_LETTERS",
           "CORRECTABLE_CASES", "ConvBnSite", "FP32_PASSES", "InputSource",
           "LayerResidual", "MAX_MEAN_SHIFT", "MODE_MEAN", "MODE_MEAN_VAR",
           "MODE_VAR", "QBN_MODES", "QbnCorrection", "QbnReport",
           "SKIP_LAYOUT", "SKIP_NONE", "SKIP_NO_BATCHNORM",
           "SKIP_NO_RESIDUAL", "SKIP_UNCORRECTABLE_CASE", "VAR_FLOOR_RATIO",
           "activation_kind", "activation_moments", "apply_qbn",
           "arm_model", "bn_output_moments", "bundled_activation",
           "case_fractions", "clipped_linear_moments", "conv_bn_sites",
           "group_tau", "harness_group_size", "input_moments",
           "perp_energy", "pre_activation_moments", "qbn_correction",
           "qbn_criteria", "quadrature_moments", "quantize_with_residuals",
           "relu_moments", "residual_input_source", "row_alpha",
           "site_from_source"]
