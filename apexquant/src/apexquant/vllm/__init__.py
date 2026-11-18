"""Optional vLLM integration for ApexQuant packed weights."""

from .adapter import ApexVLLMLinear, make_kv_cache, make_moe
from .plugin import ApexVLLMManifest, load, single_request_compilation_config

_REGISTERED = False


def register():
    """Register the ApexQuant quantization method with vLLM, if installed."""
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from vllm.model_executor.layers.quantization import register_quantization_config
    except ImportError:
        return
    from .quantization import ApexVLLMQuantizationConfig
    register_quantization_config("apexquant")(ApexVLLMQuantizationConfig)
    from vllm.config.cache import register_kv_cache_dtype
    from vllm.v1.attention.backends.registry import register_backend
    from vllm.v1.kv_cache_interface import KVQuantMode

    @register_kv_cache_dtype("apexquant_e8")
    class ApexQuantE8DType:
        name = "apexquant_e8"

        def torch_dtype(self):
            import torch
            return torch.float8_e4m3fn

        def quant_mode(self):
            return KVQuantMode.FP8_PER_TENSOR

    @register_kv_cache_dtype("apexquant_e8p2")
    class ApexQuantE8P2DType:
        name = "apexquant_e8p2"

        def torch_dtype(self):
            import torch
            return torch.uint8

        def quant_mode(self):
            return KVQuantMode.CUSTOM

    @register_kv_cache_dtype("apexquant_int8")
    class ApexQuantInt8DType:
        name = "apexquant_int8"

        def torch_dtype(self):
            import torch
            return torch.int8

        def quant_mode(self):
            return KVQuantMode.INT8_PER_TOKEN_HEAD

    @register_kv_cache_dtype("apexquant_int4")
    class ApexQuantInt4DType:
        name = "apexquant_int4"

        def torch_dtype(self):
            import torch
            return torch.uint8

        def quant_mode(self):
            return KVQuantMode.INT4_PER_TOKEN_HEAD

    @register_kv_cache_dtype("apexquant_trellis")
    class ApexTrellisDType:
        name = "apexquant_trellis"

        def torch_dtype(self):
            import torch
            return torch.uint8

        def quant_mode(self):
            return KVQuantMode.CUSTOM

    register_backend(
        "APEXQUANT_INT4",
        "apexquant.vllm.int4_attention.ApexQuantInt4AttentionBackend",
    )
    register_backend(
        "APEXQUANT_INT8",
        "apexquant.vllm.int4_attention.ApexQuantInt8AttentionBackend",
    )
    register_backend(
        "APEXQUANT_E8",
        "apexquant.vllm.int4_attention.ApexQuantE8AttentionBackend",
    )
    register_backend(
        "APEXQUANT_E8_RESIDUAL",
        "apexquant.vllm.e8_residual_attention.ApexE8ResidualAttentionBackend",
    )
    register_backend(
        "APEXQUANT_TRELLIS",
        "apexquant.vllm.trellis_attention.ApexTrellisAttentionBackend",
    )
    from vllm.platforms import current_platform
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    platform_cls = type(current_platform)
    if not getattr(platform_cls, "_apexquant_trellis_selector", False):
        original_selector = platform_cls.get_attn_backend_cls

        def select_backend(cls, selected_backend, attn_selector_config, num_heads=None):
            if selected_backend is None:
                if attn_selector_config.kv_cache_dtype == "apexquant_e8":
                    selected_backend = AttentionBackendEnum.APEXQUANT_E8
                elif attn_selector_config.kv_cache_dtype == "apexquant_e8p2":
                    selected_backend = AttentionBackendEnum.APEXQUANT_E8_RESIDUAL
                elif attn_selector_config.kv_cache_dtype == "apexquant_int8":
                    selected_backend = AttentionBackendEnum.APEXQUANT_INT8
                elif attn_selector_config.kv_cache_dtype == "apexquant_int4":
                    selected_backend = AttentionBackendEnum.APEXQUANT_INT4
                elif attn_selector_config.kv_cache_dtype == "apexquant_trellis":
                    selected_backend = AttentionBackendEnum.APEXQUANT_TRELLIS
            return original_selector(selected_backend, attn_selector_config, num_heads)

        platform_cls.get_attn_backend_cls = classmethod(select_backend)
        platform_cls._apexquant_trellis_selector = True
    _REGISTERED = True


__all__ = ["ApexVLLMLinear", "ApexVLLMManifest", "load", "make_kv_cache", "make_moe", "register", "single_request_compilation_config"]
