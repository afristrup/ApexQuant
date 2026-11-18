"""Fast ApexQuant KV backend using vLLM fused INT4 attention."""

from typing import ClassVar

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVCacheLayout


class ApexQuantInt4AttentionBackend(TritonAttentionBackend):
    """ApexQuant INT4 cache on vLLM fused Triton paged attention."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["apexquant_int4"]

    @staticmethod
    def get_name() -> str:
        return "APEXQUANT_INT4"

    @staticmethod
    def get_impl_cls() -> type["ApexQuantInt4AttentionImpl"]:
        return ApexQuantInt4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "apexquant_int4"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)


class ApexQuantInt4AttentionImpl(TritonAttentionImpl):
    """vLLM Triton INT4 implementation with ApexQuant registration."""


class ApexQuantInt8AttentionBackend(TritonAttentionBackend):
    """ApexQuant E8 cache on vLLM fused Triton paged attention."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["apexquant_int8"]

    @staticmethod
    def get_name() -> str:
        return "APEXQUANT_INT8"

    @staticmethod
    def get_impl_cls() -> type["ApexQuantInt8AttentionImpl"]:
        return ApexQuantInt8AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "apexquant_int8"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)


class ApexQuantInt8AttentionImpl(TritonAttentionImpl):
    """vLLM Triton INT8 implementation with ApexQuant registration."""


class ApexQuantE8AttentionBackend(TritonAttentionBackend):
    """ApexQuant E8 cache on vLLM fused Triton paged attention."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["apexquant_e8"]

    @staticmethod
    def get_name() -> str:
        return "APEXQUANT_E8"

    @staticmethod
    def get_impl_cls() -> type["ApexQuantE8AttentionImpl"]:
        return ApexQuantE8AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "apexquant_e8"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)


class ApexQuantE8AttentionImpl(TritonAttentionImpl):
    """vLLM Triton FP8 implementation with ApexQuant registration."""
