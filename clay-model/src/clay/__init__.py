"""Clay foundation model, without Lightning.

The network is a straight torch port of ``claymodel``; the pieces that made it a
Lightning project (the module wrapper, the datamodule, the callbacks, the
training objective with its frozen DINOv2 teacher) are not here, because this
package exists to define the model and run it forward.

Layout:
    ``clay.model``      the network, one module per file
    ``clay.ops``        each op, choosing a Triton kernel or a torch reference
    ``clay.kernels``    the Triton kernels themselves
    ``clay.inference``  loading published weights and producing embeddings
"""

from .inference.embed import embed
from .inference.load_checkpoint_state_dict import load_checkpoint_state_dict
from .inference.load_sensor_metadata import load_sensor_metadata
from .model.build_decoder import build_decoder
from .model.build_encoder import build_encoder
from .model.decoder import Decoder
from .model.encoder import Encoder

__all__ = [
    "Decoder",
    "Encoder",
    "build_decoder",
    "build_encoder",
    "embed",
    "load_checkpoint_state_dict",
    "load_sensor_metadata",
]
