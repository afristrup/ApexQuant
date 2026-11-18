"""Clay v1.5 on EuroSAT: linear probe accuracy and head KL per trellis depth."""

from __future__ import annotations

import os
from pathlib import Path

PORT = Path(__file__).resolve().parents[3] / "clay-model"
METADATA = PORT / "configs" / "metadata.yaml"
REPO = "made-with-clay/Clay"
CHECKPOINT = "v1.5/clay-v1.5.ckpt"
PLATFORM = "sentinel-2-l2a"
EUROSAT = Path(os.environ.get("EUROSAT_DIR", "/mnt/bulk/eurosat/unpacked/ds/images/"
                              "remote_sensing/otherDatasets/sentinel_2/tif"))
# EuroSAT band order B01..B12 then B8A; Clay's S2 order is B02-B08, B8A, B11, B12.
BANDS = [1, 2, 3, 4, 5, 6, 7, 12, 10, 11]
CHIPS = 10_000
TRAIN_SHARE = 0.8
SEED = 0
GROUP = 512
PASSES = (1, 2, 3, 4)
STAGES = ("trellis", "e8p", "scalar", "trellis_sinkhorn")
SINKHORN_ITERS = 10
MIN_IN_FEATURES = 32
BATCH = 128
