# Reproduce "ApexQuant: Data-Free Elastic Quantization by Residual
# Re-Isotropization".
#
# The method in one line: quantize a weight row in passes, drawing a fresh
# random Hadamard rotation before each one, so the residual's marginal returns
# exactly to the density the stage's codebook was built for (proposition 1),
# which gives a closed-form decay err(P) = rho^P (proposition 2) and makes
# every prefix of P passes a valid lower-rate model on its own. Three stages
# are interchangeable inside that recursion: scalar Lloyd-Max, E8 lattice, and
# a Viterbi-coded trellis with register width L=14 (2^14 states).
#
# Start with `make stages`. It needs no download, no GPU and about five
# seconds, and it is the one target that checks the paper's central claim
# directly: that the three stages' rho^2 constants are known before any
# weight is read. Everything else here either exercises a Triton kernel (needs
# CUDA) or loads a real checkpoint (needs the network and, for the vision
# table, a gated Hugging Face repo).
#
# ---------------------------------------------------------------------------
# Where each table lives.
#
# Table 2 (per-stage rate-distortion)         -> make stages
# Appendix B (Hadamard vs. a true Haar draw)   -> make rotation-check
# Table 3 / 7 / 8 (LLM KL, four models)        -> make llm-table
# Table 4 / 5 (Clay v1.5 embedding error)      -> make vision-table
# Table 6 (Viterbi encoder throughput)         -> make kernel-bench
# Table 13 (elastic-serving decode latency)    -> make serving-bench
#
# serving-bench and its extended sweep (make serving-sweep) are the one place
# this package still depends on a forked third-party kernel: gemlite's
# GEMM_SPLITK, pinned in the optional `gemlite` extra. Nothing else here needs
# it, and the dispatcher checks its availability before ever routing to it.
# ---------------------------------------------------------------------------

SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help setup preflight test stages rotation-check llm-table vision-table kernel-bench serving-bench serving-sweep download convert convert-preflight clean

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY   ?= uv run --project $(ROOT) python

# The model haar_substitution_llm.py measures by default. Any decoder-only
# causal LM works; the paper's own repeat of this check is on Phi-1.5.
MODEL      ?= microsoft/phi-1_5
# Space-separated HF repo ids for the LLM table. Left empty, iclr_llm_table.py
# falls back to its own default four (TinyLlama-1.1B, Phi-1.5, Mistral-7B,
# Qwen2.5-7B) -- the same models Table 3 reports.
MODELS     ?=
# Where the LLM table's resumable run writes its results. Re-running `make
# llm-table` after an interrupt continues rather than restarts, same as the
# root Makefile's `convert`.
LLM_OUT    ?= results/llm-table
# The register width the Viterbi encoder is sized for. 14 is what table 2
# reports; 12 is what production conversions use because 14 costs about 3x the
# encode time for a small rho improvement. See trellis_encoder_levers.py's own
# docstring before changing this -- 16 spills the fused kernel's registers and
# falls back to the ~68x slower torch path, which is a different measurement,
# not a bigger number on the same one.
TRELLIS_BITS ?= 14

# `make convert` variables: turn any Hugging Face causal LM into a progressive
# artifact at a chosen stage and depth. MODEL and OUT are required and checked
# by convert-preflight, same as the root repo Makefile's `convert`; everything
# below has the paper's own operating point as its default.
OUT          ?=
# scalar | e8p | trellis, increasing order of strength (table 2). Note the
# flag spelling: it's e8p, not e8 -- quantise_hf_progressive.py's argparse
# choices are exactly ("scalar", "e8p", "trellis").
STAGE        ?= trellis
# How many residual passes to store. Prefixes are bit-identical to a fresh
# shallower encode, so `PASSES=3` also gives you `--passes 1`/`--passes 2` for
# free by reading fewer of the stored planes (proposition 2's nesting
# property) -- there's no reason to run this twice for the shallow tiers.
PASSES       ?= 3
BITS         ?= 2
GROUP_SIZE   ?= 512
SEED         ?= 42
# Regex of tensor names to leave alone. Needed for anything that is not purely
# a language model: "layers." matches a vision tower's blocks exactly as it
# matches a text decoder's, so on a VLM checkpoint the default rule would
# sweep up the image encoder alongside the text decoder. Quantising an image
# encoder is a different claim from quantising a language model, so it has to
# be opted into.
EXCLUDE      ?=
# Keep the downloaded source shards instead of deleting each after use. Off by
# default so a large conversion doesn't also need double the disk.
KEEP_SHARDS  ?= 0
# Where `make download` puts a model, its own step so the transfer is
# watchable and resumable rather than buried inside the encode loop.
SRC          ?=

help:
	@echo "make stages          Table 2: the three stages' rho^2, CPU, ~5s, no download"
	@echo "make rotation-check  Appendix B: Hadamard vs. dense Haar, MODEL=$(MODEL)"
	@echo "make llm-table       Table 3/7/8: LLM KL sweep, MODELS=\"$(MODELS)\" (empty = the paper's four)"
	@echo "make vision-table    Table 4/5: Clay v1.5 embedding error at P=1,2,3"
	@echo "make kernel-bench    Table 6: Viterbi encoder throughput, TRELLIS_BITS=$(TRELLIS_BITS)"
	@echo "make serving-bench   Table 13: tiered-decode latency, CUDA + the gemlite extra"
	@echo "make serving-sweep   the mix/batch-size sweep behind table 13's prose"
	@echo "make convert         MODEL=<hf id|dir> OUT=<dir> STAGE=$(STAGE) PASSES=$(PASSES)  quantise a real model"
	@echo "  STAGE=scalar|e8p|trellis  BITS=$(BITS) GROUP_SIZE=$(GROUP_SIZE) TRELLIS_BITS=$(TRELLIS_BITS)"
	@echo "make download        MODEL=<hf id> SRC=<dir>  fetch a model over the Hub first (large models)"
	@echo "make test            the package's own test suite"
	@echo "make setup           uv sync, all extras (triton + gemlite + the experiment deps)"
	@echo "make clean           remove caches and this Makefile's own result dirs"

setup:
	@uv sync --project $(ROOT) --all-extras

# Fails before the expensive step rather than partway through it. CUDA absence
# is not fatal here (stages and rotation-check are CPU-honest), so this only
# warns; kernel-bench checks it again itself and refuses to run without CUDA,
# because an "exact Viterbi on CPU" run is not a slower version of that target,
# it is a different, impractical one -- same reasoning as the root Makefile's
# convert-preflight.
preflight:
	@$(PY) -c "import sys; assert sys.version_info >= (3, 10), 'need Python >= 3.10'"
	@$(PY) -c "\
import torch; \
print('### CUDA:', 'available' if torch.cuda.is_available() else 'NOT available (CPU-only targets still work)')"
	@$(PY) -c "\
import importlib.util as u; \
print('### triton:', 'available' if u.find_spec('triton') else 'NOT available (falls back to the torch reference path, slower)')"
	@$(PY) -c "\
import importlib.util as u; \
print('### gemlite:', 'available' if u.find_spec('gemlite') else 'NOT available (only serving-bench/serving-sweep need it -- make setup installs it)')"

test:
	@uv run --project $(ROOT) pytest $(ROOT)/tests

# The fastest correctness check there is: no weight is read, no rotation is
# drawn. apex_beta_levels, E8PCodec and TrellisCodec each expose their stage's
# closed-form rho^2 (equation 3), computed by quadrature over the sphere's
# coordinate marginal before any data exists. This just prints them next to
# the paper's table 2 (scalar 0.1171, E8 0.0906, trellis 0.0720 at d=512,
# b=2, L=14) so a drift shows up as a number, not a hunch.
stages:
	@$(PY) -m apexquant.cli stages

# Appendix B's substitution test: redraw the deployed randomized-Hadamard
# rotation as a dense, honestly-Haar random-orthogonal matrix instead, and
# check the per-pass error moves by parts in a thousand, not more. Defaults
# to 8 decoder layers and seed 42, matching the paper's own repeat of the
# vision-encoder version of this check on a language model.
rotation-check: preflight
	@$(PY) $(ROOT)/experiments/haar_substitution_llm.py --model $(MODEL)

# The seven-arm, two-bit-budget sweep: rtn, quarot, and five data-free
# apex configurations (scalar, l_p, E8, trellis, and the paper's own
# two-pass progressive artifact) against real HQQ, SINQ and the calibrated
# A-SINQ, all through one protocol so the KL column in table 3/7/8 is
# comparable across rows. Writes to $(LLM_OUT) and resumes if interrupted,
# same resumability contract as the root Makefile's `convert`.
llm-table: preflight
	@mkdir -p $(LLM_OUT)
	@$(PY) $(ROOT)/experiments/iclr_llm_table.py --out $(LLM_OUT) $(if $(MODELS),--models $(MODELS),)

# Clay v1.5 through the folded trellis kernel: quantize the encoder's linears
# in place, run a real 224px chip through it, and read the embedding the model
# actually produces at each of the three nested tiers, rather than a weight-
# space proxy. Needs a Clay v1.5 checkpoint reachable the way clay-model's own
# loader expects (see ../clay-model/src/clay/inference/load_checkpoint_state_dict.py);
# this is the one target here that needs both CUDA and a real download.
vision-table: preflight
	@$(PY) $(ROOT)/experiments/progressive_trellis_clay.py

# What the Viterbi encoder actually costs and what buys the state bits back:
# the path-cost-in-registers kernel against the DRAM-bound reference, at the
# register width TRELLIS_BITS. Table 6 reports L=14, 68x over the reference;
# L=16 is included in the sweep on purpose, to show the register budget's
# fallback rather than hide it.
kernel-bench: preflight
	@$(PY) -c "import torch, sys; sys.exit(0) if torch.cuda.is_available() else (print('### kernel-bench needs CUDA: the Viterbi encoder on CPU is not a slower run of this target, it is a different one.'), sys.exit(1))"
	@$(PY) $(ROOT)/experiments/trellis_encoder_levers.py --trellis-bits $(TRELLIS_BITS)

# Table 13: five ways to serve a batch that mixes decode tiers from one nested
# artifact, on the Mistral-7B 32-block stack's linear shapes with random
# weights -- no checkpoint download, since the claim is about the kernel's
# weight traffic, not about Mistral specifically. Graph-captured so eager
# per-call overhead doesn't distort the comparison (that mistake is why an
# earlier eager-timed version of this measurement got the sign wrong).
# Needs both CUDA and the gemlite extra: `make setup` installs the latter,
# this target refuses early and says so if either is missing.
serving-bench: preflight
	@$(PY) -c "\
import importlib.util as u, sys, torch; \
missing = [] if torch.cuda.is_available() else ['CUDA']; \
missing += [] if u.find_spec('gemlite') else ['gemlite']; \
sys.exit(0) if not missing else \
(print(f'### serving-bench needs {missing}; run: uv sync --extra gemlite (needs CUDA too if that is what is missing)'), sys.exit(1))"
	@$(PY) $(ROOT)/experiments/tiered_decoding.py

# The follow-up sweep behind table 13's prose: how the rider ratio moves as
# the tier mix shifts, and where the batch turns compute-bound enough for
# tiering's saving over "just serve everyone at premium" to become material.
serving-sweep: preflight
	@$(PY) -c "\
import importlib.util as u, sys, torch; \
missing = [] if torch.cuda.is_available() else ['CUDA']; \
missing += [] if u.find_spec('gemlite') else ['gemlite']; \
sys.exit(0) if not missing else \
(print(f'### serving-sweep needs {missing}; run: uv sync --extra gemlite (needs CUDA too if that is what is missing)'), sys.exit(1))"
	@$(PY) $(ROOT)/experiments/tiered_sweep.py

# `hf download` rather than snapshot_download in python: prints a live
# per-file progress bar, and interrupting it is safe because the cache is
# content-addressed, so re-running resumes rather than restarts. Its own step,
# same reasoning as the root repo Makefile's `download`: converting straight
# from a repo id fetches one shard at a time inside the encode loop, so a slow
# link shows up as a conversion sitting at zero percent GPU with nothing to
# look at, where a separate download step is at least watchable.
download:
	@if [ -z "$(MODEL)" ] || [ -z "$(SRC)" ]; then \
	  echo "make download needs MODEL and SRC:"; \
	  echo "  make download MODEL=Qwen/Qwen2.5-7B-Instruct SRC=/mnt/bulk/qwen7b"; \
	  exit 1; \
	fi
	@mkdir -p $(SRC)
	@echo "### fetching $(MODEL) -> $(SRC)  [$$(date -u +%H:%M:%SZ)]"
	@uv run --project $(ROOT) hf download $(MODEL) --local-dir $(SRC)
	@echo "### done  [$$(date -u +%H:%M:%SZ)]. Now: make convert MODEL=$(SRC) OUT=<artifact dir>"

# Fails before the download rather than partway through a multi-hour
# conversion. The CUDA check is the one that matters for the trellis stage:
# exact Viterbi on a CPU is not a slower conversion, it is an impractical one,
# and the script will otherwise start happily without saying so.
convert-preflight:
	@if [ -z "$(MODEL)" ] || [ -z "$(OUT)" ]; then \
	  echo "make convert needs MODEL and OUT:"; \
	  echo "  make convert MODEL=mistralai/Mistral-7B-v0.1 OUT=out/mistral7b-trellis-p3"; \
	  exit 1; \
	fi
	@case "$(STAGE)" in scalar|e8p|trellis) ;; *) \
	  echo "STAGE must be scalar, e8p or trellis (got '$(STAGE)')"; exit 1;; esac
	@if [ "$(STAGE)" = "trellis" ]; then \
	  $(PY) -c "import torch, sys; \
sys.exit(0) if torch.cuda.is_available() else \
(print('### no CUDA device. The trellis stage is exact Viterbi over'), \
 print('###   2**$(TRELLIS_BITS) states per weight; on CPU this is not a slower'), \
 print('###   conversion, it is an impractical one. Refusing.'), sys.exit(1))"; \
	fi

# Turn any Hugging Face causal LM into a progressive artifact. Data-free --
# no calibration corpus, no Hessian, no forward pass -- so exactly one weight
# matrix is resident at a time regardless of model size (measured under 2 GiB
# on the widest Llama-3.1-70B weight, against 141 GB for that model dense).
# Encode once at the deepest PASSES you want: shallower tiers are bit-identical
# prefixes of it (proposition 2), so there's no reason to convert twice.
convert: convert-preflight
	@mkdir -p $(OUT)
	@echo "### $(MODEL) -> $(OUT)  [$$(date -u +%H:%M:%SZ)]"
	@echo "### stage=$(STAGE) passes=$(PASSES) bits=$(BITS) group_size=$(GROUP_SIZE) trellis_bits=$(TRELLIS_BITS) seed=$(SEED)"
	@$(PY) $(ROOT)/experiments/quantise_hf_progressive.py \
	  --model $(MODEL) --out $(OUT) \
	  --stage $(STAGE) --passes $(PASSES) --bits $(BITS) \
	  --group-size $(GROUP_SIZE) --seed $(SEED) \
	  --trellis-bits $(TRELLIS_BITS) \
	  $(if $(EXCLUDE),--exclude '$(EXCLUDE)',) \
	  $(if $(filter 1,$(KEEP_SHARDS)),--keep-shards,)
	@echo "### done  [$$(date -u +%H:%M:%SZ)]  artifact in $(OUT)"

clean:
	@find $(ROOT) -name '__pycache__' -type d -prune -exec rm -rf {} +
	@rm -rf $(ROOT)/.pytest_cache $(ROOT)/$(LLM_OUT)
