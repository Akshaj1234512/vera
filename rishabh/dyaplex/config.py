"""DyaPlex constants and configurations (arXiv 2606.03874v1).

Every value is tagged with its source:
  [P §x]  stated in the paper (section / equation / table / appendix)
  [I file] inherited from a pinned upstream file
  [A]     assumption: the paper does not say; chosen here and listed in DEVIATIONS.md
Two configurations are kept side by side: REFERENCE (paper scale) and TINY (the
smallest end-to-end test). TINY changes only sizes and step counts, never wiring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

# --------------------------------------------------------------------------- #
# Rates and token layout
# --------------------------------------------------------------------------- #
SEAMLESS_FPS = 30.0      # [I seamless_interaction README] SMPL-H and movement features at 30 Hz
MOTION_FPS = 25.0        # [P D.3] motion resampled to 25 fps before tokenization
TOKEN_HZ = 12.5          # [P §3.1, §3.2] Mimi frame rate == motion token rate
DOWNSAMPLE = 2           # [P D.3] one strided 2x stage: 25 fps -> 12.5 Hz

PARTS = ("upper", "hands", "lower_trans", "face")      # [P Fig. 2a, D.3 table] code order within a frame [A]
PART_DIM = {"upper": 78, "hands": 180, "lower_trans": 57, "face": 56}          # [P D.3 table]
PART_QUANTIZERS = {"upper": 6, "hands": 6, "lower_trans": 6, "face": 4}       # [P D.3 table]
CODEBOOK_SIZE = 1024                                                         # [P D.3]
PART_BAND = {part: i * CODEBOOK_SIZE for i, part in enumerate(PARTS)}        # [P D.3] disjoint bands
V_MOT = 4096                                                                 # [P §3.2]
K = sum(PART_QUANTIZERS.values())                                            # 22 [P §3.2]
BODY_CODES = 18                                                              # [P §3.2, §4.3]
TAG_A, TAG_B = V_MOT, V_MOT + 1                                              # [P §4.2] V = V_mot + 2
VOCAB = V_MOT + 2
L_STEP = 2 * (K + 1)                                                         # 46 [P Eq. 1]

# SMPL-X joint indices per part [P D.3 table] == [I GestureLSM rvq_beatx_train.py:172-210]
UPPER_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
HAND_JOINTS = tuple(range(25, 55))
LOWER_JOINTS = (0, 1, 2, 4, 5, 7, 8, 10, 11)
JAW_JOINT = 22
EXPRESSION_DIM = 50
SMPLX_JOINTS = 55


def code_slots() -> list[tuple[str, int]]:
    """(part, quantizer index) for each of the K code slots of one speaker, in order."""
    return [(part, q) for part in PARTS for q in range(PART_QUANTIZERS[part])]


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CodecConfig:
    # Architecture: [I GestureLSM train_rvq.sh] except causal + down_t [P D.3 (i), (ii)]
    width: int = 512
    code_dim: int = 128
    depth: int = 3
    dilation_growth_rate: int = 3
    down_t: int = 1              # [P D.3 (ii)] one stage instead of two
    stride_t: int = 2
    causal: bool = True          # [P D.3 (i)]
    activation: str = "relu"
    nb_code: int = CODEBOOK_SIZE
    mu: float = 0.99             # [I rvq_beatx_train.py --mu]
    quantize_dropout_prob: float = 0.2   # [I rvq_beatx_train.py:262]
    # Training: [I train_rvq.sh, rvq_beatx_train.py:287-348]
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    batch_size: int = 256
    window: int = 64
    total_iter: int = 300_000
    warmup_iter: int = 400
    milestones: tuple[int, ...] = (200_000,)
    gamma: float = 0.05
    commit: float = 0.02
    loss_vel: float = 0.0        # [I rvq_beatx_train.py:307,346] upstream hard-codes 0 despite --loss-vel 0.5
    sample_codebook_temp: float = 0.5   # [I models/vq/model.py forward]


@dataclass(frozen=True)
class SpeechConfig:
    # [I personaplex@3428dfd loaders.py _lm_kwargs]
    dim: int = 4096
    num_heads: int = 32
    num_layers: int = 32
    hidden_scale: float = 4.125
    context: int = 3000
    depformer_dim: int = 1024
    depformer_num_heads: int = 16
    depformer_num_layers: int = 6
    text_card: int = 32000
    card: int = 2048
    # [I offline.py:214] 0.5 s silence spacer = int(0.5 * 12.5) frames
    silence_frames: int = 6
    voice_prompt_s: float = 10.0  # [P D.1] ~10 s voice clip
    text_prompt: str = "You enjoy having a good conversation."   # [P D.1], wrapped in <system> tags [I offline.py:83]
    # Speech frame s <-> LMGen.step call that received frame s (see speech.py). [A] offset
    state_offset: int = 0


@dataclass(frozen=True)
class TowerConfig:
    layers: int = 32             # [P §4.2] one per PersonaPlex layer
    d_model: int = 1024          # [P §4.2]
    heads: int = 16              # [P §4.2]
    cross_heads: int = 12        # [P §4.2]
    cross_head_dim: int = 64     # [P §4.2]
    speech_dim: int = 4096       # [P §3.1]
    ff_scale: float = 4.125      # [A] Moshi/PersonaPlex hidden_scale; SwiGLU hidden = 2*ff/3 [I gating.py:56-63]
    rope_max_period: float = 10_000.0   # [A] Moshi max_period
    window_tokens: int = 4096    # [P §4.2] motion window == speech window (89 frames)
    speech_window_frames: int | None = None   # None = window_tokens // 46; §5 runtime keeps 89 at 1024 tokens
    norm_eps: float = 1e-5       # [A] RMSNorm, pre-norm
    supervise: str = "body"      # "body" = 18 codes [P §4.3]; "all" = 22 codes (face variant) [A]
    supervise_speakers: str = "AB"   # [A] both speakers (X-6 in the review)
    beta_va: float = 0.01        # [P §4.3]


@dataclass(frozen=True)
class TowerTrainConfig:
    # [P D.2]
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    batch_size: int = 512
    iterations: int = 30_000


@dataclass(frozen=True)
class SampleConfig:
    temperature: float = 1.0     # [P §4.4, D.2]
    top_k: int | None = 200      # [P §4.4]; D.2 says no top-k (X-1)


# Joint body + hands + face variant used for the qualitative face demos [P §4.3, C.2]
REFERENCE_FACE_TOWER = replace(TowerConfig(), supervise="all")

REFERENCE = {
    "codec": CodecConfig(),
    "speech": SpeechConfig(),
    "tower": TowerConfig(),
    "tower_train": TowerTrainConfig(),
    "sample": SampleConfig(),
}

# Smallest end-to-end test. Wiring is identical; only sizes and steps shrink.
TINY = {
    # Reference codec architecture, short training.
    "codec": replace(CodecConfig(), total_iter=600, milestones=(400,), batch_size=64),
    # Same 32 layers / 17 streams / delays / context as PersonaPlex, random weights, small width.
    "speech": replace(SpeechConfig(), dim=64, num_heads=4, depformer_dim=32, depformer_num_heads=2,
                      depformer_num_layers=2, text_card=512),
    # 32 blocks so cross-attention still pairs one-to-one with the 32 speech layers.
    # Face-enabled variant (supervises all 22 codes): the sim drives a face, and the body-only
    # objective leaves the 4 face codes untrained.
    "tower": replace(TowerConfig(), d_model=128, heads=4, cross_heads=4, cross_head_dim=32, speech_dim=64,
                     window_tokens=1024, supervise="all"),
    "tower_train": replace(TowerTrainConfig(), batch_size=4, iterations=800),
    "sample": SampleConfig(),
}


def describe(config: dict) -> dict:
    return {name: asdict(value) for name, value in config.items()}
