"""All Model G dimensions in one place. Change sizes here, not in the model."""
from dataclasses import dataclass

# The 6 cue heads G can fire, in output order.
CUE_HEADS = ("gaze", "nod", "jaw", "lids", "brow_left", "brow_right")

# Communicative roles each event can carry.
ROLES = ("attending", "acknowledging", "articulating", "yielding")

# Per-head capability descriptor b_m, in feature order.
CAP_FEATURES = ("enabled", "min", "max", "max_speed", "latency")

# Layout of the 11 values each head predicts per frame.
ENABLE = 0              # 1 logit: does this head fire now?
ROLE = slice(1, 5)      # 4 logits: which role
PARAMS = slice(5, 9)    # 4 continuous cue parameters (normalized target, etc.)
ONSET = 9               # seconds until the cue starts
DURATION = 10           # how long the cue lasts


@dataclass
class GConfig:
    # Inputs
    d_speech: int = 512          # x_t: speech feature size (PersonaPlex stand-in)
    n_motion: int = 8            # M_h: human facial motion channels
    n_cue_heads: int = len(CUE_HEADS)
    d_cap: int = len(CAP_FEATURES)

    # Outputs
    n_roles: int = len(ROLES)
    n_params: int = 4

    # Transformer trunk
    d_model: int = 256
    n_layers: int = 4
    n_attn_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.0
    max_frames: int = 256        # longest context, in frames
    frame_rate_hz: float = 12.5  # Moshi/PersonaPlex frame rate

    @property
    def d_out(self) -> int:
        """Values per head per frame: enable + roles + params + onset + duration = 11."""
        return 1 + self.n_roles + self.n_params + 2
