"""Model G: maps (speech features, human motion, robot capabilities) to cue events."""
import torch
import torch.nn as nn

from vera.config import GConfig, ENABLE

DISABLED_LOGIT = -1e4  # sigmoid(-1e4) == 0.0 exactly in float32


class ModelG(nn.Module):
    def __init__(self, cfg: GConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # Input projections: everything becomes width D.
        self.speech_proj = nn.Linear(cfg.d_speech, D)   # x   [B,T,512] -> [B,T,D]
        self.motion_proj = nn.Linear(cfg.n_motion, D)   # m_h [B,T,8]   -> [B,T,D]
        self.cap_proj = nn.Linear(cfg.d_cap, D)         # b_m [B,6,5]   -> [B,6,D]
        self.cap_id = nn.Embedding(cfg.n_cue_heads, D)  # tells each capability token which head it is
        self.pos = nn.Embedding(cfg.max_frames, D)      # frame position 0..T-1

        # Trunk: standard Transformer layers. Causality comes from the mask you pass in.
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.n_attn_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)

        # Output: one vector per frame -> 6 heads x 11 values.
        self.out = nn.Linear(D, cfg.n_cue_heads * cfg.d_out)

    def forward(self, x: torch.Tensor, m_h: torch.Tensor, b_m: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, T, d_speech]  speech features
        m_h: [B, T, n_motion]  human facial motion
        b_m: [B, H, d_cap]     capability descriptor, H = 6 cue heads; b_m[..., 0] = enabled
        returns [B, T, H, 11]
        """
        B, T, _ = x.shape
        H = self.cfg.n_cue_heads

        dev = x.device

        # 1. Frame tokens [B, T, D]: speech + human motion + frame position.
        frames = self.speech_proj(x) + self.motion_proj(m_h) + self.pos(torch.arange(T, device=dev))

        # 2. Capability tokens [B, H, D]: what each cue head can do + which head it is.
        caps = self.cap_proj(b_m) + self.cap_id.weight

        # 3. Sequence [B, H + T, D]: capability tokens first, then frames.
        seq = torch.cat([caps, frames], dim=1)

        # 4. Attention mask [H + T, H + T], True = blocked.
        #    Capability tokens see only each other (seeing frames would leak the future);
        #    frame t sees every capability token and frames 0..t.
        mask = torch.ones(H + T, H + T, dtype=torch.bool, device=dev)
        mask[:, :H] = False
        mask[H:, H:] = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), diagonal=1)

        # 5. Trunk, keeping only the frame positions.
        h = self.trunk(seq, mask=mask)[:, H:]

        # 6. Project to 6 heads x 11 values.
        out = self.out(h).view(B, T, H, self.cfg.d_out)

        # 7. Hard capability mask: a disabled head's enable logit is DISABLED_LOGIT on every frame.
        disabled = (b_m[..., 0] < 0.5)[:, None, :]                  # [B, 1, H]
        enable = out[..., ENABLE].masked_fill(disabled, DISABLED_LOGIT)
        return torch.cat([enable.unsqueeze(-1), out[..., ENABLE + 1:]], dim=-1)
