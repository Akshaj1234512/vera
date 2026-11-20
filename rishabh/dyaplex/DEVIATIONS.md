# DyaPlex reproduction: deviations, assumptions and stand-ins

Paper: arXiv 2606.03874v1. Upstream pins: PersonaPlex `3428dfd` (code), `nvidia/personaplex-7b-v1@fdaf409` (weights, gated), GestureLSM `f52ac2f`.
Tags: **[A]** assumption where the paper is silent, **[S]** stand-in used only by the tiny end-to-end test, **[X]** paper inconsistency resolved by choice.

## Stand-ins (tiny end-to-end test only; replace before any real run)

| What | Paper | Here | Replace with |
|---|---|---|---|
| Speech tower weights | PersonaPlex-7B, d_s=4096 | Same `LMModel` class and wiring (32 layers, 17 streams, delays, context 3000), random weights, d_s=64 **[S]** | `speech.load_real_lm()` on a CUDA host |
| Audio tokens | Mimi, 8 codebooks at 12.5 Hz | 8 log band energies per 80 ms, quantized to 2048 bins **[S]** | `mimi.encode` of 24 kHz audio |
| Inner monologue | SentencePiece subwords at word onsets, PAD/EPAD | PAD everywhere; `inner_monologue()` is implemented but unused **[S]** | Seamless transcripts + `tokenizer_spm_32k_3.model` |
| Text prompt | `<system> You enjoy having a good conversation. <system>` | 12 placeholder ids **[S]** | Tokenized prompt |
| Voice prompt | Clean ~10 s clip of the agent, Mimi-tokenized | The agent's first 10 s, stand-in tokens **[S]** | Clean clip selection (procedure unpublished) |
| Face track | SPECTRE → FLAME 6D jaw + 50 expression | 6D jaw from Seamless JawDrop, 24 normalized FAUs in expression slots 0-23 **[S]** | SPECTRE on source video (needs FLAME license) |
| FLAME → rig action units | — | Identity selection of the FAU slots **[S]** | `gr-fit-map` ridge fit on paired clips |
| Sizes | d_m=1024, h=16, h_c=12, window 4096 | d_m=128, h=4, h_c=4, d_c=32, window 1024 (22 frames) **[S]** | `config.REFERENCE` |
| Training | Codecs: 300k iters; tower: 30k iters × batch 512 on 64 H100 | Codecs 600 iters, batch 64; tower 1,600 iters total, batch 4, on 2 pairs (resumed across passes; see `runs/dyaplex-e2e/REPORT.md`) **[S]** | Full runs |
| Tower objective in the tiny test | Body-only (18 codes) for benchmarks; face variant for demos | Face-enabled variant (`supervise="all"`), because the sim drives a face. Under the body-only objective the 4 face codes are never trained and the rig's jaw ends up uncorrelated with speech **[S]** | `REFERENCE` (body) / `REFERENCE_FACE_TOWER` |

## Assumptions (apply to the reference configuration too)

- **Validity**: invalid SMPL-H frames stay on the timeline, filled by slerp, and are masked from statistics, codec windows and losses. Dropping them outright would desynchronize motion and audio **[A]**.
- **Smoothing / rescaling**: Savitzky–Golay window 9, order 2 on translation at 30 fps; translation rescale factor 1 **[A]**.
- **Resampling** 30 → 25 fps: slerp for rotations, linear for translation and face **[A]**.
- **SMPL-H → SMPL-X**: body 0–21 unchanged, hands → 25–54, jaw 22 from the face track, eyes 23–24 identity **[A]**.
- **Normalization**: per-part mean/std refit on the training split (GestureLSM ships BEATX stats) **[A]**.
- **Role assignment**: A = first participant id, B = second; training uses both orders **[A]**.
- **Speech time base**: H[s] = state from the `LMGen.step` call that received frame s (= `forward_train` position s−1). Verified identical between teacher-forced and streaming paths. `state_offset` is configurable because the paper's exact offset is unpublished **[A]**.
- **Tower internals**: pre-norm RMSNorm; SwiGLU hidden = 2·int(4.125·d)/3 (Moshi `gating.py`); cross-attention output projection h_c·d_c → d_m; RoFormer adjacent-pair RoPE with max period 10,000 on flat token index (self-attention) and on frame index (cross-attention, Eq. 3); separate LM head over V = 4098; N(0, 0.02) init **[A]**.
- **Speech window**: keys with q − 89 < s ≤ q (Eq. 4 plus the 4096-token window); training windows equal the motion window **[A]**.
- **Self-attention window at inference**: KV cache keeps the last `window_tokens` keys (sliding window). The paper's "receptive field bounded to 89 frames" could also mean a hard re-computed window; the two agree while a session is shorter than the window **[A]**.
- **Supervision**: body-only model supervises 18 body codes of **both** speakers. The paper says "18 positions per frame", but both-speaker generation needs A predictions **[X-6]**. The face variant uses `supervise="all"` **[A]**.
- **Layout**: both variants use K = 22 / 46 tokens per frame, which is the only layout consistent with 89 frames ≈ 7.1 s and 1024 tokens ≈ 1.8 s **[X-2]**.
- **VA head**: label at each code position = VAD of the speaker who owns that code, at that frame center **[A]**.
- **Sampling**: τ = 1.0; top-k = 200 by default (§4.4). D.2 says no top-k; set `SampleConfig(top_k=None)` **[X-1]**.
- **Optimizer schedule**: constant LR, no warmup (the paper gives none) **[A]**.
- **Codec loss**: GestureLSM passes `--loss-vel 0.5` but hard-codes the velocity term to 0; we inherit 0 **[I]**.
- **PersonaPlex revision**: `3428dfd` (last before the paper). It has context 3000 (HEAD has 2048) **[A]**.

## Implementation notes (no effect on the maths)

- Attention uses PyTorch's fused `scaled_dot_product_attention` with a boolean mask. Masked keys still get exactly zero weight, so the causality checks hold bit-for-bit. An explicit path is kept for attention maps (Fig. 5).
- The codec's reconstruction loss is `SmoothL1Loss(beta=1, mean)` written out explicitly; the MPS kernel fails on some shapes. The values are identical.
- Face features are built at 25 fps from resampled FAU values; resampling 6D vectors linearly would break orthonormality. Degenerate (all-zero) 6D vectors map to the identity rotation.
- Movement features use `movement_v4` when at least 50% of its frames are valid, otherwise `movement` v1 (same rule as `speech_ai.seamless`).

## Environment

- Local runs import PersonaPlex at `3428dfd` with torch 2.15 / numpy 2.5, outside its pins (torch < 2.5, numpy < 2.2). That is fine for the tiny LM. Real 7B runs should use NVIDIA's own environment on a CUDA host.
- GestureLSM `QuantizeEMAReset.reset_codebook` hard-codes `.cuda()`. `codec.py` swaps in a device-agnostic copy with identical logic.
