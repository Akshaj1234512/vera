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

## Evaluation, ablations and state extraction

- **Δ-User pairing**: the shuffled condition scores (GT partner, agent generated from a *shuffled* partner) against (GT partner, GT agent). This is the only pairing where a model that ignores partner motion gets exactly 0%, as Table 2 states for the baselines **[A]**. Shuffling is a seeded derangement **[A]**.
- **BeatAlign**: ViBES `alignment(0.3, 7, mmae)`: speed by central differences, normalized by per-joint mean speed (refit on Seamless GT; ViBES uses a BEAT2 file), argrelextrema order 7 (kept at 25 fps), speed threshold 0.3, librosa onsets with hop 512 at 24 kHz, audio-anchored Gaussian hit rate with σ = 0.3 s, mean over the 13 upper-body joints. 2 s trimmed at both ends (ViBES: 60 frames at 30 fps) **[I/A]**.
- **Diversity**: A2P `calculate_diversity` (10,000 random frame pairs, which needs more than 10,000 frames); the paper reports the mean **[I]**.
- **FGD / P-FD units**: joint positions in metres; values reported ×10³ like Table 2 **[A]**.
- **w/o Partner**: same 46-token layout; [B] positions never attend [A] positions in self-attention, and only B is supervised. Dropping the A slots would be the other reading; masking keeps the time base identical **[A]**.
- **State extraction**:
  - Both role orders are extracted **[A]**.
  - Voice clip: the agent's own 10 s window with the most agent and least partner speech by VAD, normalized to −24 LUFS as `LMGen.load_voice_prompt` does **[A selection / I normalization]**.
  - Clips longer than the context minus the prompt are cut into consecutive chunks, each with a fresh prompt **[A]**. The pinned 3000-frame context fits about 3.8 min of conversation per chunk.
  - EPAD sits on the frame before each word's first token **[A]**.
- **Runtime**: the §5 1024-token setting keeps the speech window at 89 frames (`speech_window_frames`) **[P]**. Precision is unstated; we measure fp32 **[A]**.

## Training pipeline choices (GPU runbook)

- **Splits**: train = official train, val = official dev (codec checkpoint selection, tower validation), test = official test pairs passing the audio check. `test330` = the first 330 test pairs by sha1(pair key). The paper's 330 ids are unpublished, so Table 2 numbers are not directly comparable **[A]**.
- **Voice-clip filter**: a pair is dropped if either participant's best 10 s window is less than 50% their own speech by VAD. The paper drops ~1.4% for "no voice clip" but gives no rule **[A]**.
- **Audio check (test only)**: both WAVs readable, RMS above −50 dBFS, at least 5% VAD speech, and length within 2 s of the motion **[A]**.
- **Normalization**: per-part mean/std over valid train frames (the face part uses face-filtered pairs only) **[A]**.
- **Speech states during training**: computed online per micro-batch over [prompt + chunk up to the last window]. A clip is split into chunks of (3000 − prompt) frames, each with a fresh prompt. Windows are 89 frames inside one chunk, so the window's speech history goes back to the chunk start, not just the window **[A]**.
- **Label masking**: codes are not supervised where the owning speaker's body frame (body codes) or face frame (face codes) is invalid **[A]**. The paper only says invalid frames are excluded from training input.
- **Precision / parallelism**: bf16 autocast, TF32, DDP; each rank holds a frozen bf16 PersonaPlex; optional activation checkpointing **[A]**.
- **Codec held-out set**: the val split. The face codec is held out on all val participants' face-valid windows, since the 20% rule restricts training only **[A]**.

## Open questions (resolve on the real run or by asking the authors)

1. **BeatAlign scale.** Our ViBES-faithful BeatAlign gives about 0.5–0.6 on ground-truth smoke clips; the paper's GT row is 0.049. A unit mismatch would reproduce numbers that low (motion beats in frames against audio onsets in seconds, with σ = 0.3). Check ViBES's published numbers with its own code, and compare our GT value on the real test set.
2. **FD numeric floor.** With zero root translation the root joint is constant, so the 66-D covariance is singular. `sqrtm` then gives a floor of about −1e-4 (−0.13 on the ×10³ scale) even for identical sets. ViBES's code behaves the same; we keep it.
3. **Top-k**: 200 (§4.4) vs none (D.2). Evaluate both.
4. **Speech/motion offset** (`state_offset`): 0 by our definition; sweep −1/0/+1 and compare against Fig. 5's diagonal.

## Implementation notes (no effect on the maths)

- Attention uses PyTorch's fused `scaled_dot_product_attention` with a boolean mask. Masked keys still get exactly zero weight, so the causality checks hold bit-for-bit. An explicit path is kept for attention maps (Fig. 5).
- The codec's reconstruction loss is `SmoothL1Loss(beta=1, mean)` written out explicitly; the MPS kernel fails on some shapes. The values are identical.
- Face features are built at 25 fps from resampled FAU values; resampling 6D vectors linearly would break orthonormality. Degenerate (all-zero) 6D vectors map to the identity rotation.
- Movement features use `movement_v4` when at least 50% of its frames are valid, otherwise `movement` v1 (same rule as `speech_ai.seamless`).

## Environment

- Local runs import PersonaPlex at `3428dfd` with torch 2.15 / numpy 2.5, outside its pins (torch < 2.5, numpy < 2.2). That is fine for the tiny LM. Real 7B runs should use NVIDIA's own environment on a CUDA host.
- GestureLSM `QuantizeEMAReset.reset_codebook` hard-codes `.cuda()`. `codec.py` swaps in a device-agnostic copy with identical logic.
