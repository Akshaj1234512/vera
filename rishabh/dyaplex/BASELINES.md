# Baselines (App. E.2–E.3): what is prepared and what remains

Prepared here:
- `python -m dyaplex export-baselines` writes the same filtered Seamless body data in each baseline's representation (25 fps):
  - DualTalk: 69-D axis-angle = 63 body + 3 global orientation + 3 translation.
  - A2P: 132-D, 22 joints × 6D.
  - Plus validity masks and both speakers' audio paths.
- `python -m dyaplex evaluate --predictions <dir>` scores a baseline's outputs with exactly the same metrics, FK and test pairs as DyaPlex. Write one `<pair>__<agent>.npz` per test pair, holding `pose6d [T, 22, 6]` or `axis_angle [T, 22, 3]` at 25 fps. Δ-User is 0% by design for these baselines, so `*_shuffled` arrays are optional.

Not done: changing the baseline codebases themselves. Exact changes from the paper:

## DualTalk (github.com/ZiqiaoPeng/DualTalk @ 5ed7187)

- Keep the four modules at the original depth: 2 LSTM layers, 3 transformer encoder layers, 1 transformer decoder layer. Keep the `facebook/wav2vec2-large-960h-lv60-self` audio backbone [P E.2].
- Replace the 56-D blendshape output (50 expression + 3 jaw + 3 neck) with 69-D SMPL-X: 63 body pose + 3 global orientation + 3 translation, all axis-angle (`dualtalk_69`). Resize the motion encoder and output head to match.
- Training:
  - wav2vec2's convolutional feature extractor stays frozen; its other layers are fine-tuned. Everything else is trained from random init.
  - Loss is DualTalk's original: per-component MSE plus a frame-to-frame velocity term.
  - Adam, lr 1e-4, on 8 GPUs. Batch size and step count are not reported [U].
- Expected outcome per the paper: collapse to a near-constant pose (per-frame velocity std below 1e-5).

## Audio2Photoreal (github.com/facebookresearch/audio2photoreal @ eaadca7)

- Keep the three stages: TemporalVertexCodec VQ-VAE, Guide transformer, FiLMTransformer diffusion. Keep the public classifier-free guidance recipe and the frozen vq-wav2vec encoder [P E.3].
- Output is 132-D SMPL-X: 22 joints in 6D (`a2p_132`). Subtract each clip's mean translation; FK uses zero translation. Discard the face branch.
- Frame rate: the code hard-codes audio-feature lengths and conditioning positions for 30 fps. Recompute them for 25 fps from the fixed-stride wav2vec feature extractor for 24-s windows.
- Schedule:
  - Tokenizer: 300,000 iterations, 1 GPU.
  - Guide: 30,000 steps, 1 GPU.
  - Diffusion: 800,000 steps on 8 GPUs (per-GPU batch 4, effective 32).
- Inference: chunked diffusion over consecutive 24-s windows with the released guidance.
