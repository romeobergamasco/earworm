"""Audio embeddings from pretrained models (similar songs, genre).

A song is cut into 10 s segments spread over its whole length; each model turns
a segment into a vector. The song's embedding is the L2-normalised mean of its
segment vectors (docs/spec.md). Models run locally on CPU, at indexing time:
the server never loads them.

Both models are pinned to the exact hub revision that was reviewed (MERT ships
its own model code, run with trust_remote_code), so they cannot change silently.
"""

import time
from dataclasses import dataclass

import numpy as np

SEGMENT_S = 10.0
MAX_SEGMENTS = 6

CLAP_ID = "laion/clap-htsat-unfused"
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
MERT_ID = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"  # code read on 2026-11-14


def segment_starts(duration_s: float, seconds: float = SEGMENT_S, n_max: int = MAX_SEGMENTS):
    """Start times of up to n_max segments, evenly spread between 10 % and 90 %
    of the song (skips intros and fade-outs); one segment per 30 s of audio."""
    if duration_s <= seconds:
        return [0.0]
    n = int(min(n_max, max(1, duration_s // 30)))
    lo, hi = 0.1 * duration_s, 0.9 * duration_s - seconds
    if hi <= lo:
        return [max(0.0, (duration_s - seconds) / 2)]
    return list(np.linspace(lo, hi, n)) if n > 1 else [(lo + hi) / 2]


def segments(y: np.ndarray, sr: int, seconds: float = SEGMENT_S, n_max: int = MAX_SEGMENTS):
    """(n, seconds * sr) float32 array of segments (the last one zero-padded if short)."""
    length = int(seconds * sr)
    out = []
    for s in segment_starts(len(y) / sr, seconds, n_max):
        seg = y[int(s * sr) : int(s * sr) + length]
        out.append(np.pad(seg, (0, length - len(seg))))
    return np.stack(out).astype(np.float32)


def song_embedding(segment_vectors: np.ndarray) -> np.ndarray:
    """Mean of the (L2-normalised) segment vectors, L2-normalised."""
    v = segment_vectors / np.linalg.norm(segment_vectors, axis=-1, keepdims=True)
    m = v.mean(axis=0)
    return (m / np.linalg.norm(m)).astype(np.float32)


@dataclass
class Timed:
    vectors: np.ndarray
    seconds: float


class ClapEmbedder:
    """CLAP audio tower: one 512-d vector per segment, from 48 kHz audio."""

    sr = 48_000
    name = "clap-htsat-unfused"

    def __init__(self) -> None:
        import torch
        from transformers import ClapFeatureExtractor, ClapModel

        self._torch = torch
        self.features = ClapFeatureExtractor.from_pretrained(CLAP_ID, revision=CLAP_REVISION)
        self.model = ClapModel.from_pretrained(CLAP_ID, revision=CLAP_REVISION).eval()

    def embed(self, segs: np.ndarray) -> Timed:
        t0 = time.perf_counter()
        inputs = self.features(list(segs), sampling_rate=self.sr, return_tensors="pt")
        with self._torch.inference_mode():
            out = self.model.get_audio_features(**inputs)
        out = getattr(out, "pooler_output", out)  # tensor, or a model output in newer versions
        return Timed(out.numpy().astype(np.float32), time.perf_counter() - t0)


class MertEmbedder:
    """MERT-v1-95M: per segment, the time-averaged output of each of its 13
    layers (13 x 768), from 24 kHz audio. Which layers to use is decided later."""

    sr = 24_000
    name = "mert-v1-95m"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self._torch = torch
        self.features = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, revision=MERT_REVISION)
        self.model = AutoModel.from_pretrained(
            MERT_ID, revision=MERT_REVISION, trust_remote_code=True
        ).eval()

    def _layer_outputs(self, input_values):
        """Input of the first transformer layer, then the output of each layer.
        Captured with hooks: with transformers 5, MERT's own code no longer
        returns `hidden_states` (its weights all load, checked)."""
        layers = self.model.encoder.layers
        seen = []
        hooks = [layers[0].register_forward_pre_hook(lambda _m, args: seen.append(args[0]))]
        for layer in layers:
            hooks.append(
                layer.register_forward_hook(
                    lambda _m, _a, out: seen.append(out[0] if isinstance(out, tuple) else out)
                )
            )
        try:
            self.model(input_values)
        finally:
            for h in hooks:
                h.remove()
        return seen  # 13 tensors (n, frames, 768)

    def embed(self, segs: np.ndarray) -> Timed:
        t0 = time.perf_counter()
        inputs = self.features(list(segs), sampling_rate=self.sr, return_tensors="pt")
        with self._torch.inference_mode():
            hidden = self._layer_outputs(inputs["input_values"])
        layers = self._torch.stack(hidden).mean(dim=2)  # (13, n, 768)
        vectors = layers.permute(1, 0, 2).numpy().astype(np.float32)  # (n, 13, 768)
        return Timed(vectors, time.perf_counter() - t0)


def embed_segments(path, embedder) -> np.ndarray:
    """Segment vectors of an audio file (decoded at the model's rate)."""
    from earworm.audio import load

    y, sr = load(path, embedder.sr)
    return embedder.embed(segments(y, sr)).vectors


def embed_file(path, embedder) -> np.ndarray:
    """Song embedding of an audio file."""
    return song_embedding(embed_segments(path, embedder))
