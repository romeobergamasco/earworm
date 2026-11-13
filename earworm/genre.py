"""Genre prediction from segment embeddings, with the model exported by
notebooks/02_genre_classifier.ipynb (T21): plain numpy arrays, no PyTorch nor
scikit-learn needed.

    model = GenreModel.load("models/genre_model.npz")
    pred = model.predict(segment_vectors)   # one song

The song's prediction is the mean of its segments' probabilities (as in T21).
Used at indexing time only: the server reads the stored predictions.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path("models") / "genre_model.npz"


@dataclass(frozen=True)
class Prediction:
    label: str
    proba: float
    top3: list[dict]  # [{"label", "name", "proba"}], best first

    def row(self, song_id: int, model_version: str) -> dict:
        """A row of the genre_predictions table."""
        return {
            "song_id": song_id,
            "model_version": model_version,
            "label": self.label,
            "proba": round(self.proba, 4),
            "top3": self.top3,
        }


class GenreModel:
    def __init__(self, arrays: dict, info: dict, version: str) -> None:
        self.mean, self.scale = arrays["mean"], arrays["scale"]
        self.layers = [(arrays[f"W{i}"], arrays[f"b{i}"]) for i in range(int(arrays["n_layers"]))]
        self.classes = [str(c) for c in arrays["classes"]]
        self.info = info
        self.embed_model = info["embed_model"]
        self.names = info.get("classes", {})
        self.version = version  # model_version stored with each prediction

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "GenreModel":
        path = Path(path)
        raw = path.read_bytes()
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        info = json.loads(str(arrays.pop("info")))
        kind = str(arrays["kind"])
        version = (
            f"{kind}-{info['embed_model']}-tax{info['taxonomy_version']}-"
            f"{hashlib.sha256(raw).hexdigest()[:8]}"
        )
        return cls(arrays, info, version)

    def probas(self, segments: np.ndarray) -> np.ndarray:
        """(n_segments, n_classes) probabilities."""
        h = np.asarray(segments, dtype=np.float32)
        if h.ndim == 3:  # MERT: layers -> mean (as in the T21 export)
            h = h.mean(axis=1)
        h = (h - self.mean) / self.scale
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        p = np.exp(h - h.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, segments: np.ndarray) -> Prediction:
        p = self.probas(segments).mean(axis=0)
        order = np.argsort(-p)
        top3 = [
            {"label": self.classes[i], "name": self.names.get(self.classes[i], self.classes[i]),
             "proba": round(float(p[i]), 4)}
            for i in order[:3]
        ]  # fmt: skip
        return Prediction(self.classes[order[0]], float(p[order[0]]), top3)
