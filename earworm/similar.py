"""Similar songs: nearest neighbours of the song embeddings (cosine).

The catalogue is small (1 000 songs x 512 floats = 2 MB), so the server keeps
every embedding in memory and precomputes each song's neighbours once per
catalogue load: answering is then a dictionary lookup.
"""

import json

import numpy as np

K = 5


def parse_vector(value) -> np.ndarray:
    """pgvector as returned by PostgREST ("[0.1,0.2,...]") or a list."""
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


def to_pgvector(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6g}" for x in np.asarray(v, dtype=np.float32)) + "]"


def neighbours(
    ids: list[int], vectors: np.ndarray, k: int = K, center: bool = True
) -> dict[int, list[int]]:
    """For each song id, the ids of its k most similar songs, best first, itself excluded.

    `center`: subtract the catalogue mean before the cosine. The embeddings of
    music all share a large common direction; removing it makes the
    neighbourhoods reflect what differs between songs (measured in T18).
    """
    if len(ids) == 0:
        return {}
    X = np.asarray(vectors, dtype=np.float32)
    if center and len(ids) > 1:
        X = X - X.mean(axis=0)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(ids) - 1)
    top = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return {ids[i]: [ids[j] for j in row] for i, row in enumerate(top)}
