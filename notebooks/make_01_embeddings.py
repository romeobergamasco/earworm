"""Writes notebooks/01_embeddings.ipynb (T18: CLAP or MERT?), cells only.

    python notebooks/make_01_embeddings.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_embeddings.ipynb

Data: notebooks/t18_embed.py (catalogue, via YouTube) and notebooks/t18_fma.py
(FMA-small sample). Only vectors are stored, never audio.
"""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("""# T18 — Choix du modèle d'embedding : CLAP ou MERT ?

Les **sons similaires**, et plus tard le **genre prédit**, reposent sur un *embedding* : un vecteur qui résume le son d'un morceau. Deux modèles pré-entraînés sont candidats.

| | CLAP (`laion/clap-htsat-unfused`) | MERT (`m-a-p/MERT-v1-95M`) |
|---|---|---|
| Entraînement | contrastif audio ↔ texte | auto-supervisé, sur de la musique |
| Entrée | 48 kHz | 24 kHz |
| Vecteur | 512 dimensions | 13 couches × 768 dimensions |

**Recette** (`earworm/embeddings.py`, la même que l'indexeur) : chaque morceau est coupé en segments de 10 s, jusqu'à 6, répartis entre 10 % et 90 % de sa durée. L'embedding du morceau est la moyenne normalisée de ses segments. L'audio est effacé juste après le calcul : seuls les vecteurs sont gardés.

**Deux jeux de données** :
1. **Le catalogue Earworm** : les morceaux qui ont une classe de genre (taxonomie v1, T20). C'est le cas d'usage réel.
2. **FMA-small** : 50 extraits de 30 s par genre, 8 genres équilibrés. C'est un jeu public, aux étiquettes plus propres, qui sert de contrôle indépendant.

**Règle de décision, fixée avant les mesures** : CLAP est le choix par défaut de la spec. MERT ne le remplace que s'il fait **mieux en précision@5 de genre sur le catalogue**, avec un intervalle de confiance à 95 % (bootstrap apparié) qui **exclut zéro**, **et** s'il n'est **pas moins bon sur FMA**. En effet, MERT coûte environ 10 fois plus de calcul par morceau, et ses vecteurs sont plus gros (768 dimensions contre 512)."""),
    code("""import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from indexer.genres import label, taxonomy

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
EMB = ROOT / "data" / "embeddings"
names = taxonomy().names


def load(folder, items, key_id, labels, artists):
    ids, clap, mert, clap_s, mert_s, y, art = [], [], [], [], [], [], []
    for it in items:
        f = folder / f"{it[key_id]}.npz"
        lab = labels(it)
        if not f.exists() or not lab:
            continue
        with np.load(f) as z:
            ids.append(it[key_id]); clap.append(z["clap"].astype(np.float32)); mert.append(z["mert"].astype(np.float32))
            clap_s.append(float(z["clap_s"])); mert_s.append(float(z["mert_s"]))
        y.append(lab); art.append(artists(it))
    return {"ids": ids, "clap_seg": clap, "mert_seg": mert, "clap_s": np.array(clap_s),
            "mert_s": np.array(mert_s), "y": np.array(y), "artist": np.array(art)}


songs = json.loads((EMB / "t18" / "songs.json").read_text("utf-8"))
cat = load(EMB / "t18", songs, "id", lambda s: label(s["mb_genres"], s["mb_tags"]).label,
           lambda s: s["artist"].lower().split(" feat")[0].split(" & ")[0].strip())
tracks = json.loads((EMB / "t18_fma" / "tracks.json").read_text("utf-8"))
fma = load(EMB / "t18_fma", tracks, "track_id", lambda t: t["genre"], lambda t: t["artist"])
print(f"catalogue : {len(cat['ids'])} morceaux étiquetés sur {len(songs)} prévus (téléchargement YouTube en cours si moins)")
print(f"FMA       : {len(fma['ids'])} extraits sur {len(tracks)}")"""),
    md("""## Métriques

- **précision@5** : sur les 5 plus proches voisins (cosinus), la part qui a **la même classe de genre**. **Les morceaux du même artiste sont exclus des voisins** : sinon on mesurerait surtout la reconnaissance de l'artiste, pas du style.
- **k-NN** : la classe majoritaire parmi les 5 voisins est-elle la bonne ?
- **sonde linéaire** : F1 macro d'une régression logistique en validation croisée à 5 plis, **groupés par artiste**. C'est un avant-goût du classifieur de T21.
- Les vecteurs sont **centrés** (on retire la moyenne du jeu) avant le cosinus. Sans ça, avec MERT, tous les morceaux se ressemblent à 0,9 de cosinus, et le voisinage est dominé par cette direction commune. La version non centrée est aussi mesurée.
- **MERT** donne 13 couches. La comparaison principale utilise la **moyenne des 13 couches**, fixée à l'avance. La meilleure couche est montrée à titre indicatif : la choisir sur ces mêmes données avantagerait MERT."""),
    code("""def song_vec(seg):  # mean of L2-normalised segments, L2-normalised (earworm.embeddings.song_embedding)
    v = seg / np.linalg.norm(seg, axis=-1, keepdims=True)
    m = v.mean(0)
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def prep(X, center=True):
    X = X - X.mean(0) if center else X
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def neighbours(X, artist, k=5):
    S = X @ X.T
    S[artist[:, None] == artist[None, :]] = -np.inf  # itself and the same artist
    return np.argsort(-S, axis=1)[:, :k]


def probe(X, y, artist):
    keep = np.isin(y, [c for c, n in Counter(y).items() if n >= 8])
    if len(set(y[keep])) < 2:  # too few songs yet
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced"))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(clf, X[keep], y[keep], groups=artist[keep], cv=cv)
    return f1_score(y[keep], pred, average="macro")


def scores(X, y, artist):
    nn = neighbours(X, artist)
    prec = (y[nn] == y[:, None]).mean(1)
    vote = np.array([Counter(y[r]).most_common(1)[0][0] for r in nn])
    return prec, {"précision@5": prec.mean(), "k-NN": (vote == y).mean(), "sonde (F1 macro)": probe(X, y, artist)}


def prepare(d):
    d["clap"] = np.stack([song_vec(s) for s in d["clap_seg"]])
    d["mert_layers"] = np.stack([song_vec(s) for s in d["mert_seg"]])  # (n, 13, 768)
    d["mert"] = d["mert_layers"].mean(1)
    return d


def compare(d, title):
    y, artist = d["y"], d["artist"]
    chance = sum((n / len(y)) ** 2 for n in Counter(y).values())
    layer_prec = [scores(prep(d["mert_layers"][:, l]), y, artist)[0].mean() for l in range(13)]
    best = int(np.argmax(layer_prec))
    variants = {"CLAP": prep(d["clap"]), "CLAP non centré": prep(d["clap"], False),
                "MERT (13 couches)": prep(d["mert"]), "MERT non centré": prep(d["mert"], False),
                f"MERT couche {best} (indicatif)": prep(d["mert_layers"][:, best])}
    per_query = {}
    print(f"### {title} — {len(y)} morceaux, {len(set(artist))} artistes, hasard précision@5 ≈ {chance:.3f}\\n")
    print("| variante | précision@5 | k-NN | sonde (F1 macro) |\\n|---|---|---|---|")
    for name, X in variants.items():
        prec, s = scores(X, y, artist)
        per_query[name] = prec
        print(f"| {name} | " + " | ".join(f"{v:.3f}" for v in s.values()) + " |")
    a, b = per_query["CLAP"], per_query["MERT (13 couches)"]
    rng = np.random.default_rng(0)
    diffs = [(b[i] - a[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(5000))]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\\nMERT − CLAP (précision@5) : {(b - a).mean():+.3f}, IC 95 % [{lo:+.3f}, {hi:+.3f}]")
    plt.figure(figsize=(7, 2.8))
    plt.plot(range(13), layer_prec, "o-", label="MERT, couche seule")
    plt.axhline(per_query["MERT (13 couches)"].mean(), color="C1", ls="--", label="MERT, 13 couches")
    plt.axhline(per_query["CLAP"].mean(), color="C2", ls=":", label="CLAP")
    plt.axhline(chance, color="grey", lw=0.8, label="hasard")
    plt.title(title, fontsize=10); plt.xlabel("couche de MERT"); plt.ylabel("précision@5")
    plt.legend(fontsize=7); plt.tight_layout(); plt.show()
    return {"diff": (b - a).mean(), "ci": (lo, hi)}"""),
    md("## Catalogue Earworm"),
    code("""cat = prepare(cat)
res_cat = compare(cat, "Catalogue Earworm (taxonomie v1)")"""),
    md("## FMA-small"),
    code("""fma = prepare(fma)
res_fma = compare(fma, "FMA-small (8 genres)")"""),
    md("## Coût"),
    code("""both = np.concatenate([cat["clap_s"], fma["clap_s"]]), np.concatenate([cat["mert_s"], fma["mert_s"]])
print("| | CLAP | MERT |\\n|---|---|---|")
print(f"| CPU par morceau du catalogue (médiane, 6 segments, 6 cœurs) | {np.median(cat['clap_s']):.1f} s | {np.median(cat['mert_s']):.1f} s |")
print(f"| CPU pour 1 000 morceaux | {np.median(cat['clap_s']) * 1000 / 60:.0f} min | {np.median(cat['mert_s']) * 1000 / 60:.0f} min |")
print("| taille du vecteur | 512 | 768 |")
print(f"| stockage pour 1 000 morceaux (float32) | {512 * 4 * 1000 / 1e6:.1f} Mo | {768 * 4 * 1000 / 1e6:.1f} Mo |")"""),
    md("""## Écoute : 10 listes de voisins

Dix morceaux du catalogue tirés au hasard, avec leurs 5 plus proches voisins pour chaque modèle (hors même artiste). C'est le critère qualitatif du ticket. Les listes se lisent comme des recommandations : aurait-on envie d'écouter ces morceaux juste après ?"""),
    code("""meta = {s["id"]: s for s in songs}
name = lambda i: f"{meta[cat['ids'][i]]['artist']} – {meta[cat['ids'][i]]['title']} ({meta[cat['ids'][i]]['year'] or '?'})"
nn_c = neighbours(prep(cat["clap"]), cat["artist"])
nn_m = neighbours(prep(cat["mert"]), cat["artist"])
for i in np.random.default_rng(1).choice(len(cat["ids"]), min(10, len(cat["ids"])), replace=False):
    print(f"\\n▶ {name(i)}  [{names[cat['y'][i]]}]")
    for c, m in zip(nn_c[i], nn_m[i]):
        print(f"   CLAP  {name(c)[:50]:50s} [{cat['y'][c]}]   |   MERT  {name(m)[:50]:50s} [{cat['y'][m]}]")"""),
    md("## Décision"),
    code("""mert_wins = res_cat["ci"][0] > 0 and res_fma["diff"] >= 0
print("catalogue : MERT − CLAP = %+.3f, IC [%+.3f, %+.3f]" % (res_cat["diff"], *res_cat["ci"]))
print("FMA       : MERT − CLAP = %+.3f, IC [%+.3f, %+.3f]" % (res_fma["diff"], *res_fma["ci"]))
print("→ modèle retenu :", "MERT" if mert_wins else "CLAP (choix par défaut confirmé)")"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
    )
    nbf.write(nb, Path(__file__).with_name("01_embeddings.ipynb"))


if __name__ == "__main__":
    main()
