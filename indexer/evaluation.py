"""Measurement tooling (docs/spec.md section 7; T26). Runs locally.

    python -m indexer eval-session [--limit 10] [--note "café, 65 dB(A)"] [--open]
    python -m indexer eval-session --latency --count 50 --note wifi
    python -m indexer eval-report <run_id>
    python -m indexer eval-clips      10 s excerpts of the test set, kept locally
    python -m indexer eval-bench      synthetic non-regression bench on those excerpts

A session: this script opens a run in Supabase and prints a PWA link with
?run_id=...; open it on the phone. For each trial the script shows the song to
play (and where to start); the operator plays it on another device and taps
Listen on the phone. The PWA sends its result and the latency it measured
(server/evals.py); this script pairs the result with the expected song, says
right or wrong, and moves on (or redoes the trial). The report computes the
top-1 precision with its Wilson interval, the false positives, and p50/p95 of
the latency, then saves a Markdown file in docs/eval/reports/.
"""

import math
import os
import secrets
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from indexer.testset import EVAL_DIR, EXCERPT_S, Trial

PWA_URL = os.environ.get("EARWORM_PWA_URL", "https://romeoberga-earworm.static.hf.space")
REPORTS_DIR = EVAL_DIR / "reports"
CLIPS_DIR = Path("data") / "eval" / "clips"  # git-ignored, never published
Z95 = 1.959964


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95 % confidence interval of a proportion k/n (Wilson score interval)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentiles(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    p50, p95 = np.percentile(np.asarray(values, dtype=float), [50, 95])
    return float(p50), float(p95)


# ------------------------------------------------------------------ sessions
def open_run(db, kind: str, note: str | None = None) -> str:
    run_id = secrets.token_urlsafe(12)  # 16 characters, unguessable: it is the session's key
    db.insert("eval_runs", {"id": run_id, "kind": kind, "note": note})
    return run_id


def close_run(db, run_id: str) -> None:
    db.update("eval_runs", {"closed_at": datetime.now(UTC).isoformat()}, id=f"eq.{run_id}")


def pwa_link(run_id: str, base: str = PWA_URL) -> str:
    return f"{base.rstrip('/')}/?run_id={run_id}"


def results(db, run_id: str, after: int = 0) -> list[dict]:
    return db.select(
        "eval_results",
        "*",
        run_id=f"eq.{run_id}",
        discarded="is.false",
        id=f"gt.{after}",
        order="id",
    )


def correct(trial: Trial, row: dict) -> bool:
    if trial.kind == "positive":
        return bool(row["match"]) and row["song_id"] == trial.song_id
    return not row["match"]


def wait_result(
    db,
    run_id: str,
    after: int,
    poll_s: float = 1.5,
    sleep=time.sleep,
    out: Callable[[str], None] = print,
    hint_after_s: float = 45,
) -> dict:
    """The next result sent by the PWA for this run (KeyboardInterrupt to stop waiting).
    If nothing comes for a while, say why it may be (the phone not on the session link)."""
    waited, hinted = 0.0, False
    while True:
        rows = results(db, run_id, after)
        if rows:
            return rows[0]
        if not hinted and waited >= hint_after_s:
            hinted = True
            out("  ... toujours rien reçu du téléphone. Il doit afficher en haut le bandeau")
            out("      « Mode mesure · session ... » : sinon, rescannez le QR code de la session.")
        sleep(poll_s)
        waited += poll_s


def print_link(url: str, out: Callable[[str], None] = print) -> None:
    """The session link, and its QR code to scan with the phone (segno, optional)."""
    try:
        import io

        import segno

        buf = io.StringIO()
        segno.make(url, error="l").terminal(out=buf, compact=True, border=2)
        out(buf.getvalue().rstrip())
    except ImportError:
        pass
    out(f"Sur le téléphone, scannez ce QR code ou ouvrez : {url}")
    out("Le haut de l'appli doit afficher « Mode mesure · session ... ».")


def done_trials(db, run_id: str) -> set[int]:
    return {r["trial"] for r in results(db, run_id) if r.get("trial") is not None}


def session(
    db,
    run_id: str,
    trials: list[Trial],
    titles: dict[int, str],
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    open_url: Callable[[str], object] | None = None,
    wait: Callable[..., dict] = wait_result,
) -> dict[str, int]:
    """Play the trials one by one; pairs each PWA result with its trial."""
    done = done_trials(db, run_id)
    todo = [t for t in trials if t.trial not in done]
    last = max((r["id"] for r in results(db, run_id)), default=0)
    counts = {"right": 0, "wrong": 0, "redone": 0}
    i = 0
    while i < len(todo):
        t = todo[i]
        out("")
        out(f"Essai {i + 1}/{len(todo)} (n° {t.trial}) : « {t.artist} – {t.title} »")
        out(f"  1. Sur cet ordinateur, jouez la vidéo à partir de {t.start} : {t.url}")
        if open_url:
            open_url(t.url)
            out("     (elle vient de s'ouvrir dans le navigateur, déjà calée sur ce moment)")
        out("  2. Sur le téléphone, appuyez sur Écouter.")
        out("  En attente du résultat du téléphone... (Ctrl+C : passer l'essai ou quitter)")
        try:
            row = wait(db, run_id, last)
        except KeyboardInterrupt:
            choice = ask("\n(s)auter cet essai, (q)uitter, ou Entrée pour attendre encore : ")
            if choice.strip().lower() == "q":
                break
            if choice.strip().lower() == "s":
                i += 1
            continue
        last = row["id"]
        db.update(
            "eval_results",
            {"trial": t.trial, "expected_kind": t.kind, "expected_song_id": t.song_id},
            id=f"eq.{row['id']}",
        )
        ok = correct(t, row)
        got = titles.get(row["song_id"], f"#{row['song_id']}") if row["match"] else "non trouvé"
        out(f"  {'✓ juste' if ok else '✗ faux'} : {got}  ({row['latency_ms']} ms)")
        out("  Mettez la vidéo en pause.")
        choice = ask("  Entrée = suivant, r = refaire cet essai, q = quitter : ").strip().lower()
        if choice == "r":
            db.update("eval_results", {"discarded": True}, id=f"eq.{row['id']}")
            counts["redone"] += 1
            continue
        counts["right" if ok else "wrong"] += 1
        if choice == "q":
            break
        i += 1
    return counts


def latency_session(
    db, run_id: str, count: int, out: Callable[[str], None] = print, wait=wait_result
) -> int:
    """No expected song: the operator just taps Listen `count` times (spec 7.2)."""
    last, n = max((r["id"] for r in results(db, run_id)), default=0), 0
    out(f"Appuyez {count} fois sur Écouter sur le téléphone (Ctrl+C pour arrêter).")
    try:
        while n < count:
            row = wait(db, run_id, last)
            last, n = row["id"], n + 1
            out(f"  {n}/{count} : {row['latency_ms']} ms")
    except KeyboardInterrupt:
        pass
    return n


# ------------------------------------------------------------------ report
def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "?"


def _pct(x: float) -> str:
    return f"{x * 100:.0f} %"


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r.get("expected_kind") == "positive"]
    neg = [r for r in rows if r.get("expected_kind") == "negative"]
    right = [r for r in pos if r["match"] and r["song_id"] == r["expected_song_id"]]
    wrong_song = [r for r in pos if r["match"] and r["song_id"] != r["expected_song_id"]]
    fps = [r for r in neg if r["match"]]
    server = [r["server_ms"]["total"] for r in rows if (r.get("server_ms") or {}).get("total")]
    return {
        "trials": len(rows),
        "positives": len(pos),
        "right": len(right),
        "not_found": sum(not r["match"] for r in pos),
        "wrong_song": len(wrong_song),
        "precision": len(right) / len(pos) if pos else None,
        "precision_ci": wilson(len(right), len(pos)) if pos else None,
        "negatives": len(neg),
        "false_positives": len(fps),
        "fp_rate": len(fps) / len(neg) if neg else None,
        "fp_ci": wilson(len(fps), len(neg)) if neg else None,
        "latency": percentiles([r["latency_ms"] for r in rows]),
        "server": percentiles(server),
        "network": percentiles([r["network_ms"] for r in rows if r.get("network_ms") is not None]),
        "wrong_rows": wrong_song + fps,
        "missed_rows": [r for r in pos if not r["match"]],
    }


def report_markdown(
    run: dict, rows: list[dict], titles: dict[int, str], meta: dict, now: datetime | None = None
) -> str:
    s = summarize(rows)
    now = now or datetime.now(UTC)
    lines = [
        f"# Session de mesure `{run['id']}`",
        "",
        f"- Date : {now:%Y-%m-%d %H:%M} UTC (session ouverte le {str(run['created_at'])[:16]})",
        f"- Type : {run['kind']}" + (f" ; note : {run['note']}" if run.get("note") else ""),
        f"- Commit : `{meta.get('commit', '?')}` ; "
        f"`PARAMS_VERSION` = {meta.get('params_version', '?')}",
        f"- Catalogue : {meta.get('catalogue_songs', '?')} morceaux",
        f"- Essais retenus : {s['trials']} (les essais refaits sont exclus)",
        "",
    ]
    if s["positives"] or s["negatives"]:
        lines += ["| mesure | résultat | IC 95 % (Wilson) |", "|---|---|---|"]
        if s["positives"]:
            lo, hi = s["precision_ci"]
            lines.append(
                f"| précision top-1 (positifs) | **{s['right']}/{s['positives']} = "
                f"{_pct(s['precision'])}** | [{_pct(lo)} ; {_pct(hi)}] |"
            )
        if s["negatives"]:
            lo, hi = s["fp_ci"]
            lines.append(
                f"| faux positifs (négatifs) | {s['false_positives']}/{s['negatives']} = "
                f"{_pct(s['fp_rate'])} | [{_pct(lo)} ; {_pct(hi)}] |"
            )
        lines += [
            "",
            f"Échecs sur les positifs : {s['not_found']} non trouvé(s), "
            f"{s['wrong_song']} mauvais morceau(x).",
            "",
        ]
    lines += ["| temps (ms) | p50 | p95 |", "|---|---|---|"]
    for name, key in (
        ("latence (fin de capture → affichage)", "latency"),
        ("dont réseau + serveur", "network"),
        ("dont serveur seul", "server"),
    ):
        if s[key]:
            lines.append(f"| {name} | {s[key][0]:.0f} | {s[key][1]:.0f} |")
    if s["wrong_rows"] or s["missed_rows"]:
        lines += ["", "## Détail des erreurs", "", "| essai | attendu | réponse | score | ratio |"]
        lines.append("|---|---|---|---|---|")
        for r in sorted(s["wrong_rows"] + s["missed_rows"], key=lambda r: r.get("trial") or 0):
            expected = (
                titles.get(r["expected_song_id"], "(négatif)")
                if r["expected_song_id"]
                else "(négatif)"
            )
            got = titles.get(r["song_id"], f"#{r['song_id']}") if r["match"] else "non trouvé"
            lines.append(
                f"| {r.get('trial', '?')} | {expected} | {got} | {r.get('score') or '–'} "
                f"| {r.get('ratio') or '–'} |"
            )
    return "\n".join(lines) + "\n"


def save_report(run_id: str, text: str, folder: Path = REPORTS_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now(UTC):%Y-%m-%d}_{run_id[:8]}.md"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ excerpts and bench
def clip_path(t: Trial, folder: Path = CLIPS_DIR) -> Path:
    return folder / f"{t.youtube_id}_{t.start_s}.npy"


def cut(y: np.ndarray, start_s: int, sr: int) -> np.ndarray:
    a = int(start_s * sr)
    clip = y[a : a + EXCERPT_S * sr]
    if len(clip) < EXCERPT_S * sr * 0.9:
        raise ValueError(f"excerpt too short: {len(clip) / sr:.1f} s")
    return (np.clip(clip, -1, 1) * 32767).astype(np.int16)


def keep_clips(
    trials: list[Trial],
    downloader=None,
    folder: Path = CLIPS_DIR,
    pause_s: float = 0,
    block_wait_s: float = 1800,
    max_blocks: int = 8,
    sleep=time.sleep,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    """10 s excerpt of each test trial, for the synthetic bench (spec 7.1).

    The only audio kept by Earworm (DECISIONS.md): 120 excerpts of 10 s, local
    and git-ignored. The full download is deleted right after the cut."""
    from earworm.audio import SAMPLE_RATE, load
    from indexer.fetch import downloaded_audio

    folder.mkdir(parents=True, exist_ok=True)
    counts = {"kept": 0, "already": 0, "error": 0}
    todo = [t for t in trials if not clip_path(t, folder).exists()]
    counts["already"] = len(trials) - len(todo)
    blocked = 0
    for n, t in enumerate(todo, 1):
        if n > 1:
            sleep(pause_s)
        label = f"[{n}/{len(todo)}] n° {t.trial} {t.artist} - {t.title}"
        try:
            url = f"https://www.youtube.com/watch?v={t.youtube_id}"
            with downloaded_audio(url, downloader=downloader) as audio:
                y, sr = load(audio.path, SAMPLE_RATE)
            np.save(clip_path(t, folder), cut(y, t.start_s, sr))
        except Exception as e:
            counts["error"] += 1
            out(f"{label}: error {type(e).__name__}: {e}"[:300])
            if "not a bot" in str(e):
                blocked += 1
                if blocked >= max_blocks:
                    out("YouTube keeps blocking: stopping (run again later)")
                    break
                out(f"blocked by YouTube: waiting {block_wait_s / 60:.0f} min")
                sleep(block_wait_s)
            continue
        blocked = 0
        counts["kept"] += 1
        out(f"{label}: kept")
    return counts


def load_clip(t: Trial, folder: Path = CLIPS_DIR) -> np.ndarray | None:
    p = clip_path(t, folder)
    return np.load(p).astype(np.float32) / 32767 if p.exists() else None


def bench(
    trials: list[Trial],
    index,
    snrs: tuple = (None, 15, 10, 5, 0),
    noise: str = "babble",
    seed: int = 0,
    folder: Path = CLIPS_DIR,
) -> dict:
    """Each excerpt through a small speaker, a room and `noise` at each SNR (dB;
    None = no noise), then the same decision as the server."""
    from earworm.bench import degrade
    from earworm.fingerprint import fingerprint

    rng = np.random.default_rng(seed)
    per_snr = {snr: {"right": 0, "positives": 0, "fp": 0, "negatives": 0} for snr in snrs}
    used = 0
    for t in trials:
        clean = load_clip(t, folder)
        if clean is None:
            continue
        used += 1
        for snr in snrs:
            matches = index.query(fingerprint(degrade(clean, noise, snr, rng)))
            best = matches[0] if matches else None
            found = best is not None and best.is_confident
            c = per_snr[snr]
            if t.kind == "positive":
                c["positives"] += 1
                c["right"] += int(found and best.song_id == t.song_id)
            else:
                c["negatives"] += 1
                c["fp"] += int(found)
    return {"clips": used, "per_snr": per_snr, "noise": noise}


def bench_markdown(res: dict) -> str:
    lines = [
        f"{res['clips']} extraits, bruit « {res['noise']} » après petit haut-parleur et pièce",
        "",
        "| RSB | précision top-1 | IC 95 % | faux positifs |",
        "|---|---|---|---|",
    ]
    for snr, c in res["per_snr"].items():
        name = "sans bruit" if snr is None else f"{snr} dB"
        if c["positives"]:
            lo, hi = wilson(c["right"], c["positives"])
            prec = f"{c['right']}/{c['positives']} = {_pct(c['right'] / c['positives'])}"
            ci = f"[{_pct(lo)} ; {_pct(hi)}]"
        else:
            prec, ci = "–", "–"
        fp = f"{c['fp']}/{c['negatives']}" if c["negatives"] else "–"
        lines.append(f"| {name} | {prec} | {ci} | {fp} |")
    return "\n".join(lines) + "\n"
