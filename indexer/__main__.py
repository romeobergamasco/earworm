"""Local indexer command line.

python -m indexer add <YouTube link> [more links...]   download, analyse, delete, -> Supabase
python -m indexer lookup <YouTube link>   metadata only (no audio download)
python -m indexer db-check   check the Supabase project's access rules (.env)
python -m indexer export-local [--out data/supabase_index.npz]   index from Supabase
python -m indexer embed-missing   embeddings for the songs that have none (T19)
python -m indexer genre-labels [--dry-run]   genre class of every song (T20)
python -m indexer export-genre-data   segment embeddings + classes for Colab (T20)
python -m indexer predict-genres [--all]   genre prediction of every song (T22)
python -m indexer coverage   songs without an embedding or a genre (spec 7.3)
python -m indexer review   accept or reject the visitors' proposals (T25)
python -m indexer eval-draw | eval-session | eval-report | eval-clips | eval-bench
    measurement tooling (T26, docs/spec.md section 7; see indexer/evaluation.py)

`add` and `import-charts` also compute each new song's embedding (CLAP, a few
seconds per song) and, when models/genre_model.npz exists (T21), its genre;
--no-embed skips both (catch up later with embed-missing and predict-genres).
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

from earworm.youtube import parse_youtube_id
from indexer.fetch import YtDlpDownloader
from indexer.musicbrainz import AUTO_ACCEPT, Metadata, MusicBrainz, confirm
from indexer.titles import guess_artist_track


def lookup(url: str, interactive: bool = True, mb: MusicBrainz | None = None) -> Metadata | None:
    """YouTube link -> MusicBrainz metadata, asking a human when unsure."""
    mb = mb or MusicBrainz()
    info = YtDlpDownloader().info(parse_youtube_id(url))
    artist, track = guess_artist_track(info.track or info.title, info.artist or info.channel)
    print(f"YouTube: {info.title}  [{info.channel}]")
    cands = mb.search(artist, track)
    best = cands[0] if cands else None
    if best and best.confidence >= AUTO_ACCEPT:
        print(f"auto: {best.artist} - {best.title} ({best.year}), confidence {best.confidence:.2f}")
    elif interactive:
        best = confirm(artist, track, cands, mb)
    else:
        best = None
    if best is None:
        print("no MusicBrainz match kept: YouTube metadata only")
        return None
    return mb.metadata(best)


def cmd_lookup(args: argparse.Namespace) -> int:
    meta = lookup(args.url, interactive=not args.no_input)
    if meta:
        print(f"  {meta.artist} - {meta.title} ({meta.year or '?'})")
        print(f"  genres: {', '.join(meta.genres) or '-'}")
        print(f"  cover: {meta.cover_url or '-'}")
        print(f"  https://musicbrainz.org/recording/{meta.recording_id}")
    return 0 if meta else 1


def cmd_db_check(args: argparse.Namespace) -> int:
    from earworm.config import load_env, require
    from indexer.dbcheck import run

    load_env()
    try:
        cfg = require("SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_PUBLISHABLE_KEY")
    except RuntimeError as e:
        print(e)
        return 2
    results = run(cfg["SUPABASE_URL"], cfg["SUPABASE_SECRET_KEY"], cfg["SUPABASE_PUBLISHABLE_KEY"])
    for name, ok, detail in results:
        print(f"  {'OK  ' if ok else 'FAIL'} {name}  {detail}")
    failed = [r for r in results if not r[1]]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


def _secret_db():
    from earworm.config import load_env, require
    from earworm.db import Supabase

    load_env()
    cfg = require("SUPABASE_URL", "SUPABASE_SECRET_KEY")
    return Supabase(cfg["SUPABASE_URL"], cfg["SUPABASE_SECRET_KEY"])


def cmd_export_local(args: argparse.Namespace) -> int:
    from earworm.remote import build_index

    t0 = time.perf_counter()
    idx = build_index(_secret_db())
    idx.save(args.out)
    print(
        f"{idx.n_songs} songs, {idx.n_hashes} hashes from Supabase -> {args.out} "
        f"[{time.perf_counter() - t0:.1f} s]"
    )
    return 0


def cmd_charts_build(args: argparse.Namespace) -> int:
    from collections import Counter

    from indexer import charts
    from indexer.wiki import fetch_html

    entries = charts.build(fetch_html)
    charts.save(entries)
    by_decade = Counter(e.year // 10 * 10 for e in entries)
    by_chart = Counter(e.chart for e in entries)
    print(f"{len(entries)} chart songs -> {charts.CHARTS_CSV}  {dict(by_chart)}")
    print("  " + ", ".join(f"{d}s: {n}" for d, n in sorted(by_decade.items())))
    return 0


def cmd_charts_resolve(args: argparse.Namespace) -> int:
    from indexer import charts
    from indexer.resolve import YOUTUBE_CSV, load_resolved, resolve_all

    entries = charts.load()
    if args.limit:
        entries = entries[: args.limit]
    resolve_all(entries)
    done = load_resolved()
    found = sum(1 for r in done.values() if r["youtube_id"])
    print(f"{found}/{len(done)} chart songs have a YouTube video -> {YOUTUBE_CSV}")
    return 0


def _embedder(args: argparse.Namespace):
    if args.no_embed:
        return None
    from earworm.embeddings import ClapEmbedder

    print("loading the embedding model (CLAP)...")
    return ClapEmbedder()


def _genre_model(args: argparse.Namespace):
    from earworm.genre import DEFAULT_PATH, GenreModel

    if args.no_embed or not DEFAULT_PATH.exists():
        return None
    model = GenreModel.load(DEFAULT_PATH)
    print(f"genre model: {model.version}")
    return model


def cmd_predict_genres(args: argparse.Namespace) -> int:
    from earworm.genre import GenreModel
    from indexer.predict import predict_missing

    counts = predict_missing(_secret_db(), GenreModel.load(args.model), redo_all=args.all)
    return 0 if not counts["no_segments"] else 1


def cmd_coverage(args: argparse.Namespace) -> int:
    from indexer.predict import coverage

    c = coverage(_secret_db())
    print(f"{c['songs']} songs | without embedding: {len(c['without_embedding'])} "
          f"| without genre: {len(c['without_genre'])}")  # fmt: skip
    for k in ("without_embedding", "without_genre"):
        if c[k]:
            print(f"  {k}: {c[k][:30]}{' ...' if len(c[k]) > 30 else ''}")
    return 0 if not (c["without_embedding"] or c["without_genre"]) else 1


def cmd_embed_missing(args: argparse.Namespace) -> int:
    from earworm.embeddings import ClapEmbedder
    from indexer.embed import embed_missing

    counts = embed_missing(
        _secret_db(), ClapEmbedder(), cache=args.cache, pause_s=args.pause,
        block_wait_s=args.block_wait, max_blocks=args.max_blocks,
    )  # fmt: skip
    print("done: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return 0 if not counts["error"] else 1


def cmd_genre_labels(args: argparse.Namespace) -> int:
    from indexer.genres import relabel

    st = relabel(_secret_db(), dry_run=args.dry_run)
    return 0 if not st["unknown_genres"] else 1


def cmd_export_genre_data(args: argparse.Namespace) -> int:
    from earworm.remote import fetch_songs
    from indexer.dataset import export

    data = export(fetch_songs(_secret_db()), args.model, args.out)
    songs = len(set(data["song_id"].tolist()))
    labelled = len(set(data["song_id"][data["label"] != ""].tolist()))
    print(f"{len(data['X'])} segments of {songs} songs ({labelled} with a class) -> {args.out}")
    return 0


def cmd_import_charts(args: argparse.Namespace) -> int:
    from indexer import charts
    from indexer.catalog import LocalCatalog, SupabaseCatalog
    from indexer.importer import run_import
    from indexer.pipeline import add_song
    from indexer.resolve import load_resolved

    catalog = LocalCatalog(args.local) if args.local else SupabaseCatalog(_secret_db())
    print(f"catalogue: {len(catalog.songs)} songs, target {args.target}")
    add = partial(add_song, embedder=_embedder(args), genre_model=_genre_model(args))
    counts = run_import(
        catalog, charts.load(), load_resolved(), args.target, add=add, mb=MusicBrainz()
    )
    print(
        f"done: {len(catalog.songs)} songs  |  " + ", ".join(f"{k}: {v}" for k, v in counts.items())
    )
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from indexer.catalog import SupabaseCatalog
    from indexer.pipeline import add_song
    from indexer.review import open_proposals, preview, review

    db = _secret_db()
    if not open_proposals(db):
        print("no proposal to review")
        return 0
    mb, downloader = MusicBrainz(), YtDlpDownloader()
    catalog = SupabaseCatalog(db)
    add = partial(
        add_song, catalog=catalog, mb=mb, interactive=True, source="proposal",
        embedder=_embedder(args), genre_model=_genre_model(args),
    )  # fmt: skip
    counts = review(db, add, lambda p: preview(p, downloader, mb))
    print(f"{len(catalog.songs)} songs in Supabase")
    return 0 if not counts["failed"] else 1


def _titles(db) -> dict[int, str]:
    from earworm.remote import fetch_songs, song_title

    return {s["id"]: song_title(s) for s in fetch_songs(db)}


def cmd_eval_draw(args: argparse.Namespace) -> int:
    from earworm.fingerprint import PARAMS_VERSION
    from earworm.remote import fetch_songs
    from indexer import testset
    from indexer.resolve import load_resolved

    old = testset.load()
    if old and not args.force:
        print(f"{testset.TESTSET_CSV} exists ({len(old)} trials): --force to draw new positives")
        return 1
    songs = fetch_songs(_secret_db())
    negatives = [t for t in old if t.kind == "negative"]
    trials = testset.draw(songs, load_resolved(), seed=args.seed, keep_negatives=negatives)
    testset.save(trials, testset.meta_for(songs, args.seed, PARAMS_VERSION))
    pos = sum(t.kind == "positive" for t in trials)
    print(f"{pos} positives + {len(trials) - pos} negatives (seed {args.seed}, "
          f"catalogue {len(songs)}) -> {testset.TESTSET_CSV}")  # fmt: skip
    return 0


def cmd_eval_session(args: argparse.Namespace) -> int:
    import webbrowser

    from indexer import evaluation as ev
    from indexer import testset

    db = _secret_db()
    if args.resume:
        run_id = args.resume
    else:
        run_id = ev.open_run(db, "latency" if args.latency else "mic", args.note)
    print(f"session {run_id}")
    print(f"Sur le téléphone, ouvrez : {ev.pwa_link(run_id, args.app or ev.PWA_URL)}")
    if args.latency:
        n = ev.latency_session(db, run_id, args.count)
    else:
        trials = testset.load()
        if not trials:
            print("no test set: run eval-draw first")
            return 1
        trials = trials[: args.limit] if args.limit else trials
        open_url = webbrowser.open if args.open else None
        counts = ev.session(db, run_id, trials, _titles(db), open_url=open_url)
        n = counts["right"] + counts["wrong"]
        print()
        print(f"{counts['right']} juste(s), {counts['wrong']} faux, {counts['redone']} refait(s)")
    print(f"{n} résultat(s). Rapport : python -m indexer eval-report {run_id}")
    return 0


def cmd_eval_report(args: argparse.Namespace) -> int:
    from earworm.fingerprint import PARAMS_VERSION
    from indexer import evaluation as ev

    db = _secret_db()
    runs = db.select("eval_runs", "*", id=f"eq.{args.run_id}")
    if not runs:
        print(f"unknown session {args.run_id}")
        return 1
    rows = ev.results(db, args.run_id)
    titles = _titles(db)
    meta = {"commit": ev._git_commit(), "params_version": PARAMS_VERSION,
            "catalogue_songs": len(titles)}  # fmt: skip
    text = ev.report_markdown(runs[0], rows, titles, meta)
    print(text)
    if not args.no_save:
        print(f"-> {ev.save_report(args.run_id, text)}")
    if args.close:
        ev.close_run(db, args.run_id)
        print("session closed")
    return 0


def cmd_eval_clips(args: argparse.Namespace) -> int:
    from indexer import evaluation as ev
    from indexer import testset

    trials = testset.load()
    counts = ev.keep_clips(
        trials[: args.limit] if args.limit else trials,
        pause_s=args.pause, block_wait_s=args.block_wait, max_blocks=args.max_blocks,
    )  # fmt: skip
    print("done: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return 0 if not counts["error"] else 1


def cmd_eval_bench(args: argparse.Namespace) -> int:
    from earworm.index import Index
    from earworm.remote import build_index
    from indexer import evaluation as ev
    from indexer import testset

    if args.index:
        index = Index.load(args.index)
    else:
        print("building the index from Supabase...")
        index = build_index(_secret_db())
    res = ev.bench(testset.load(), index, noise=args.noise, seed=args.seed)
    print(ev.bench_markdown(res))
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    from indexer.catalog import LocalCatalog, SupabaseCatalog
    from indexer.pipeline import add_song

    if args.local:
        catalog, where = LocalCatalog(args.local), str(args.local)
    else:
        catalog, where = SupabaseCatalog(_secret_db()), "Supabase"
    mb = MusicBrainz()
    embedder, genre_model = _embedder(args), _genre_model(args)
    counts: dict[str, int] = {}
    for i, url in enumerate(args.urls, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(args.urls)}] {url}")
        try:
            res = add_song(
                url, catalog, mb=mb, interactive=not args.no_input, source=args.source,
                embedder=embedder, genre_model=genre_model,
            )  # fmt: skip
        except Exception as e:  # one bad link must not stop a batch
            counts["error"] = counts.get("error", 0) + 1
            print(f"  error: {type(e).__name__}: {e}")
            continue
        counts[res.status] = counts.get(res.status, 0) + 1
        s = res.song
        print(f"  {res.status}: #{s['id']} {s['artist']} - {s['title']} ({s['year'] or '?'})")
        print(f"  {res.detail}  [{time.perf_counter() - t0:.0f} s]")
    summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
    print()
    print(f"{len(catalog.songs)} songs in {where}  |  {summary}")
    return 0 if not counts.get("error") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m indexer")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add", help="add YouTube songs to the catalogue (Supabase)")
    p.add_argument("urls", nargs="+")
    p.add_argument("--local", type=Path, default=None, help="local catalogue folder instead")
    p.add_argument(
        "--source", default="manual", choices=["playlist", "chart", "proposal", "manual"]
    )
    p.add_argument(
        "--no-input", action="store_true", help="never ask; YouTube metadata when unsure"
    )
    p.add_argument("--no-embed", action="store_true", help="skip the embedding (see embed-missing)")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("lookup", help="find a YouTube video's song in MusicBrainz")
    p.add_argument("url")
    p.add_argument("--no-input", action="store_true", help="never ask; skip when unsure")
    p.set_defaults(func=cmd_lookup)

    p = sub.add_parser("db-check", help="check the Supabase project's access rules")
    p.set_defaults(func=cmd_db_check)

    p = sub.add_parser("charts-build", help="yearly charts from Wikipedia -> sources/charts.csv")
    p.set_defaults(func=cmd_charts_build)

    p = sub.add_parser("charts-resolve", help="find each chart song's YouTube video")
    p.add_argument("--limit", type=int, default=0, help="only the first N chart songs")
    p.set_defaults(func=cmd_charts_resolve)

    p = sub.add_parser("import-charts", help="import chart songs until the catalogue has N")
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--local", type=Path, default=None, help="local catalogue folder instead")
    p.add_argument("--no-embed", action="store_true", help="skip the embedding (see embed-missing)")
    p.set_defaults(func=cmd_import_charts)

    p = sub.add_parser("embed-missing", help="embeddings for the songs that have none")
    p.add_argument("--cache", type=Path, default=Path("data") / "embeddings" / "t18",
                   help="reuse the segment vectors computed for T18")  # fmt: skip
    p.add_argument("--pause", type=float, default=90, help="seconds between two downloads")
    p.add_argument("--block-wait", type=float, default=1800, help="wait when YouTube blocks")
    p.add_argument("--max-blocks", type=int, default=8, help="stop after N blocks in a row")
    p.set_defaults(func=cmd_embed_missing)

    p = sub.add_parser("genre-labels", help="genre class of every song (indexer/genre_map.yaml)")
    p.add_argument("--dry-run", action="store_true", help="show the statistics, change nothing")
    p.set_defaults(func=cmd_genre_labels)

    p = sub.add_parser("export-genre-data", help="segment embeddings + classes for Colab")
    p.add_argument("--model", default="clap-htsat-unfused")
    p.add_argument("--out", type=Path, default=Path("data") / "genre_dataset.npz")
    p.set_defaults(func=cmd_export_genre_data)

    p = sub.add_parser("predict-genres", help="genre prediction of every song (T22)")
    p.add_argument("--model", type=Path, default=Path("models") / "genre_model.npz")
    p.add_argument("--all", action="store_true", help="redo every song, not only missing ones")
    p.set_defaults(func=cmd_predict_genres)

    p = sub.add_parser("review", help="accept or reject the visitors' proposals")
    p.add_argument("--no-embed", action="store_true", help="skip the embedding (see embed-missing)")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("eval-draw", help="draw the measurement test set (docs/eval/)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true", help="redraw the positives, keep negatives")
    p.set_defaults(func=cmd_eval_draw)

    p = sub.add_parser("eval-session", help="measurement session with the phone (spec 7)")
    p.add_argument("--limit", type=int, default=0, help="only the first N trials")
    p.add_argument("--note", default=None, help='e.g. "café, 65 dB(A)" or "4g"')
    p.add_argument("--open", action="store_true", help="open each video in the browser")
    p.add_argument("--latency", action="store_true", help="latency only (no expected song)")
    p.add_argument("--count", type=int, default=50, help="with --latency: how many requests")
    p.add_argument("--resume", default=None, metavar="RUN_ID", help="continue a session")
    p.add_argument("--app", default=None, help="PWA address (default: the Space)")
    p.set_defaults(func=cmd_eval_session)

    p = sub.add_parser("eval-report", help="report of a measurement session")
    p.add_argument("run_id")
    p.add_argument("--no-save", action="store_true", help="print only")
    p.add_argument("--close", action="store_true", help="close the session afterwards")
    p.set_defaults(func=cmd_eval_report)

    p = sub.add_parser("eval-clips", help="keep the test set's 10 s excerpts (local only)")
    p.add_argument("--limit", type=int, default=0, help="only the first N trials")
    p.add_argument("--pause", type=float, default=90, help="seconds between two downloads")
    p.add_argument("--block-wait", type=float, default=1800, help="wait when YouTube blocks")
    p.add_argument("--max-blocks", type=int, default=8, help="stop after N blocks in a row")
    p.set_defaults(func=cmd_eval_clips)

    p = sub.add_parser("eval-bench", help="synthetic bench on the excerpts (non-regression)")
    p.add_argument("--index", type=Path, default=None, help="a saved index (export-local)")
    p.add_argument("--noise", default="babble", choices=["babble", "pink"])
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_eval_bench)

    p = sub.add_parser("coverage", help="songs without an embedding or a genre prediction")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("export-local", help="build a local .npz index from Supabase")
    p.add_argument("--out", type=Path, default=Path("data") / "supabase_index.npz")
    p.set_defaults(func=cmd_export_local)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
