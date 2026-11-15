-- Earworm, migration 005: evaluation sessions (T26, docs/spec.md section 7).
-- Run once after 004_genre_predictions.sql: SQL Editor -> New query -> paste -> Run.
--
-- A session (eval_runs) is opened by the local script `python -m indexer eval-session`,
-- which prints a PWA link with ?run_id=...; in that mode the PWA sends every
-- result, with the latency measured on the phone, to POST /api/eval/{run_id}
-- (eval_results). The local script then pairs each result with the song that
-- was played (trial, expected_*). No audio and nothing about the visitor is stored.

create table public.eval_runs (
  id          text primary key check (id ~ '^[A-Za-z0-9_-]{8,40}$'),
  kind        text not null check (kind in ('mic', 'latency')),
  note        text check (char_length(note) <= 200),   -- "wifi", "4g", "café, 65 dB(A)"...
  created_at  timestamptz not null default now(),
  closed_at   timestamptz
);

create table public.eval_results (
  id                bigint generated always as identity primary key,
  run_id            text not null references public.eval_runs (id) on delete cascade,
  created_at        timestamptz not null default now(),
  match             boolean not null,
  song_id           bigint,          -- answer of the server (null: not found)
  score             int,
  ratio             real,
  latency_ms        int not null check (latency_ms between 0 and 600000),
  resample_ms       int,
  network_ms        int,
  server_ms         jsonb,           -- timing_ms of /api/identify
  capture_rate      int,
  trial             int,             -- set by the local script
  expected_kind     text check (expected_kind in ('positive', 'negative')),
  expected_song_id  bigint,
  discarded         boolean not null default false   -- trial redone
);
create index eval_results_run on public.eval_results (run_id, id);

alter table public.eval_runs    enable row level security;
alter table public.eval_results enable row level security;
revoke all on public.eval_runs, public.eval_results from anon, authenticated;
