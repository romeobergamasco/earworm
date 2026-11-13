-- Earworm, schema v1 (docs/spec.md, section 4).
-- Run once on an empty Supabase project: SQL Editor -> New query -> paste -> Run.
--
-- Security model: the browser NEVER talks to Supabase. Only the server (Hugging
-- Face secret) and the local indexer (.env) do, with the secret key. Hence:
--   * Row Level Security on every table, with no policy at all;
--   * and all privileges revoked from the public roles (anon, authenticated),
--     so even a leaked publishable key reads and writes nothing.

create extension if not exists vector with schema extensions;

-- ---------------------------------------------------------------- songs
create table public.songs (
  id               bigint generated always as identity primary key,
  youtube_id       text not null unique check (youtube_id ~ '^[A-Za-z0-9_-]{11}$'),
  title            text not null,
  artist           text not null,
  year             int check (year between 1800 and 2100),
  mb_recording_id  uuid,
  mb_genres        text[] not null default '{}',   -- raw MusicBrainz genres
  mb_tags          text[] not null default '{}',
  genre_label      text,                           -- class of the v1 taxonomy (T20), if known
  cover_url        text,
  duration_s       real not null check (duration_s > 0),
  source           text not null check (source in ('playlist', 'chart', 'proposal', 'manual')),
  youtube_title    text,
  youtube_channel  text,
  created_at       timestamptz not null default now()
);

-- ---------------------------------------------------------------- fingerprints
-- One compressed blob per song (earworm.fingerprint.to_bytes): header b"EWFP",
-- blob version, count, then zlib(sorted hash deltas uint32 | anchor frames uint16).
create table public.fingerprints (
  song_id         bigint primary key references public.songs (id) on delete cascade,
  params_version  smallint not null,   -- earworm.fingerprint.PARAMS_VERSION
  n_hashes        int not null check (n_hashes >= 0),
  data            bytea not null
);

-- ---------------------------------------------------------------- embeddings
create table public.embeddings (
  song_id    bigint primary key references public.songs (id) on delete cascade,
  model      text not null,                    -- e.g. 'clap-htsat-unfused'
  embedding  extensions.vector(512) not null   -- mean of segment embeddings, L2-normalised
);
create index embeddings_cosine on public.embeddings
  using hnsw (embedding extensions.vector_cosine_ops);

-- ---------------------------------------------------------------- genre predictions
create table public.genre_predictions (
  song_id        bigint primary key references public.songs (id) on delete cascade,
  model_version  text not null,
  label          text not null,
  proba          real not null check (proba between 0 and 1),
  top3           jsonb not null
);

-- ---------------------------------------------------------------- proposals
create table public.proposals (
  id           bigint generated always as identity primary key,
  youtube_id   text not null check (youtube_id ~ '^[A-Za-z0-9_-]{11}$'),
  title_hint   text check (char_length(title_hint) <= 200),
  artist_hint  text check (char_length(artist_hint) <= 200),
  client_hash  text not null,          -- sha256(salt || ip): the IP itself is never stored
  status       text not null default 'pending'
               check (status in ('pending', 'accepted', 'rejected', 'indexed', 'failed')),
  reason       text,
  song_id      bigint references public.songs (id) on delete set null,
  created_at   timestamptz not null default now(),
  reviewed_at  timestamptz
);
create index proposals_rate_limit on public.proposals (client_hash, created_at);
create unique index proposals_one_pending on public.proposals (youtube_id) where status = 'pending';

-- ---------------------------------------------------------------- catalogue version
-- The server polls this to know when to reload its in-memory index.
create table public.catalog_meta (
  id          smallint primary key default 1 check (id = 1),
  version     int not null default 0,
  updated_at  timestamptz not null default now()
);
insert into public.catalog_meta (id, version) values (1, 0);

create function public.bump_catalog_version() returns trigger
  language plpgsql
  set search_path = ''
as $$
begin
  update public.catalog_meta set version = version + 1, updated_at = now() where id = 1;
  return null;
end;
$$;

-- any change to the fingerprints means the server must reload
create trigger fingerprints_changed
  after insert or update or delete on public.fingerprints
  for each statement execute function public.bump_catalog_version();

-- ---------------------------------------------------------------- security
alter table public.songs             enable row level security;
alter table public.fingerprints      enable row level security;
alter table public.embeddings        enable row level security;
alter table public.genre_predictions enable row level security;
alter table public.proposals         enable row level security;
alter table public.catalog_meta      enable row level security;

revoke all on all tables    in schema public from anon, authenticated;
revoke all on all sequences in schema public from anon, authenticated;
revoke all on all functions in schema public from anon, authenticated, public;
-- and for anything created later in this schema
alter default privileges in schema public revoke all on tables    from anon, authenticated;
alter default privileges in schema public revoke all on sequences from anon, authenticated;
alter default privileges in schema public revoke all on functions from anon, authenticated, public;
