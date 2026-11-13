-- Earworm, migration 003: song embeddings (similar songs, T19).
-- Run once after 002_add_song.sql: SQL Editor -> New query -> paste -> Run.
--
-- The table public.embeddings already exists (001_init.sql). This migration:
--   1. tells the server to reload when embeddings change (catalogue version);
--   2. lets add_song store the embedding in the same transaction as the song
--      (the function gains two optional arguments: model and embedding).

create trigger embeddings_changed
  after insert or update or delete on public.embeddings
  for each statement execute function public.bump_catalog_version();

drop function public.add_song(jsonb, text, smallint, int);

create function public.add_song(
  p_song            jsonb,               -- songs columns (no id, no created_at)
  p_fingerprint_hex text,                -- earworm.fingerprint.to_bytes(...).hex()
  p_params_version  smallint,
  p_n_hashes        int,
  p_model           text default null,   -- e.g. 'clap-htsat-unfused'
  p_embedding       text default null    -- pgvector text: '[0.1,0.2,...]'
) returns jsonb
  language plpgsql
  set search_path = ''
as $$
declare
  new_song public.songs;
begin
  insert into public.songs (
    youtube_id, title, artist, year, mb_recording_id, mb_genres, mb_tags,
    genre_label, cover_url, duration_s, source, youtube_title, youtube_channel
  )
  select
    r.youtube_id, r.title, r.artist, r.year, r.mb_recording_id,
    coalesce(r.mb_genres, '{}'), coalesce(r.mb_tags, '{}'),
    r.genre_label, r.cover_url, r.duration_s, r.source, r.youtube_title, r.youtube_channel
  from jsonb_populate_record(null::public.songs, p_song) as r
  returning * into new_song;

  insert into public.fingerprints (song_id, params_version, n_hashes, data)
  values (new_song.id, p_params_version, p_n_hashes, decode(p_fingerprint_hex, 'hex'));

  if p_embedding is not null then
    insert into public.embeddings (song_id, model, embedding)
    values (new_song.id, p_model, p_embedding::extensions.vector);
  end if;

  return to_jsonb(new_song);
end;
$$;

-- only the secret key (service_role) may call it
revoke all on function public.add_song(jsonb, text, smallint, int, text, text)
  from public, anon, authenticated;
grant execute on function public.add_song(jsonb, text, smallint, int, text, text)
  to service_role;
