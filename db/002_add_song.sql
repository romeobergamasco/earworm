-- Earworm, migration 002: add a song and its fingerprints in ONE transaction.
-- Run once after 001_init.sql: SQL Editor -> New query -> paste -> Run.
--
-- PostgREST runs each HTTP request in its own transaction, so two separate
-- inserts (song, then fingerprints) could leave a song without fingerprints if
-- the second one failed. This function does both, or neither.

create function public.add_song(
  p_song            jsonb,     -- songs columns (no id, no created_at)
  p_fingerprint_hex text,      -- earworm.fingerprint.to_bytes(...).hex()
  p_params_version  smallint,
  p_n_hashes        int
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

  return to_jsonb(new_song);
end;
$$;

-- only the secret key (service_role) may call it
revoke all on function public.add_song(jsonb, text, smallint, int) from public, anon, authenticated;
grant execute on function public.add_song(jsonb, text, smallint, int) to service_role;
