-- Earworm, migration 004: genre predictions (T22).
-- Run once after 003_embeddings.sql: SQL Editor -> New query -> paste -> Run.
--
-- The table public.genre_predictions already exists (001_init.sql). The server
-- must reload when predictions change, as for fingerprints and embeddings.

create trigger genre_predictions_changed
  after insert or update or delete on public.genre_predictions
  for each statement execute function public.bump_catalog_version();
