# db

Schéma SQL Supabase (migrations numérotées).

- `001_init.sql` : tables, RLS sans politique, droits retirés aux rôles publics, version du catalogue mise à jour par déclencheur.
- `002_add_song.sql` : fonction `add_song`, qui écrit un morceau et son empreinte dans une seule transaction (clé secrète seulement).

Application : Supabase → SQL Editor → coller le fichier → Run. Vérification : `python -m indexer db-check` (lit `.env`).
