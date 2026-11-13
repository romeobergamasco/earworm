# db

Schéma SQL Supabase (migrations numérotées).

- `001_init.sql` : tables, RLS sans politique, droits retirés aux rôles publics, version du catalogue mise à jour par déclencheur.

Application : Supabase → SQL Editor → coller le fichier → Run. Vérification : `python -m indexer db-check` (lit `.env`).
