# db

Schéma SQL Supabase (migrations numérotées).

- `001_init.sql` : tables, RLS sans politique, droits retirés aux rôles publics, version du catalogue mise à jour par déclencheur.
- `002_add_song.sql` : fonction `add_song`, qui écrit un morceau et son empreinte dans une seule transaction (clé secrète seulement).
- `003_embeddings.sql` : `add_song` avec embedding, rechargement du serveur quand les embeddings changent.
- `004_genre_predictions.sql` : rechargement du serveur quand les prédictions de genre changent.
- `005_eval.sql` : sessions de mesure (T26) : `eval_runs` (ouvertes par le script local) et `eval_results` (résultats envoyés par la PWA en mode éval).

Application : Supabase → SQL Editor → coller le fichier → Run. Vérification : `python -m indexer db-check` (lit `.env`).
