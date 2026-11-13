# earworm (paquet partagé)

Code commun à l'indexeur et au serveur : chargement audio, empreintes (spectrogramme, pics, hachage), matching, validation des liens YouTube. Tout est calcul pur, sauf `db.py` (client REST Supabase) et `config.py` (lecture de `.env`).
