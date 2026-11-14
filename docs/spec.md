# Earworm — spécification v1

Statut : **validé le 2026-11-13** · Source de vérité du cadrage : [`DECISIONS.md`](../DECISIONS.md).
Ce document précise *comment* on réalise ce qui y est décidé. En cas de conflit, `DECISIONS.md` l'emporte.

---

## 1. Périmètre

**Dans la v1**
1. Reconnaissance d'un morceau à partir d'environ 10 s capturées au micro, par empreintes codées à la main.
2. Pour chaque morceau reconnu : **sons similaires** (embeddings d'un modèle pré-entraîné) et **genre prédit** (classifieur entraîné sur Colab).
3. **Proposition d'un morceau** par lien YouTube, stockée dans une file validée à la main depuis la machine locale.
4. Catalogue d'environ 1 000 morceaux, appli en ligne (PWA sur un Space Hugging Face Static, API sur Render : hébergement provisoire, voir `DECISIONS.md`).

**Hors v1** : fredonnement (phase 2), comptes utilisateurs, historique côté serveur, recherche textuelle, lecture audio, catalogue de plus de 5 000 morceaux.

### Choix techniques posés par cette spec (à valider)
Ces points ne sont pas tranchés dans `DECISIONS.md` ; voici ce que je propose.

| Sujet | Proposition | Raison |
|---|---|---|
| Où sont calculés embeddings et genre | **À l'indexation, en local**, puis stockés. Le serveur ne fait tourner aucun modèle. | Le résultat de la reconnaissance est un morceau du catalogue : ses voisins et son genre sont connus d'avance. Le serveur reste léger et la latence ne dépend que du matching. |
| Stockage des empreintes | **Un blob compact par morceau** (`bytea`), et non une ligne par hash. Le serveur charge tout en mémoire au démarrage. | Mesuré au T04 : ~140 hashes/s, soit ~30 M de hashes pour 1 000 morceaux de 3 min 30. En une ligne par hash, Postgres dépasserait largement les 500 Mo (~60 o/ligne avec l'index). En blob compressé (différences entre hashes triés + zlib) : ~65 Mo (mesuré au T08). La recherche en mémoire tient aussi plus facilement les 3 s. |
| Modèle d'embedding | **CLAP** (`laion/clap-htsat-unfused`, 512 dimensions), **confirmé par T18** face à MERT-v1-95M. | Sur FMA-small, CLAP fait 0,48 de précision@5 de genre contre 0,31 pour MERT, pour un coût 10 fois moindre (0,9 s par morceau contre 10 s). Les deux se valent pour un classifieur entraîné. Il tourne sur CPU via `transformers`, et son espace texte-audio pourra resservir. Détails dans `docs/results.md`. |
| Similarité | Embeddings stockés avec **pgvector** dans Supabase ; le serveur les charge en mémoire avec le catalogue et précalcule les 5 voisins (cosinus) de chaque morceau (T19). | 1 000 vecteurs de 512 nombres font 2 Mo : un produit matriciel suffit, sans requête à la base à chaque réponse. |
| Format envoyé par le navigateur | PCM 16 bits mono **déjà rééchantillonné à 11 025 Hz** côté client (~220 Ko pour 10 s). | Pas de ffmpeg ni de souci de codec (Safari, webm/opus) côté serveur. |
| Framework serveur | **FastAPI** + uvicorn, dans un Space Docker. | Léger, sert l'API et les fichiers statiques de la PWA. |

---

## 2. Parcours utilisateur

### 2.1 Reconnaître un morceau
1. L'utilisateur ouvre l'appli (PWA installable, HTTPS, Space Hugging Face Static). Elle réveille l'API (Render) dès l'ouverture.
2. Il appuie sur **Écouter**. Le navigateur demande l'accès au micro (première fois seulement).
3. Capture de **10 s**, avec une jauge de progression et le niveau sonore. Les traitements du navigateur sont désactivés (`echoCancellation`, `noiseSuppression`, `autoGainControl` à `false`) : ils déforment le spectre.
4. Fin de capture → rééchantillonnage local → envoi. Un indicateur « recherche… » s'affiche.
5. **Résultat trouvé** : titre, artiste, année, pochette si MusicBrainz / Cover Art Archive en a une, lien vers la vidéo YouTube source, **genre prédit** (avec probabilité), **5 sons similaires** (cliquables : ils ouvrent leur fiche).
6. **Rien trouvé** : message clair, bouton « réessayer », et lien « il n'est pas dans le catalogue ? proposez-le ».
7. Erreurs gérées : micro refusé, micro absent, serveur en réveil (Space endormi), réseau coupé.

### 2.2 Proposer un morceau
1. Depuis l'écran « Proposer », l'utilisateur colle un lien YouTube et, s'il le veut, titre et artiste.
2. Le serveur vérifie le format du lien, extrait l'identifiant vidéo, refuse les doublons (déjà au catalogue ou déjà en file) et applique la limite de fréquence.
3. Réponse : « Merci, votre proposition sera examinée. » Aucun compte, aucune donnée personnelle demandée.

### 2.3 Valider les propositions (mainteneur, en local)
1. `python -m indexer review` liste les propositions en attente.
2. Pour chacune : affichage du titre YouTube et de la meilleure correspondance MusicBrainz → **accepter**, **refuser** (avec motif) ou **passer**.
3. Accepter lance le pipeline d'indexation (§5). La proposition passe à `indexed` ou à `failed` (avec l'erreur).

---

## 3. API HTTP

Base : `https://<space>.hf.space`. JSON en UTF-8. Aucune authentification côté utilisateur ; la clé Supabase reste sur le serveur (secret du Space).

### `POST /api/identify`
- **Corps** : `application/octet-stream`, PCM signé 16 bits little-endian, mono, 11 025 Hz. En-tête `X-Sample-Rate: 11025` (vérifié).
- **Contraintes** : durée entre 3 s et 15 s, sinon `400`. Corps limité à 400 Ko (`413`).
- **Réponse 200 (trouvé)**
  ```json
  {
    "match": true,
    "song": {"id": 42, "title": "…", "artist": "…", "year": 2019,
             "youtube_id": "…", "cover_url": "…|null"},
    "confidence": {"score": 87, "ratio": 6.2, "offset_s": 61.4},
    "genre": {"label": "rock", "name": "Rock", "proba": 0.71,
              "top3": [{"label": "rock", "name": "Rock", "proba": 0.71}, …]},
    "similar": [{"id": 7, "title": "…", "artist": "…", "year": 1984,
                 "youtube_id": "…", "cover_url": "…|null"}, …],
    "timing_ms": {"fingerprint": 180, "match": 40, "total": 260}
  }
  ```
- **Réponse 200 (non trouvé)** : `{"match": false, "timing_ms": {…}}`
- `score` = hauteur du pic de l'histogramme des décalages ; `ratio` = score / score du 2ᵉ morceau. Les seuils de décision sont réglés au ticket T07.

### `GET /api/songs/{id}`
Fiche d'un morceau, au même format que `song` + `genre` + `similar`. Sert quand on clique un son similaire.

### `POST /api/proposals`
- **Corps** : `{"url": "https://www.youtube.com/watch?v=…", "title": "…?", "artist": "…?"}`
- Liens acceptés : `youtube.com/watch?v=`, `youtu.be/`, `music.youtube.com/watch?v=`, `youtube.com/shorts/`. Identifiant : 11 caractères `[A-Za-z0-9_-]`.
- **Limite** : 5 propositions par heure et 20 par jour, par client. Le client est identifié par le **hash SHA-256 salé de son IP** ; l'IP en clair n'est jamais stockée.
- Réponses : `201` créé · `409` déjà au catalogue ou déjà en attente · `422` lien invalide · `429` limite atteinte.

### `GET /api/health`
`{"status": "ok", "songs": 1000, "hashes": 4123456, "index_loaded_at": "…", "catalog_version": 17}`

### Rechargement de l'index
Le serveur lit `catalog_meta.version` toutes les 10 minutes ; si la valeur a augmenté, il recharge l'index en tâche de fond et bascule d'un coup. Aucun endpoint d'écriture admin n'est exposé.

---

## 4. Schéma Supabase

Postgres avec l'extension `vector`. **RLS activée sur toutes les tables, sans aucune politique publique** : seul le serveur (clé `service_role`, secret HF) et l'indexeur local (même clé, dans `.env`) y accèdent. Le navigateur ne parle jamais à Supabase.

```sql
create extension if not exists vector;

create table songs (
  id            bigint generated always as identity primary key,
  youtube_id    text unique not null,
  title         text not null,
  artist        text not null,
  year          int,
  mb_recording_id uuid,            -- MusicBrainz
  mb_genres     text[] default '{}',   -- étiquettes brutes MusicBrainz
  genre_label   text,               -- classe de la taxonomie v1 (vérité terrain si connue)
  cover_url     text,
  duration_s    real not null,
  source        text not null check (source in ('playlist','chart','proposal','manual')),
  created_at    timestamptz default now()
);

-- Empreintes : un blob par morceau.
-- Format (earworm.fingerprint.to_bytes) : en-tête b"EWFP" + version + nombre, puis zlib(différences des hashes triés
-- en uint32 | trames d'ancrage en uint16), little-endian. ~100 Ko par morceau mesuré.
create table fingerprints (
  song_id       bigint primary key references songs(id) on delete cascade,
  n_hashes      int not null,
  data          bytea not null,
  params_version smallint not null  -- version des paramètres d'empreinte (réindexation si changement)
);

create table embeddings (
  song_id       bigint primary key references songs(id) on delete cascade,
  model         text not null,       -- ex. 'clap-htsat-unfused'
  embedding     vector(512) not null -- moyenne normalisée des segments
);
create index on embeddings using hnsw (embedding vector_cosine_ops);

create table genre_predictions (
  song_id       bigint primary key references songs(id) on delete cascade,
  model_version text not null,
  label         text not null,
  proba         real not null,
  top3          jsonb not null
);

create table proposals (
  id            bigint generated always as identity primary key,
  youtube_id    text not null,
  title_hint    text,
  artist_hint   text,
  client_hash   text not null,       -- sha256(sel || ip)
  status        text not null default 'pending'
                check (status in ('pending','accepted','rejected','indexed','failed')),
  reason        text,
  song_id       bigint references songs(id),
  created_at    timestamptz default now(),
  reviewed_at   timestamptz
);
create index on proposals (client_hash, created_at);
create unique index proposals_one_pending on proposals (youtube_id) where status = 'pending';

create table catalog_meta (
  id            smallint primary key default 1 check (id = 1),
  version       int not null default 0,   -- incrémenté par l'indexeur à chaque ajout
  updated_at    timestamptz default now()
);
```

**Budget de stockage estimé (1 000 morceaux)** : empreintes ~65 Mo (mesuré sur 10 morceaux réels, extrapolé), embeddings ~2 Mo, le reste est négligeable. 5 000 morceaux donneraient ~320 Mo : ça passe sous le quota, avec une marge réduite. Réduire la densité ou mieux compresser fera partie du travail d'optimisation prévu après la v1. Le quota de transfert sortant de Supabase gratuit suffit pour charger l'index à chaque redémarrage du Space ; un cache disque local sera ajouté si ça devient un problème.

---

## 5. Pipeline d'indexation locale

Commande : `python -m indexer add <url|fichier_de_liste>`, et `python -m indexer review` pour les propositions. Tourne **uniquement en local**.

```
URL YouTube
  │ 1. métadonnées yt-dlp (titre, chaîne, durée) ; refus si durée > 12 min ou < 60 s
  │ 2. déjà au catalogue ? (youtube_id) → stop
  │ 3. téléchargement audio seul dans un dossier temporaire
  │ 4. décodage → mono 11 025 Hz (empreintes) et 48 kHz (CLAP)
  │ 5. doublon acoustique ? → reconnaissance d'un extrait de 10 s contre l'index ; si trouvé → stop + log
  │ 6. empreintes → blob
  │ 7. embedding CLAP : segments de 10 s pris sur toute la durée, moyenne, normalisation L2
  │ 8. genre prédit (classifieur chargé depuis models/)
  │ 9. MusicBrainz : recherche titre + artiste, meilleure correspondance avec score ;
  │    sous le seuil → confirmation manuelle (mode interactif) ou mise de côté
  │ 10. écriture Supabase dans une transaction, catalog_meta.version += 1
  └ 11. SUPPRESSION de l'audio (bloc finally : même en cas d'erreur)
```

**Garde-fous** (voir `DECISIONS.md`) : l'audio n'existe que dans un `TemporaryDirectory`, supprimé dans un `finally` ; un test vérifie que le dossier est vide après un succès *et* après une erreur. Aucun fichier audio n'est versionné (`.gitignore`) ni servi par le serveur.

**MusicBrainz** : 1 requête par seconde maximum, avec un User-Agent d'application (`Earworm/0.1 ( <url du dépôt GitHub> )`), comme l'exige leur règlement. Pochettes : Cover Art Archive, via l'URL seulement (rien n'est stocké).

**Catalogue initial** : un fichier `indexer/sources.txt` (playlists de l'utilisateur, classements) ; l'import reprend là où il s'est arrêté grâce à l'unicité de `youtube_id`. Débit attendu sur le Ryzen 5 : quelques dizaines de secondes par morceau, surtout pour CLAP. Les 1 000 morceaux se font en plusieurs sessions.

### 5.1 Empreintes : paramètres
Réglés au T07 (banc synthétique) puis corrigés au T08 (vrais enregistrements au téléphone) : `PARAMS_VERSION = 3`. Mesures dans `docs/results.md`.

| Paramètre | Valeur |
|---|---|
| Fréquence d'échantillonnage | 11 025 Hz, mono |
| STFT | fenêtre de Hann 2 048, saut 512 (~46 ms), magnitude en dB |
| Pics | maximum local sur 21 bins × 11 trames (~113 Hz × 0,5 s), 10 dB au-dessus de la médiane de la trame, bande 100–5 000 Hz, plafond de **20 pics/s** (réglé au T07) |
| Paires | chaque pic ancre est relié aux **8** pics suivants de sa zone cible (Δt de 1 à 63 trames, Δf de ±200 bins) → ~148 hashes/s (réglé au T07) |
| Hash | `f1/4` (10 bits) · `f2/4` (10 bits) · `Δt/2` (6 bits) → uint32. Les fréquences sont codées par cases de 4 bins (21,5 Hz), et Δt par 2 trames : un pic enregistré au téléphone tombe souvent à 1 ou 2 bins de sa position dans le fichier (T08) |
| Matching | pour chaque hash de la requête, recherche par `searchsorted` dans le tableau global trié ; histogramme des `(song_id, offset_db − offset_req)` ; score = pic maximal |
| Décision | trouvé si `score ≥ 30` **et** `ratio ≥ 2` (T08 : ≤ 0,01 % de faux positifs simulés pour un catalogue de 10 à 5 000 morceaux, voir `docs/results.md`) |

Index en mémoire côté serveur (T17b) : un seul tableau `uint64` trié (`hash << 32 | morceau << 16 | trame`), soit 8 octets par hash. Pour ~33 M de hashes (1 000 morceaux), cela fait ~255 Mo, ce qui tient dans les 512 Mo de Render.

---

## 6. Genre : données et entraînement

- **Taxonomie v1** (T20, `indexer/genre_map.yaml`) : 10 classes, à savoir pop, rock (dont metal), hip-hop/rap, électro/dance, R&B/soul/funk, jazz/blues, country/folk, chanson/variété française, latino et reggae/Caraïbes. Le classique a été retiré : un seul morceau dans le catalogue.
- **Étiquettes** : les genres MusicBrainz, ou à défaut leurs tags, passent par une table de correspondance écrite à la main vers la taxonomie. Chaque genre est pondéré par son palier de votes, retrouvé à partir de l'ordre de MusicBrainz, et « pop » ne compte qu'en l'absence d'autre classe. Les morceaux sans étiquette sont exclus de l'entraînement, mais reçoivent quand même une prédiction.
- **Données** : plusieurs segments de 10 s par morceau → embeddings par segment, exportés depuis Supabase/local vers Colab (fichier `.npz` sans audio).
- **Séparation entraînement / test par morceau** (jamais par segment), stratifiée par classe.
- **Modèles** : régression logistique (référence), puis petit MLP en PyTorch. Métriques : précision, F1 macro, matrice de confusion.
- **Comparaison** : même protocole sur un sous-ensemble de **FMA-small** (8 genres équilibrés, téléchargeable gratuitement, embeddings calculés sur Colab), pour situer les résultats face à un jeu de référence.
- **Export** : poids légers (`.pt` ou `joblib`) placés dans `models/` en local, non versionnés s'ils sont lourds. Ils servent à l'indexeur seulement.

---

## 7. Protocole de mesure des critères de fin

Les critères viennent de `DECISIONS.md`. Cette section fixe comment on les mesure pour que les chiffres du README soient défendables.

### 7.1 Précision au micro en milieu bruité (critère : ≥ 90 %)
- **Jeu de test** : 100 morceaux tirés au hasard dans le catalogue (graine fixée, liste versionnée dans `docs/eval/`), avec un départ aléatoire entre 20 % et 80 % de la durée.
- **Négatifs** : 20 morceaux **absents** du catalogue, pour mesurer les faux positifs.
- **Montage** : la musique sort d'un appareil (téléphone ou enceinte), à 1 m environ du téléphone qui enregistre via la PWA déployée. Le bruit vient d'une pièce réellement bruyante (conversation, cuisine, café) **ou** d'un fond sonore de café diffusé par un 2ᵉ appareil. On note le niveau ambiant avec une appli sonomètre gratuite (cible : environ 60 à 70 dB(A)).
- **Déroulé** : un script lit la liste et affiche le morceau suivant ; l'opérateur lance l'extrait sur l'appareil source puis appuie sur **Écouter**. Le serveur journalise chaque requête avec un `run_id` (passé en paramètre de l'URL de la PWA en mode éval), et le script compare au morceau attendu.
- **Métriques** :
  - précision top-1 sur les 100 positifs (une réponse « non trouvé » compte comme un échec) ;
  - taux de faux positifs sur les 20 négatifs (cible ≤ 5 %) ;
  - intervalle de confiance à 95 % (Wilson) publié avec le chiffre. Avec 100 essais, 90 % observé donne environ [83 %, 94 %] : on l'écrit tel quel.
- **Banc synthétique (non-régression, automatisé)** : les mêmes 100 extraits, découpés à partir des fichiers avant suppression **ou** rejoués à partir d'enregistrements de test (les enregistrements micro bruts sont gardés en local, jamais publiés), mélangés à du bruit à 0, 5, 10 et 15 dB de RSB. On l'utilise pour régler les paramètres, pas comme critère de fin.

> Point d'attention : l'audio étant supprimé après indexation, le banc synthétique a besoin de ses extraits. Proposition : pendant l'indexation, conserver **uniquement pour les 100 morceaux du jeu de test** un extrait de 10 s en local (`data/eval/`, ignoré par git, jamais publié). Les garde-fous restent respectés pour tout le reste du catalogue. À valider.

### 7.2 Latence (critère : < 3 s)
- **Définition** : temps entre la **fin de la capture** et l'**affichage du résultat**, mesuré dans la PWA (`performance.now()`), avec le Space **éveillé**. Il inclut le rééchantillonnage, l'envoi, le traitement serveur et le rendu.
- **Mesure** : sur les 120 essais du §7.1, plus 50 requêtes lancées depuis le portable en Wi-Fi et 50 depuis un téléphone en 4G.
- **Critère** : **p95 < 3 s** ; on publie aussi p50, et le temps serveur (`timing_ms`) à part.
- **Réveil à froid** : mesuré et publié séparément (le Space gratuit s'endort après une période d'inactivité). Il ne compte pas dans le critère, mais la PWA affiche « réveil du serveur… » si `/api/health` met plus de 2 s à répondre.

### 7.3 Autres critères
- 1 000 morceaux : `GET /api/health` → `songs ≥ 1000`.
- Similaires et genre pour chaque résultat : test automatique qui vérifie que chaque `song` a une ligne dans `embeddings` et dans `genre_predictions`.
- Ajout par lien YouTube : une proposition faite depuis le téléphone sur l'appli en ligne, validée en local, puis reconnue au micro.
- Résultats consignés dans `docs/results.md` (date, commit, `params_version`, chiffres) et repris dans le README.

---

## 8. Structure du dépôt

```
earworm/     paquet partagé : audio, fingerprint, matching (aucun accès réseau)
indexer/     local : yt-dlp, MusicBrainz, embeddings, écriture Supabase, revue des propositions
server/      FastAPI : API + service des fichiers statiques de web/
web/         PWA HTML/CSS/JS simple
notebooks/   Colab : choix d'embedding, classifieur de genre
db/          migrations SQL Supabase
tests/       pytest
docs/        spec, plan, protocole et résultats d'évaluation
```

Déploiement (provisoire, voir `DECISIONS.md`) : l'**API** tourne sur **Render** (offre gratuite) à partir du `Dockerfile` du dépôt GitHub, secrets Supabase dans Render ; elle n'accepte les appels du navigateur que depuis l'adresse du Space (CORS, `ALLOWED_ORIGINS`). La **PWA** est publiée sur un **Space Hugging Face Static** par `deploy/push_static.py`, avec l'adresse de l'API dans `web/config.js`. Contraintes de Render gratuit : 512 Mo de RAM (index compacté, T17b), mise en veille après 15 min sans trafic (~1 min de réveil, mesuré à part).

---

## 9. Risques identifiés

| Risque | Parade |
|---|---|
| Précision < 90 % au micro (haut-parleur de téléphone, bruit) | Banc synthétique tôt (T07) ; réglage de la densité de pics et du fan-out ; essais au micro dès T08, avant tout le reste. |
| Correspondance YouTube → MusicBrainz approximative | Score de correspondance, confirmation manuelle sous le seuil, métadonnées yt-dlp en repli. |
| Peu d'étiquettes de genre MusicBrainz sur une partie du catalogue | Exclusion de l'entraînement, comparaison FMA, taxonomie à adapter. |
| Space endormi ou lent | Chargement de l'index en tâche de fond, message de réveil, mesure séparée. |
| Changement côté YouTube qui casse yt-dlp | Indexation locale uniquement ; `yt-dlp` mis à jour avant chaque session d'import. |
