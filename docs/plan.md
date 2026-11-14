# Earworm — plan v1 en tickets

Statut : **validé le 2026-11-13** · S'appuie sur [`spec.md`](spec.md).

Règles :
- Tickets **dans l'ordre**. Chacun tient en une ou deux séances de quelques heures.
- Un ticket est fini quand **tous** ses critères d'acceptation sont vérifiés, et qu'il est commité.
- Les jalons A et B donnent un système qui marche **de bout en bout en local**, sans réseau : rien de Supabase ni du déploiement avant.
- Tests : `pytest` pour tout ce qui est déterministe ; scripts de mesure pour le reste.

---

## Jalon A — Reconnaissance locale sur ~10 fichiers

### T01 · Environnement Python
- `pyproject.toml` (Python 3.12), paquet `earworm` installable en mode éditable, `pytest`, `ruff`.
- Dépendances de base : `numpy`, `scipy`, `librosa`, `soundfile`.
- **Acceptation** : `pip install -e .[dev]` puis `pytest` passe (un test trivial) ; `ruff check` passe ; la commande d'installation est documentée dans `README.md`.

### T02 · Chargement audio et spectrogramme
- `earworm.audio.load(path) -> (np.ndarray float32 mono, 11025)` ; `earworm.fingerprint.spectrogram(y)` (STFT Hann 2048 / saut 512, en dB).
- **Acceptation** : test sur une sinusoïde générée à 1 000 Hz : le bin maximal correspond à 1 000 Hz ± 1 bin ; un MP3 et un WAV du même son donnent des spectrogrammes de même forme.

### T03 · Détection des pics (constellation)
- `find_peaks(S) -> array[(frame, bin)]`, maximum local et seuil adaptatif.
- Script `python -m earworm.tools.plot_peaks <fichier>` qui sauvegarde un PNG spectrogramme + pics.
- **Acceptation** : densité entre 20 et 40 pics/s sur 3 morceaux réels ; test : un signal fait de 3 sinusoïdes donne des pics sur ces 3 fréquences ; le PNG est lisible et sert dans le README plus tard.

### T04 · Hachage par paires
- `hashes(peaks) -> array[(hash uint32, offset uint32)]` selon la spec §5.1 ; format blob `to_bytes` / `from_bytes`.
- **Acceptation** : déterminisme (même fichier → mêmes hashes) ; test sur des pics construits à la main → hash attendu bit à bit ; aller-retour blob sans perte ; hashes/s et taille du blob mesurés sur morceaux réels, avec un stockage extrapolé à 1 000 morceaux ≤ 150 Mo (sinon réglage du fan-out ou compression). *(Le critère initial « 20 à 40 hashes/s » était une erreur : on obtient pics/s × fan-out.)*

### T05 · Index en mémoire et matching
- `earworm.index.Index` : `add(song_id, hashes)`, `freeze()` (tri par hash), `query(hashes) -> [(song_id, score, ratio, offset)]`.
- **Acceptation** : sur 10 morceaux, un extrait **propre** de 10 s découpé à un endroit aléatoire est reconnu **10/10** avec le bon décalage (± 1 trame) ; une requête de 10 s prend < 200 ms sur le portable.

### T06 · CLI locale de bout en bout
- `python -m earworm index <dossier> --out data/local_index.npz` et `python -m earworm identify <fichier>`.
- **Acceptation** : avec une dizaine de fichiers personnels de l'utilisateur, `identify` sur un extrait recadré avec Audacity (ou ffmpeg) renvoie le bon titre ; un morceau absent renvoie « non trouvé ».

**→ Jalon A atteint : empreinte et reconnaissance de bout en bout en local.**

---

## Jalon B — Robustesse mesurée

### T07 · Banc d'essai synthétique et seuils
- `python -m earworm.bench` : extraits de 10 s + bruit (bruit rose et ambiance de café libre de droits) à 0 / 5 / 10 / 15 dB de RSB, et extraits de morceaux absents.
- Réglage de la densité de pics, du fan-out, de `S_min` et `R_min` ; paramètres figés dans `params_version = 1`.
- **Acceptation** : tableau Markdown généré dans `docs/results.md` (précision par RSB, faux positifs) ; ≥ 90 % à 5 dB et ≤ 2 % de faux positifs sur les fichiers locaux ; les seuils retenus sont justifiés dans le tableau.

### T08 · Essai au micro, en local
- `python -m earworm listen` : enregistre 10 s via `sounddevice` puis identifie.
- **Acceptation** : 10 essais avec la musique jouée depuis le téléphone à ~1 m dans une pièce calme, puis 10 dans une pièce bruyante ; résultats notés dans `docs/results.md`. Si on est sous 8/10 dans le bruit, on revient à T07 avant d'aller plus loin.
- *Réalisé (2026-11-13)* : le micro du portable n'étant pas exploitable, les essais ont été faits avec le dictaphone d'un téléphone, puis `identify` sur les fichiers. Le premier passage (5/10 en bruyant) a conduit à corriger le hachage et les seuils (`PARAMS_VERSION = 3`) : 10/10 en calme, 9/10 en bruyant. Détails dans `docs/results.md`.

**→ Jalon B atteint : on sait que la méthode tient au micro avant d'investir dans l'infra.**

---

## Jalon C — Ingestion depuis YouTube (toujours local)

### T09 · Téléchargement, analyse, suppression
- `indexer.fetch` : yt-dlp en audio seul, dans un `TemporaryDirectory`, décodage, suppression dans un `finally`.
- **Acceptation** : test (avec un faux téléchargeur) : le dossier temporaire est vide après un succès **et** après une exception ; refus des vidéos de plus de 12 min ou de moins de 60 s ; essai réel sur 3 liens.

### T10 · Métadonnées MusicBrainz
- `indexer.musicbrainz` : recherche à partir du titre YouTube nettoyé (« (Official Video) », « ft. », etc.), score de correspondance, 1 requête/s, User-Agent conforme, URL de pochette Cover Art Archive.
- **Acceptation** : sur 20 titres réels variés, le bon enregistrement est trouvé automatiquement pour ≥ 15 ; les autres passent en confirmation manuelle interactive ; tests du nettoyage de titres.
- *Réalisé (2026-11-13)* : 20/20 automatiquement, 0 erreur (`docs/results.md`). Commande `python -m indexer lookup <lien>`.

### T11 · Pipeline `indexer add` local + doublons
- Enchaîne T09 → empreinte → doublon acoustique → T10, et écrit dans l'index local `.npz` + un `catalog.json`.
- **Acceptation** : `python -m indexer add <url>` ajoute un morceau reconnaissable ensuite par `identify` ; le même morceau sous un autre lien YouTube est détecté comme doublon ; aucun fichier audio ne reste sur le disque.
- *Réalisé (2026-11-13)* : catalogue local dans `data/catalog/` ; essai réel et choix sur les remix dans `docs/results.md`.

---

## Jalon D — Supabase

### T12 · Projet et schéma
- Projet Supabase gratuit, `db/001_init.sql` (spec §4), RLS activée sans politique publique, `.env.example`.
- **Acceptation** : la migration s'applique sur une base vide ; une requête avec la clé `anon` ne lit **rien** ; la clé `service_role` lit et écrit.
- *Réalisé (2026-11-13)* : clés publishable / secret (nouveau nom des clés anon / service_role), `python -m indexer db-check` 13/13.

### T13 · L'indexeur écrit dans Supabase
- Remplace le stockage local de T11 par une écriture transactionnelle (`songs` + `fingerprints` + `catalog_meta.version`). Ajoute `python -m indexer export-local` pour retrouver un `.npz` (utile aux tests).
- **Acceptation** : 10 morceaux poussés ; relancer la même commande ne crée aucun doublon ; un index reconstruit depuis Supabase donne les mêmes résultats que l'index local sur le banc T07.
- *Réalisé (2026-11-13)* : écriture en une transaction via `db/002_add_song.sql` ; résultats identiques (`docs/results.md`).

### T14 · Import du catalogue (en continu)
- `indexer/sources.txt` (playlists + classements), import par lots avec reprise et journal d'erreurs.
- **Acceptation** : 200 morceaux indexés à la fin du ticket. La montée à 1 000 se poursuit en tâche de fond pendant les tickets suivants ; elle est vérifiée en T27.

---

## Jalon E — Serveur et PWA en ligne

### T15 · Serveur FastAPI
- Chargement de l'index depuis Supabase au démarrage (en tâche de fond), `POST /api/identify`, `GET /api/songs/{id}`, `GET /api/health`, rechargement quand `catalog_meta.version` change, journal de chaque requête avec `timing_ms`.
- **Acceptation** : tests d'API avec un PCM généré à partir d'un extrait connu → bon morceau ; `400` / `413` sur entrées invalides ; temps serveur < 500 ms pour 200 morceaux en local.

### T16 · PWA minimale
- `web/` : bouton Écouter, capture `getUserMedia` (traitements désactivés), rééchantillonnage à 11 025 Hz via `OfflineAudioContext`, envoi, affichage trouvé / non trouvé, gestion des erreurs micro et réseau. Manifest + service worker (coquille hors ligne).
- **Acceptation** : sur `localhost`, reconnaissance au micro du portable de bout en bout ; Lighthouse indique « installable » ; la latence mesurée côté client s'affiche en mode debug.

### T17 · Déploiement (provisoire : Space Static + Render)
- *Changement du 2026-11-13* : les Spaces Docker de Hugging Face sont devenus payants (`DECISIONS.md`, « Points ouverts »).
- API : `Dockerfile` déployé sur **Render** (offre gratuite) depuis le dépôt GitHub, secrets Supabase dans Render, CORS limité à l'adresse du Space.
- PWA : **Space Hugging Face Static** publié par `deploy/push_static.py`, adresse de l'API dans `web/config.js`.
- **Acceptation** : l'URL publique du Space reconnaît un morceau **depuis un téléphone** ; `/api/health` de Render répond ; aucun secret dans le dépôt (vérifié avec `git grep`) ; temps de réveil mesuré.
- **En ligne** : appli https://romeoberga-earworm.static.hf.space, API Render ; 19/20 enregistrements reconnus à distance (~0,5 s). Essai au téléphone réussi ; réveil en 26 s (`docs/results.md`).

### T17b · Index compact en mémoire (512 Mo sur Render)
- Un seul tableau `uint64` trié (`hash << 32 | morceau << 16 | trame`) au lieu de trois tableaux `uint32`, rempli morceau par morceau sans copie intermédiaire (8 octets par hash au lieu de 12, sans pic mémoire au tri).
- **Acceptation** : résultats identiques à l'index actuel (banc + enregistrements au téléphone) ; mémoire du processus serveur < 400 Mo avec un catalogue simulé de 1 000 morceaux.
- **Réalisé** : pic de 601 Mo → 155 Mo à 302 morceaux ; 369 Mo au premier chargement de 1 000 morceaux simulés ; mêmes résultats sur les 50 requêtes de référence. Le rechargement à 1 000 morceaux ne tient pas à côté de l'ancien index : le serveur libère alors l'ancien d'abord (`MEMORY_BUDGET_MB`). Détails dans `docs/results.md`.

**→ Jalon E atteint : premiers vrais utilisateurs possibles (reconnaissance seule).**

---

## Jalon F — Embeddings et sons similaires

### T18 · Choix du modèle d'embedding
- Notebook `notebooks/01_embeddings.ipynb` : CLAP vs MERT sur ~200 morceaux (embeddings calculés en local ou sur Colab, sans audio publié). Critères : précision k-NN du genre sur les étiquettes MusicBrainz disponibles, temps CPU par morceau, écoute qualitative de 10 listes de voisins.
- **Acceptation** : tableau comparatif dans le notebook et décision notée dans `docs/spec.md` (le choix par défaut, CLAP, peut être confirmé ou remplacé).
- **Réalisé : CLAP** (règle fixée d'avance). MERT est significativement moins bon en précision@5 de genre, sur le catalogue (269 morceaux : 0,27 contre 0,35) comme sur FMA-small (0,31 contre 0,48), pour 10 fois plus de calcul (`docs/results.md`).

### T19 · Embeddings dans le pipeline et similaires
- Étape d'embedding (segments de 10 s, moyenne, L2) dans `indexer add` + commande de rattrapage pour les morceaux déjà indexés ; table `embeddings` (pgvector) ; `similar` (top 5, hors le morceau lui-même) dans les réponses de l'API.
- **Acceptation** : 100 % des morceaux ont un embedding ; `similar` renvoie 5 résultats en < 100 ms ; test : un morceau a pour plus proche voisin lui-même avant exclusion.
- **Réalisé (code)** : `earworm/embeddings.py` (CLAP, segments), `earworm/similar.py` (voisins cosinus, catalogue centré), étape `embedder` dans `add_song` (`--no-embed` pour la sauter), `python -m indexer embed-missing` (réutilise les vecteurs de T18), `db/003_embeddings.sql` (`add_song` avec embedding, version du catalogue relancée par les embeddings). Le serveur garde les embeddings en mémoire et précalcule les 5 voisins à chaque chargement : réponse en une lecture de dictionnaire.

---

## Jalon G — Genre prédit

### T20 · Étiquettes et taxonomie
- Table de correspondance tags MusicBrainz → ~10 classes (`indexer/genre_map.yaml`), remplissage de `songs.genre_label`, export `.npz` des embeddings par segment + étiquettes (aucun audio).
- **Acceptation** : statistiques par classe publiées dans `docs/results.md` ; au moins 60 % du catalogue étiqueté (sinon on ajuste la taxonomie).
- **Réalisé** : 10 classes, 270/302 morceaux étiquetés (89 %), `genre_label` rempli dans Supabase et à chaque ajout, `export-genre-data`. Classique retiré, metal fondu dans le rock ; latino (6) et reggae (2) encore trop petits pour l'entraînement (`docs/results.md`).

### T21 · Entraînement sur Colab
- `notebooks/02_genre_classifier.ipynb` : séparation par morceau, régression logistique puis MLP PyTorch, F1 macro, matrice de confusion, comparaison sur FMA-small avec le même protocole.
- **Acceptation** : le notebook tourne de bout en bout sur Colab depuis un état vierge ; le MLP bat la régression logistique, sinon on garde la régression et on l'explique ; les métriques sont reportées dans `docs/results.md` ; les poids sont exportés.
- **Réalisé en local** : notebook généré par `notebooks/make_02_genre_classifier.py`, exécuté sur les vraies données (261 morceaux, 8 classes). Le **MLP ne bat pas la régression logistique** (F1 macro 0,388 contre 0,407, 2 plis gagnés sur 5), donc la régression est retenue, conformément à la règle fixée d'avance. Sur FMA-small, le MLP fait un peu mieux (0,572 contre 0,555) : c'est la taille du catalogue qui limite (`docs/results.md`). Export vérifié dans `models/genre_model.npz`. Reste le lancement sur Colab depuis un état vierge.

### T22 · Genre dans le pipeline et l'API
- Prédiction à l'indexation + rattrapage du catalogue ; table `genre_predictions` ; champ `genre` dans l'API.
- **Acceptation** : 100 % des morceaux ont une prédiction ; `model_version` est tracé ; test automatique de couverture (spec §7.3).
- **Réalisé (code)** : `earworm/genre.py` (prédiction numpy à partir de l'export T21), étape `genre_model` dans `add_song` (chargée si `models/genre_model.npz` existe), `python -m indexer predict-genres` (rattrapage, `--all` pour un nouveau modèle), `python -m indexer coverage` et test `EARWORM_LIVE=1` (spec §7.3), `db/004_genre_predictions.sql` (recharge du serveur), champ `genre` (avec le nom français de la classe) dans l'API. `model_version` = type, embedding, taxonomie et empreinte du fichier.

### T23 · PWA : fiche résultat complète
- Pochette, genre + probabilité, 5 similaires cliquables (`GET /api/songs/{id}`).
- **Acceptation** : parcours §2.1 complet sur téléphone ; affichage correct sans pochette et sans année.
- **Réalisé (code)** : fiche avec pochette (ou motif ♪, jamais d'image cassée), genre + probabilité + autres classes probables, 5 similaires cliquables (`GET /api/songs/{id}`), retour au morceau reconnu (bouton et geste retour du téléphone). Vérifié en local au format téléphone avec un faux catalogue : avec et sans pochette, sans année, sans genre, titre très long, aucun débordement. Restent la publication sur le Space et l'essai sur téléphone.

---

## Jalon H — Ajout collaboratif

### T24 · Propositions côté serveur et PWA
- `POST /api/proposals` (validation du lien, `409` / `422` / `429`, hash d'IP salé), écran « Proposer », lien depuis « non trouvé ».
- **Acceptation** : tests de chaque code de réponse ; 6ᵉ proposition dans l'heure → `429` ; aucune IP en clair en base (vérifié par requête SQL).

### T25 · Revue locale
- `python -m indexer review` (spec §2.3) branché sur le pipeline T11/T13.
- **Acceptation** : proposition faite depuis le téléphone → acceptée en local → reconnue au micro en ligne après le rechargement de l'index (≤ 10 min) ; refus avec motif enregistré.

---

## Jalon I — Mesure finale et vitrine

### T26 · Outillage d'évaluation
- Tirage du jeu de test (graine, `docs/eval/testset.csv`), conservation locale des extraits de test (spec §7.1), mode éval de la PWA (`?run_id=`), script de dépouillement (précision, intervalle de Wilson, faux positifs, p50/p95).
- **Acceptation** : une session d'essai de 10 extraits produit un rapport automatique correct.

### T27 · Campagne de mesure
- Catalogue ≥ 1 000 morceaux ; 100 positifs + 20 négatifs au micro en pièce bruyante ; 100 mesures de latence (Wi-Fi + 4G) ; réveil à froid mesuré à part.
- **Acceptation** : `docs/results.md` contient les chiffres, la date, le commit et `params_version` ; critères de `DECISIONS.md` atteints. Sinon : tickets de correction créés et campagne relancée.

### T28 · README anglais et démo
- README : pitch, architecture (schéma), fonctionnement des empreintes (PNG de T03), chiffres de T27 et T21, GIF de démo, mention des garde-fous audio, instructions d'installation.
- **Acceptation** : un lecteur extérieur comprend et lance le projet en local en suivant le README ; le GIF montre une reconnaissance réelle.

**→ v1 terminée.**
