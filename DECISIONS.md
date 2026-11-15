# Earworm (nom provisoire) — décisions de cadrage (2026-11-13)

Issues d'une séance d'interrogatoire (grilling). Base pour la spec et le plan v1.

## But
- Projet CV, cible stage/poste **IA / data**.
- **Produit avec de vrais utilisateurs** (pas seulement une démo).
- Durée : **quelques semaines**, quelques heures par semaine.
- **Zéro achat** (tout gratuit).

## Produit — v1
1. **Reconnaissance** d'un morceau à partir d'environ 10 s de micro : empreintes audio codées à la main (spectrogramme, FFT, pics, hachage), façon Shazam.
2. **Sons similaires** : embeddings audio d'un modèle **pré-entraîné** (CLAP / MERT / OpenL3), recherche par similarité dans le catalogue.
3. **Genre prédit** : **classifieur entraîné par l'utilisateur** sur Colab, au-dessus des embeddings ; étiquettes de genre MusicBrainz ; comparaison possible avec GTZAN / FMA.
4. **Ajout collaboratif** : un utilisateur colle un lien YouTube, la proposition est stockée (anonyme, avec limite de fréquence) dans une **file validée à la main** par le script local.

Phase 2 (après la v1) : **reconnaissance par fredonnement**, construite sur la même base d'embeddings.

## Architecture
| Brique | Choix |
|---|---|
| Moteur audio / ML | Python (numpy, librosa, PyTorch) |
| Entraînement | Google Colab (offre étudiante) |
| Source audio | YouTube via yt-dlp — **indexation en LOCAL** (YouTube bloque les IP cloud) |
| Garde-fous | audio supprimé juste après l'analyse, seules empreintes et embeddings conservés, aucun audio servi, mention dans le README. **Seule exception (2026-11-15, T26)** : un extrait de 10 s pour chacun des 120 morceaux du jeu de test, gardé **en local uniquement** (`data/eval/clips/`, ignoré par git, jamais publié ni servi), pour le banc de non-régression. |
| Métadonnées | MusicBrainz (titre, artiste, année, genres) |
| Base | Supabase (Postgres gratuit, 500 Mo) : empreintes, embeddings, morceaux, propositions |
| Serveur + front | ~~Hugging Face Spaces : API Python et PWA servies par le même serveur~~ → **provisoire (2026-11-13)** : PWA en HTML/CSS/JS simple sur un **Space Hugging Face Static** (gratuit) + API Python sur **Render** (offre gratuite, Docker). Pas de framework, pas de Gradio. Voir « Points ouverts ». |
| Code | GitHub public, README en anglais avec chiffres et GIF de démo |

Machine de dev : portable Ryzen 5 7530U, 16 Go RAM, GPU AMD intégré (pas de CUDA), Python 3.12. Les entraînements lourds se font sur Colab.

## Catalogue v1
- Environ **1 000 morceaux** : un mélange des playlists de l'utilisateur et des classements (top France, monde, par genre).
- Plus tard : 5 000 morceaux et plus, en traitant la vitesse de recherche comme sujet d'optimisation.

## Critères de fin de la v1
- 1 000 morceaux indexés
- ≥ 90 % de reconnaissance sur des extraits de 10 s au micro, dans une pièce bruitée
- Réponse en < 3 s
- Sons similaires et genre prédit affichés pour chaque résultat
- Ajout par lien YouTube avec file de validation
- En ligne sur Hugging Face, README en anglais avec chiffres et GIF

## Points ouverts
- **Hébergement du serveur (provisoire, à revoir quand le projet aura avancé).** En 2026, Hugging Face a rendu payants les Spaces Docker et Gradio ; seuls les Spaces Static restent gratuits. Solution retenue faute de mieux : PWA sur un Space Static, API sur Render gratuit (512 Mo de RAM, mise en veille après 15 min sans trafic, ~1 min de réveil, 750 h/mois). Ce n'est pas le système souhaité : chercher une meilleure solution gratuite (sans mise en veille, plus de mémoire) une fois la v1 avancée.
