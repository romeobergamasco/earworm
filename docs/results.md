# Earworm — résultats de mesure

Chaque section donne la date, le commit de départ et `PARAMS_VERSION`. Les rapports bruts du banc sont régénérables avec les commandes indiquées. Le dossier `results/` n'est pas versionné.

---

## T07 — Banc synthétique et seuils (2026-11-13, `PARAMS_VERSION = 2`)

### Protocole
- **Bibliothèque** : 10 morceaux de styles et d'époques variés (soul, reggae, jazz, rap, latino, metal, pop, électro, rock, synth-pop), 49 min au total.
- **Extraits** : 10 par morceau, de 10 s, tirés au hasard entre 5 % et 95 % de la durée, soit 100 extraits par condition.
- **Dégradation** (`earworm/bench.py`) :
  1. coupe des basses sous 250 Hz (haut-parleur de téléphone ou de portable) ;
  2. réverbération de pièce synthétique (RT60 de 0,5 s) ;
  3. bruit ajouté à un RSB de 10, 5, 0, -5 ou -10 dB. Deux types de bruit :
     - **pink** : bruit en 1/f, comme une ventilation, du trafic ou du vent ;
     - **babble** : 6 « voix » de bruit dans la bande de la parole, découpées au rythme des syllabes, comme un brouhaha de conversation ;
  4. gain aléatoire entre -20 et 0 dB.
- **Négatifs** : chaque requête est aussi évaluée comme si son morceau était absent du catalogue (*leave-one-out*). L'index note chaque morceau indépendamment, donc il suffit de retirer le vrai morceau des résultats.
- **Critère** : « reconnu » = bon morceau, avec `score ≥ MIN_SCORE` et `ratio ≥ MIN_RATIO`. « Faux positif » = un morceau est annoncé alors que le bon est absent.

### Choix des paramètres (balayage, tirage `--seed 0`)
`python -m earworm.bench --sweep`. Les seuils sont recalculés pour chaque réglage, pour un catalogue de 1 000 morceaux (voir plus bas).

| pics/s max | fan-out | hashes/s stockés | 0 dB | -5 dB | -10 dB |
|---|---|---|---|---|---|
| 20 | 3 | 57 | 100 % | 86 % | 29 % |
| 20 | 5 | 94 | 100 % | 90 % | 34 % |
| **20** | **8** | **148** | **100 %** | **98 %** | **46 %** |
| 30 | 3 | 85 | 100 % | 82 % | 30 % |
| 30 | 5 *(avant T07)* | 141 | 100 % | 92 % | 39 % |
| 30 | 8 | 223 | 100 % | 97 % | 48 % |
| 45 | 3 | 109 | 99 % | 81 % | 28 % |
| 45 | 5 | 181 | 100 % | 93 % | 38 % |
| 45 | 8 | 287 | 100 % | 96 % | 49 % |

À 10 et 5 dB, tous les réglages sont à 100 %. Chaque cellule couvre 200 extraits (pink et babble).

**Lecture** : le **fan-out** (nombre de paires par pic) compte bien plus que la densité de pics. Le réglage **20 pics/s × fan-out 8** stocke presque autant que l'ancien réglage (148 hashes/s contre 141, ~80 Mo pour 1 000 morceaux), mais tient mieux dans le bruit fort (-5 dB : 98 % contre 92 %). Les réglages à 223 et 287 hashes/s ne font pas mieux et dépasseraient le budget de stockage.

### Seuils de décision
Un catalogue de 1 000 morceaux offre 100 fois plus d'occasions de coïncidence qu'un banc de 10 morceaux. On estime donc le score que le meilleur de 999 morceaux absents atteint **par hasard** :
- distribution mesurée du score « par hasard » d'un morceau absent, sur toutes les requêtes ;
- sa queue (log de la probabilité) est quasi linéaire, et on la prolonge, car 10 morceaux ne permettent pas d'observer directement des probabilités de 1 sur 100 000 ;
- P(meilleur des 999 ≥ s) ≤ 1 % donne **s = 12** pour 1 000 morceaux, et 14 pour 5 000.

Seuils retenus (`earworm/index.py`) :
- **`MIN_SCORE = 24`** : le double de 12. Cette marge couvre ce que le banc ne simule pas : remix, reprises, samples partagés entre morceaux.
- **`MIN_RATIO = 2.0`** : le 1er morceau doit avoir au moins deux fois le score du 2e. Une coïncidence a presque toujours un rival proche (ratio près de 1), alors qu'une vraie reconnaissance a un ratio de plusieurs dizaines.

### Confirmation sur un nouveau tirage (`--seed 1 --fixed`, seuils figés)
`python -m earworm.bench --seed 1 --fixed` : les extraits et le bruit diffèrent de ceux qui ont servi au choix.

| condition | RSB | n | reconnus | mauvais morceau | faux positifs (morceau absent) |
|---|---|---|---|---|---|
| propre | – | 100 | 100 % | 0 | 0 / 100 |
| pink | 10 dB | 100 | 100 % | 0 | 0 / 100 |
| pink | 5 dB | 100 | **100 %** | 0 | 0 / 100 |
| pink | 0 dB | 100 | 100 % | 0 | 0 / 100 |
| pink | -5 dB | 100 | 100 % | 0 | 0 / 100 |
| pink | -10 dB | 100 | 86 % | 0 | 0 / 100 |
| babble | 10 dB | 100 | 100 % | 0 | 0 / 100 |
| babble | 5 dB | 100 | **100 %** | 0 | 0 / 100 |
| babble | 0 dB | 100 | 100 % | 0 | 0 / 100 |
| babble | -5 dB | 100 | 93 % | 0 | 0 / 100 |
| babble | -10 dB | 100 | 23 % | 0 | 0 / 100 |

**Critères T07** : ≥ 90 % à 5 dB → **100 %** ✔ · faux positifs ≤ 2 % → **0 / 1 100** ✔.

### Limites
- 10 morceaux seulement. Le seuil de score repose sur une extrapolation, à revérifier sur le vrai catalogue (T27).
- Bruit et réverbération sont synthétiques. Le vrai juge, c'est le micro : T08 (local), puis T27 (en ligne, 100 extraits).
- Le brouhaha de conversation (babble) est le cas difficile : il occupe la même bande que les voix et les instruments. À -10 dB, c'est-à-dire quand la conversation couvre largement la musique, on ne reconnaît qu'un extrait sur quatre.
- Effet de bord sur le T03 : le plafond passe de 30 à 20 pics/s. Sur les 10 morceaux, la densité moyenne est de 19 pics/s, juste sous la cible de 20 à 40 qu'annonçait le plan. Ce chiffre du plan est remplacé par le réglage mesuré ici.

### Performance
- Empreinte d'un extrait de 10 s : ~25 ms ; recherche : ~1 ms (10 morceaux).
- Index de 10 morceaux : 11 s.

---

## T08 — Essai au vrai micro (2026-11-13, `PARAMS_VERSION = 3`)

### Micro du portable : abandonné
- Par la voie normale de Windows (MME, DirectSound, WASAPI), le micro du portable (AMD « Microphone Array ») ne livre que du silence (-97 dBFS) à Python. Pourtant, il n'est ni coupé (volume à 88 %) ni bloqué par les autorisations. C'est sans doute un traitement du pilote.
- En accès direct au pilote (WDM-KS), on obtient le signal brut d'une seule capsule, très bruité. Trois essais à volume fort, normal et faible donnent 1 reconnaissance sur 3, uniquement à fort volume.
- **Décision** : les essais se font avec le dictaphone d'un **téléphone**, le micro qu'utilisera l'appli. `python -m earworm listen` reste disponible pour d'autres machines.

### Protocole
- Le portable joue le fichier (haut-parleurs intégrés, volume Windows à 50/100). Un iPhone enregistre à environ 1 m avec l'appli Dictaphone (m4a), environ 15 s par morceau, un passage au choix.
- **Calme** : pièce silencieuse. **Bruyant** : vidéo d'ambiance de café jouée en même temps par le portable, à un volume proche de celui de la musique.
- 10 morceaux × 2 conditions = 20 enregistrements, identifiés sur **10 s** (de 2 à 12 s) : `python -m earworm identify data/mic/calme --start 2 --duration 10`.

### Premier résultat (`PARAMS_VERSION = 2`, réglage du T07)
**Calme 7/10, bruyant 5/10**, aucun mauvais morceau. Le bon morceau arrive toujours en tête, mais avec un score trop faible (3 à 17 sur les extraits ratés, pour un seuil de 24).

**Cause** : la précision en fréquence. On compare les pics de l'enregistrement à ceux du passage original correspondant (avec une tolérance de ±1 trame) :

| morceau | même bin | ±1 bin | ±2 bins |
|---|---|---|---|
| Respect (Aretha Franklin) | 10 % | 21 % | 32 % |
| Billie Jean | 10 % | 26 % | 31 % |
| Three Little Birds | 10 % | 30 % | 38 % |
| Take Five (reconnu) | 50 % | 59 % | 60 % |

Le trajet haut-parleur → air → micro déplace beaucoup de pics d'un ou deux bins (5,4 Hz chacun) : vibrato de la voix, harmoniques larges, égalisation du haut-parleur. Un hash demande que ses deux pics correspondent exactement, donc 10 % × 10 % ≈ 1 % des hashes survivent. Le banc synthétique ne l'avait pas vu, car sa chaîne (filtre + réverbération + bruit) ne déplace pas les fréquences.

**Pistes écartées** :
- blanchiment du spectre (retirer à chaque fréquence son niveau moyen) : les scores baissent, car le bruit de fond ressort ;
- ignorer tout ce qui est sous 300 Hz : léger gain seulement.

### Correction : des cases plus larges dans le hash
Les fréquences sont divisées par `FREQ_QUANT` et l'écart de temps par `DT_QUANT` avant d'être codés. Score du bon morceau sur les 20 extraits :

| variante | calme | bruyant | meilleur autre morceau |
|---|---|---|---|
| T07 (1 bin, 1 trame) | 8 · 16 · 97 · 272 · 62 · 98 · 67 · 9 · 58 · 132 | 10 · 8 · 60 · 71 · 70 · 35 · 45 · 8 · 3 · 17 | 5 |
| 4 bins, 1 trame | 45 · 62 · 132 · 334 · 137 · 141 · 93 · 44 · 87 · 166 | 77 · 48 · 96 · 136 · 169 · 66 · 68 · 37 · 4 · 34 | 10 |
| **4 bins, 2 trames** (retenu) | 64 · 95 · 197 · 394 · 186 · 224 · 131 · 57 · 128 · 230 | 93 · 59 · 133 · 167 · 212 · 99 · 87 · 56 · 6 · 51 | 13 |

Le stockage baisse même un peu : ~84 Ko par morceau, soit **~65 Mo pour 1 000 morceaux** (contre ~80 Mo), car des hashes plus grossiers se compressent mieux.

### Seuils : simulation plutôt que marge arbitraire
Des cases plus larges produisent aussi plus de coïncidences. La marge « ×2 » du T07 aurait porté le seuil à 60, ce qui aurait fait rater 3 extraits sur 20. Elle est remplacée par une **simulation** (`earworm.bench.false_positive_rate`) :
- on tire au hasard les scores de N morceaux absents, selon la distribution mesurée sur le banc (avec sa queue prolongée) ;
- on compte combien de fois le meilleur passe **les deux** critères : score suffisant, et ratio d'au moins 2 avec le deuxième.

Score minimal pour au plus 0,05 % de faux positifs : **27 pour 10 morceaux, 29 pour 100**. Pour 1 000 et 5 000 morceaux, le ratio suffit à lui seul : les deux meilleurs scores de hasard sont alors toujours proches. Les petits catalogues sont donc les plus risqués.

Retenu : **`MIN_SCORE = 30`, `MIN_RATIO = 2`**. Faux positifs simulés : ≤ 0,01 % pour toutes les tailles de 10 à 5 000 morceaux.

### Résultat (`PARAMS_VERSION = 3`)
- **Vrais enregistrements au téléphone : calme 10/10, bruyant 9/10, aucun mauvais morceau.** Le seul échec est Bohemian Rhapsody vers 1 min 45 : la partie ballade, au piano et à la voix douce, couverte par le café. Le bon morceau arrive en tête avec un score de 10, sous le seuil.
- **Banc synthétique** (`--seed 1 --fixed`, nouveau tirage) : 100 % jusqu'à -5 dB, et à -10 dB 100 % (pink) / 71 % (babble), contre 86 % / 23 % en v2. 0 faux positif sur 1 100.

**Critère T08** (au moins 8/10 en bruyant) : **9/10** ✔.

### Limites
- Les paramètres ont été corrigés **en regardant ces 20 enregistrements**. Le 9/10 est donc optimiste, et le vrai chiffre viendra de la campagne T27 (100 extraits neufs, catalogue de 1 000 morceaux).
- Un seul téléphone, une seule pièce, les haut-parleurs d'un portable comme source.
- Le banc synthétique ne simule pas le décalage des pics en fréquence. À ajouter si on y retouche.
