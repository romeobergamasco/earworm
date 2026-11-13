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
