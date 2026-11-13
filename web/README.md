# web (PWA)

HTML/CSS/JS simple, sans framework, servi par le serveur (`server/main.py`).

- `app.js` : capture micro (traitements du navigateur désactivés), rééchantillonnage à 11 025 Hz (`OfflineAudioContext`), envoi en PCM 16 bits à `/api/identify`, affichage du résultat.
- `sw.js` : coquille hors ligne (l'API passe toujours par le réseau).
- `?debug=1` : latences mesurées côté client et test avec un fichier audio au lieu du micro.
