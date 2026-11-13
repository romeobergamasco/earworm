---
title: Earworm
emoji: 🎵
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Music recognition from 10 s of microphone audio
---

# Earworm

Hold your phone near the music, tap **Écouter**: Earworm recognises the song from ~10 seconds of audio using hand-written audio fingerprints (spectrogram peaks + hashing).

This Space only runs the server and the web app. Source code, method and measurements: see the project's GitHub repository.

**Audio policy:** the audio you record is sent once to be fingerprinted and is never stored. The catalogue only keeps fingerprints, never audio.
