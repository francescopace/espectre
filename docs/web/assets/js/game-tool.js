/*
 * ESPectre - Game tool
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';

    /* ================================================================ game */

    const GAME_ORB_POINTS = 100;
    const GAME_START_DELAY_MS = 700;
    const gameGhostImage = new Image();
    gameGhostImage.decoding = 'async';
    gameGhostImage.src = '/assets/images/brand/espectre-logo.svg';
    const GAME_FACTORY_IMAGE_SOURCES = Object.freeze([
        '/assets/images/game/hardware-factory.avif',
        '/assets/images/game/hardware-factory.webp'
    ]);
    let gameFactoryImage = null;
    let gameFactoryImagePromise = null;
    const gameAudio = {
        context: null,
        master: null,
        music: null,
        motionOscillator: null,
        motionGain: null,
        musicTimer: null,
        nextNoteAt: 0,
        noteIndex: 0,
        motionSmoothed: 0,
        enabled: true
    };
    const GAME_MUSIC_NOTES = [220, 0, 330, 0, 262, 0, 392, 0, 247, 0, 370, 0, 294, 0, 440, 0];

    const game = {
        phase: 'idle',   // idle | ready | running | done
        score: 0,
        orbs: 0,
        best: 0,
        distance: 0,
        scrollX: 0,
        displayDistance: -1,
        elapsed: 0,
        raf: null,
        previewRaf: null,
        readyTimer: null,
        lastFrameAt: 0,
        previewLastFrameAt: 0,
        nextSpawn: 0,
        flightActive: false,
        manualFlight: false,
        width: 840,
        height: 340,
        dpr: 1,
        ctx: null,
        player: { x: 0, y: 0, size: 54, vy: 0, grounded: true },
        entities: [],
        particles: [],
        hitFlash: 0
    };

    function gameLoadFactoryImage() {
        if (gameFactoryImagePromise) return gameFactoryImagePromise;
        gameFactoryImagePromise = new Promise((resolve) => {
            const image = new Image();
            image.decoding = 'async';
            let sourceIndex = 0;
            const finish = (loaded) => {
                image.removeEventListener('load', handleLoad);
                image.removeEventListener('error', handleError);
                if (loaded) {
                    gameFactoryImage = image;
                    gameDraw();
                }
                resolve(loaded ? image : null);
            };
            const handleLoad = () => finish(true);
            const handleError = () => {
                sourceIndex += 1;
                if (sourceIndex < GAME_FACTORY_IMAGE_SOURCES.length) {
                    image.src = GAME_FACTORY_IMAGE_SOURCES[sourceIndex];
                    return;
                }
                finish(false);
            };
            image.addEventListener('load', handleLoad);
            image.addEventListener('error', handleError);
            image.src = GAME_FACTORY_IMAGE_SOURCES[sourceIndex];
        });
        return gameFactoryImagePromise;
    }

    function gameSet(selector, value) {
        const el = $(selector);
        if (el) el.textContent = value;
    }

    function gameMsg(message) {
        gameSet('.js-game-msg', message);
    }

    function gameAudioEnsure() {
        if (!gameAudio.enabled) return null;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return null;
        if (!gameAudio.context) {
            gameAudio.context = new AudioContext();
            gameAudio.master = gameAudio.context.createGain();
            gameAudio.master.gain.value = 0.32;
            gameAudio.master.connect(gameAudio.context.destination);
            gameAudio.music = gameAudio.context.createGain();
            gameAudio.music.gain.value = 0.22;
            gameAudio.music.connect(gameAudio.master);
            gameAudio.motionGain = gameAudio.context.createGain();
            gameAudio.motionGain.gain.value = 0.0001;
            gameAudio.motionGain.connect(gameAudio.master);
        }
        gameAudio.context.resume().catch(() => {});
        return gameAudio.context;
    }

    function gameTone(frequency, start, duration, {
        type = 'sine',
        gain = 0.1,
        endFrequency = frequency,
        destination = null
    } = {}) {
        const context = gameAudio.context;
        if (!context || !frequency) return;
        const oscillator = context.createOscillator();
        const envelope = context.createGain();
        oscillator.type = type;
        oscillator.frequency.setValueAtTime(frequency, start);
        oscillator.frequency.exponentialRampToValueAtTime(Math.max(1, endFrequency), start + duration);
        envelope.gain.setValueAtTime(0.0001, start);
        envelope.gain.exponentialRampToValueAtTime(gain, start + Math.min(0.025, duration * 0.2));
        envelope.gain.exponentialRampToValueAtTime(0.0001, start + duration);
        oscillator.connect(envelope);
        envelope.connect(destination || gameAudio.master);
        oscillator.start(start);
        oscillator.stop(start + duration + 0.03);
    }

    function gamePlaySound(kind) {
        const context = gameAudioEnsure();
        if (!context) return;
        const now = context.currentTime + 0.01;
        if (kind === 'start') {
            gameTone(392, now, 0.09, { type: 'triangle', gain: 0.09, endFrequency: 523 });
            gameTone(523, now + 0.1, 0.14, { type: 'triangle', gain: 0.1, endFrequency: 659 });
        } else if (kind === 'orb') {
            gameTone(740, now, 0.06, { type: 'sine', gain: 0.08, endFrequency: 880 });
            gameTone(1047, now + 0.065, 0.12, { type: 'sine', gain: 0.07, endFrequency: 1319 });
        } else if (kind === 'hit') {
            gameTone(176, now, 0.28, { type: 'sawtooth', gain: 0.12, endFrequency: 48 });
            gameTone(88, now + 0.04, 0.35, { type: 'triangle', gain: 0.12, endFrequency: 34 });
        }
    }

    function gameScheduleMusic() {
        if (!gameAudio.context || gameAudio.musicTimer === null || game.phase === 'done') return;
        const context = gameAudio.context;
        const stepSeconds = 0.22;
        while (gameAudio.nextNoteAt < context.currentTime + 0.65) {
            const note = GAME_MUSIC_NOTES[gameAudio.noteIndex % GAME_MUSIC_NOTES.length];
            if (note) {
                gameTone(note, gameAudio.nextNoteAt, 0.16, {
                    type: 'triangle',
                    gain: 0.055,
                    endFrequency: note * 1.004,
                    destination: gameAudio.music
                });
            }
            if (gameAudio.noteIndex % 4 === 0) {
                gameTone(55, gameAudio.nextNoteAt, 0.18, {
                    type: 'sine',
                    gain: 0.065,
                    endFrequency: 52,
                    destination: gameAudio.music
                });
            }
            gameAudio.noteIndex += 1;
            gameAudio.nextNoteAt += stepSeconds;
        }
        gameAudio.musicTimer = setTimeout(gameScheduleMusic, 180);
    }

    function gameStartMusic() {
        const context = gameAudioEnsure();
        if (!context || gameAudio.musicTimer !== null) return;
        gameAudio.music.gain.cancelScheduledValues(context.currentTime);
        gameAudio.music.gain.setValueAtTime(0.0001, context.currentTime);
        gameAudio.music.gain.exponentialRampToValueAtTime(0.22, context.currentTime + 0.12);
        gameAudio.noteIndex = 0;
        gameAudio.nextNoteAt = context.currentTime + 0.04;
        gameAudio.musicTimer = setTimeout(gameScheduleMusic, 0);
    }

    function gameStartMotionSound() {
        const context = gameAudioEnsure();
        if (!context || gameAudio.motionOscillator) return;
        gameAudio.motionOscillator = context.createOscillator();
        gameAudio.motionOscillator.type = 'sine';
        gameAudio.motionOscillator.frequency.value = 100;
        gameAudio.motionOscillator.connect(gameAudio.motionGain);
        gameAudio.motionOscillator.start();
    }

    function gameUpdateMotionSound(dt) {
        if (!gameAudio.context || !gameAudio.motionOscillator || !gameAudio.motionGain) return;
        const tau = evaluationIntervalMs() / 2000;
        const alpha = 1 - Math.exp(-dt / tau);
        gameAudio.motionSmoothed += (energyFraction() - gameAudio.motionSmoothed) * alpha;
        const context = gameAudio.context;
        const now = context.currentTime;
        const frequency = 96 * Math.pow(2, gameAudio.motionSmoothed * 1.8);
        const audible = (game.phase === 'ready' || game.phase === 'running') && game.flightActive;
        gameAudio.motionOscillator.frequency.setTargetAtTime(frequency, now, 0.06);
        gameAudio.motionGain.gain.setTargetAtTime(
            audible ? 0.014 + gameAudio.motionSmoothed * 0.05 : 0.0001,
            now,
            0.08
        );
    }

    function gameStopMusic() {
        clearTimeout(gameAudio.musicTimer);
        gameAudio.musicTimer = null;
        if (gameAudio.context && gameAudio.music) {
            const now = gameAudio.context.currentTime;
            gameAudio.music.gain.cancelScheduledValues(now);
            gameAudio.music.gain.setValueAtTime(Math.max(0.0001, gameAudio.music.gain.value), now);
            gameAudio.music.gain.exponentialRampToValueAtTime(0.0001, now + 0.12);
        }
        if (gameAudio.context && gameAudio.motionGain) {
            gameAudio.motionGain.gain.setTargetAtTime(0.0001, gameAudio.context.currentTime, 0.06);
        }
    }

    function gameRenderSoundControl() {
        const button = $('.js-game-sound');
        if (!button) return;
        button.textContent = gameAudio.enabled ? 'Sound on' : 'Sound off';
        button.setAttribute('aria-label', gameAudio.enabled ? 'Mute game audio' : 'Enable game audio');
        button.setAttribute('aria-pressed', String(gameAudio.enabled));
    }

    function gameToggleSound() {
        gameAudio.enabled = !gameAudio.enabled;
        if (!gameAudio.enabled) gameStopMusic();
        else if (game.phase === 'ready' || game.phase === 'running') {
            gameStartMusic();
            gameStartMotionSound();
        }
        gameRenderSoundControl();
    }

    function gameSetPhase(phase, badge) {
        game.phase = phase;
        const screen = $('.game-screen');
        if (screen) screen.dataset.phase = phase;
        gameSet('.js-game-badge', badge);
        const play = $('.js-game-start');
        if (play) play.hidden = phase === 'ready' || phase === 'running';
        gameSyncFullscreenButton();
    }

    function gameFullscreenElement() {
        return document.fullscreenElement || document.webkitFullscreenElement || null;
    }

    function gameSyncFullscreenButton() {
        const screen = $('.game-screen');
        const button = $('.js-game-fullscreen');
        if (!screen || !button) return;
        const supported = Boolean(
            (screen.requestFullscreen || screen.webkitRequestFullscreen)
            && (document.exitFullscreen || document.webkitExitFullscreen)
        );
        const active = gameFullscreenElement() === screen;
        button.hidden = !supported;
        button.textContent = active ? 'Exit full screen' : 'Full screen';
        button.setAttribute('aria-label', active ? 'Exit fullscreen' : 'Enter fullscreen');
        button.setAttribute('aria-pressed', String(active));
    }

    function gameExitFullscreen() {
        const screen = $('.game-screen');
        if (!screen || gameFullscreenElement() !== screen) return;
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (!exit) return;
        Promise.resolve(exit.call(document)).catch(() => {});
    }

    async function gameToggleFullscreen() {
        const screen = $('.game-screen');
        if (!screen) return;
        if (gameFullscreenElement() === screen) {
            gameExitFullscreen();
            return;
        }
        const request = screen.requestFullscreen || screen.webkitRequestFullscreen;
        if (!request) return;
        try {
            await request.call(screen);
        } catch (error) {
            toast('Full screen is unavailable.');
        }
    }

    function gameOnFullscreenChange() {
        gameSyncFullscreenButton();
        requestAnimationFrame(gameResizeCanvas);
        if (gameFullscreenElement() === $('.game-screen')) {
            $('.js-game-canvas').focus({ preventScroll: true });
        }
    }

    function gameGroundY() {
        return game.height * 0.79;
    }

    function gamePlayerSize() {
        return Math.max(38, Math.min(58, game.height * 0.17));
    }

    function gameFlightY() {
        const size = game.player.size || gamePlayerSize();
        return Math.max(game.height * 0.14, gameGroundY() - size - game.height * 0.34);
    }

    function gameResetPlayer() {
        const size = gamePlayerSize();
        game.player = {
            x: game.width * 0.14,
            y: gameGroundY() - size,
            size,
            vy: 0,
            grounded: true
        };
    }

    function gameResizeCanvas() {
        const canvas = $('.js-game-canvas');
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        if (rect.width < 1 || rect.height < 1) return;

        const oldWidth = game.width;
        const oldHeight = game.height;
        const oldGround = gameGroundY();
        const oldSize = game.player.size;
        const air = oldGround - (game.player.y + oldSize);
        game.width = rect.width;
        game.height = rect.height;
        game.dpr = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = Math.round(game.width * game.dpr);
        canvas.height = Math.round(game.height * game.dpr);
        game.ctx = canvas.getContext('2d');
        if (!game.ctx) return;
        game.ctx.setTransform(game.dpr, 0, 0, game.dpr, 0, 0);

        const scaleX = oldWidth > 0 ? game.width / oldWidth : 1;
        const scaleY = oldHeight > 0 ? game.height / oldHeight : 1;
        game.scrollX *= scaleX;
        const size = gamePlayerSize();
        game.player.x = game.width * 0.14;
        game.player.size = size;
        const maxAir = Math.max(0, gameGroundY() - size - gameFlightY());
        game.player.y = gameGroundY() - size - Math.min(maxAir, Math.max(0, air * scaleY));
        game.entities.forEach((entity) => {
            entity.x *= scaleX;
            if (entity.kind === 'orb') {
                entity.y = gameOrbY(entity.lane);
                entity.radius = Math.max(5, Math.min(8, size * 0.12));
            } else {
                const dimensions = gameObstacleDimensions(entity.obstacleKind);
                entity.w = dimensions.w;
                entity.h = dimensions.h;
                entity.y = gameObstacleY(entity.obstacleKind, entity.h);
            }
        });
        gameDraw();
    }

    function gameScore() {
        return Math.floor(game.distance) + game.orbs * GAME_ORB_POINTS;
    }

    function gameUpdateStats() {
        game.score = gameScore();
        gameSet('.js-game-score', String(game.score));
        gameSet('.js-game-orbs', String(game.orbs));
        game.displayDistance = Math.floor(game.distance);
        gameSet('.js-game-distance', game.displayDistance + ' m');
        gameSet('.js-game-best', String(game.best));
    }

    function reportGameAbandon(reason) {
        if (game.phase === 'idle' || game.phase === 'done') return;
        track('game_abandon', {
            input_mode: connectionInputMode(),
            score: game.score,
            distance: Math.floor(game.distance),
            reason
        });
        gameReset();
    }

    function gameReset() {
        clearTimeout(game.readyTimer);
        cancelAnimationFrame(game.raf);
        cancelAnimationFrame(game.previewRaf);
        game.raf = null;
        game.previewRaf = null;
        game.score = 0;
        game.orbs = 0;
        game.distance = 0;
        game.scrollX = 0;
        game.elapsed = 0;
        game.entities = [];
        game.particles = [];
        game.hitFlash = 0;
        game.flightActive = false;
        game.manualFlight = false;
        gameAudio.motionSmoothed = 0;
        gameStopMusic();
        gameSetPhase('idle', 'READY');
        gameMsg('Move to fly. Stay quiet to descend. Distance earns points, and orbs add bonuses.');
        gameUpdateStats();
        const start = $('.js-game-start');
        if (start) start.textContent = 'Start game';
        gameResetPlayer();
        gameDraw();
        gameStartPreview();
    }

    function gameObstacleDimensions(kind) {
        const size = game.player.size || gamePlayerSize();
        if (kind === 'aerial_spikes') return { w: size * 1.12, h: size * 0.7 };
        if (kind === 'gate') return { w: size * 0.76, h: size * 0.86 };
        return { w: size * 1.08, h: size * 0.56 };
    }

    function gameObstacleY(kind, height) {
        if (kind === 'aerial_spikes') return gameFlightY() + game.player.size * 0.1;
        return gameGroundY() - height;
    }

    function gameAddObstacle(kind, x) {
        const dimensions = gameObstacleDimensions(kind);
        game.entities.push({
            kind: 'obstacle',
            obstacleKind: kind,
            x,
            y: gameObstacleY(kind, dimensions.h),
            ...dimensions
        });
    }

    function gameOrbY(lane) {
        const size = game.player.size || gamePlayerSize();
        return lane === 'high'
            ? gameFlightY() + size * 0.5
            : gameGroundY() - size * 0.46;
    }

    function gameAddOrb(x, lane) {
        game.entities.push({
            kind: 'orb',
            x,
            y: gameOrbY(lane),
            lane,
            radius: Math.max(5, Math.min(8, game.player.size * 0.12)),
            phase: Math.random() * Math.PI * 2
        });
    }

    function gameSpawnCourse() {
        const size = game.player.size;
        const startX = game.width + size;
        const pattern = Math.random();
        if (pattern < 0.42) {
            const obstacleX = startX + size * 3.2;
            gameAddObstacle(Math.random() < 0.56 ? 'spikes' : 'gate', obstacleX);
            for (let i = 0; i < 4; i += 1) {
                gameAddOrb(obstacleX - size * (2.8 - i * 0.72), 'high');
            }
        } else if (pattern < 0.72) {
            const obstacleX = startX + size * 3.2;
            gameAddObstacle('aerial_spikes', obstacleX);
            for (let i = 0; i < 4; i += 1) {
                gameAddOrb(obstacleX - size * (2.8 - i * 0.72), 'low');
            }
        } else if (pattern < 0.86) {
            for (let i = 0; i < 5; i += 1) {
                gameAddOrb(startX + i * size * 0.68, 'low');
            }
        } else {
            for (let i = 0; i < 5; i += 1) {
                gameAddOrb(startX + i * size * 0.7, 'high');
            }
        }
    }

    function gameRectsOverlap(a, b) {
        return a.x < b.x + b.w && a.x + a.w > b.x
            && a.y < b.y + b.h && a.y + a.h > b.y;
    }

    function gameOrbTouchesPlayer(orb, player) {
        const closestX = Math.max(player.x, Math.min(orb.x, player.x + player.w));
        const closestY = Math.max(player.y, Math.min(orb.y, player.y + player.h));
        const dx = orb.x - closestX;
        const dy = orb.y - closestY;
        return dx * dx + dy * dy <= orb.radius * orb.radius;
    }

    function gameBurst(orb) {
        for (let i = 0; i < 9; i += 1) {
            const angle = (Math.PI * 2 * i) / 9;
            const speed = 28 + Math.random() * 48;
            game.particles.push({
                x: orb.x,
                y: orb.y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 0.42
            });
        }
    }

    function gameFinish() {
        game.hitFlash = 1;
        game.best = Math.max(game.best, game.score);
        gamePlaySound('hit');
        gameStopMusic();
        gameSetPhase('done', 'GAME OVER');
        gameMsg('Obstacle hit — ' + game.score + ' points over ' + Math.floor(game.distance) + ' m.');
        gameUpdateStats();
        $('.js-game-start').textContent = 'Play again';
        track('game_over', {
            input_mode: connectionInputMode(),
            score: game.score,
            orbs: game.orbs,
            distance: Math.floor(game.distance)
        });
    }

    function gameSetFlight(active) {
        game.flightActive = Boolean(active);
        if (game.phase === 'running') {
            gameSet('.js-game-badge', game.flightActive ? 'FLY' : 'GLIDE');
        }
    }

    function gameUpdatePlayer(dt) {
        const ground = gameGroundY();
        const player = game.player;
        const targetY = game.flightActive ? gameFlightY() : ground - player.size;
        const responseSeconds = game.flightActive ? 0.15 : 0.18;
        const blend = 1 - Math.exp(-dt / responseSeconds);
        const previousY = player.y;
        player.y += (targetY - player.y) * blend;
        player.vy = dt > 0 ? (player.y - previousY) / dt : 0;
        player.grounded = Math.abs(player.y - (ground - player.size)) < 1;
    }

    function gamePreviewFrame(now) {
        const previewing = route === 'tool-game' && conn.mode
            && (game.phase === 'idle' || game.phase === 'ready');
        if (!previewing) {
            game.previewRaf = null;
            return;
        }
        const dt = Math.min(0.05, Math.max(0, (now - game.previewLastFrameAt) / 1000));
        game.previewLastFrameAt = now;
        game.elapsed += dt;
        gameUpdatePlayer(dt);
        gameUpdateMotionSound(dt);
        gameDraw();
        game.previewRaf = requestAnimationFrame(gamePreviewFrame);
    }

    function gameStartPreview() {
        if (game.previewRaf || route !== 'tool-game' || !conn.mode
                || (game.phase !== 'idle' && game.phase !== 'ready')) return;
        game.previewLastFrameAt = performance.now();
        game.previewRaf = requestAnimationFrame(gamePreviewFrame);
    }

    function gameStopPreview() {
        cancelAnimationFrame(game.previewRaf);
        game.previewRaf = null;
    }

    function gameUpdate(dt) {
        game.elapsed += dt;
        gameUpdatePlayer(dt);
        gameUpdateMotionSound(dt);
        const player = game.player;
        const speed = Math.max(190, game.width * 0.32)
            + Math.min(game.width * 0.2, game.elapsed * 5.2);

        const travel = speed * dt;
        game.scrollX += travel;
        game.distance += travel / 45;
        game.nextSpawn -= travel;
        if (game.nextSpawn <= 0) {
            gameSpawnCourse();
            game.nextSpawn = Math.max(
                game.width * 0.58,
                speed * (1.75 + Math.random() * 0.45)
            );
        }

        const hitbox = {
            x: player.x + player.size * 0.18,
            y: player.y + player.size * 0.12,
            w: player.size * 0.64,
            h: player.size * 0.76
        };
        let obstacleHit = false;
        game.entities.forEach((entity) => {
            entity.x -= travel;
            if (entity.kind === 'orb') {
                entity.phase += dt * 5;
                if (!entity.collected && gameOrbTouchesPlayer(entity, hitbox)) {
                    entity.collected = true;
                    game.orbs += 1;
                    gameBurst(entity);
                    gamePlaySound('orb');
                    gameUpdateStats();
                }
                return;
            }
            const obstacleHitbox = {
                x: entity.x + 3,
                y: entity.y + 4,
                w: Math.max(1, entity.w - 6),
                h: Math.max(1, entity.h - 4)
            };
            if (gameRectsOverlap(hitbox, obstacleHitbox)) obstacleHit = true;
        });
        game.entities = game.entities.filter((entity) => {
            const width = entity.kind === 'orb' ? entity.radius * 2 : entity.w;
            return !entity.collected && entity.x + width > -24;
        });
        game.particles.forEach((particle) => {
            particle.x += particle.vx * dt;
            particle.y += particle.vy * dt;
            particle.vy += game.height * 1.05 * dt;
            particle.life -= dt;
        });
        game.particles = game.particles.filter((particle) => particle.life > 0);
        const distance = Math.floor(game.distance);
        const score = gameScore();
        if (score !== game.score) {
            game.score = score;
            gameSet('.js-game-score', String(score));
        }
        if (distance !== game.displayDistance) {
            game.displayDistance = distance;
            gameSet('.js-game-distance', distance + ' m');
        }
        if (obstacleHit) gameFinish();
    }

    function gameSensingActive() {
        return conn.movement >= gameThreshold();
    }

    function gameOnTelemetry() {
        if (route !== 'tool-game') return;
        gameSetFlight(gameSensingActive());
        gameStartPreview();
    }

    function gameRoundedRect(ctx, x, y, width, height, radius) {
        const r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + width, y, x + width, y + height, r);
        ctx.arcTo(x + width, y + height, x, y + height, r);
        ctx.arcTo(x, y + height, x, y, r);
        ctx.arcTo(x, y, x + width, y, r);
        ctx.closePath();
    }

    function gameDrawFactoryBackdrop(ctx, width, height) {
        if (!gameFactoryImage?.complete || !gameFactoryImage.naturalWidth) return;
        const imageWidth = gameFactoryImage.naturalWidth;
        const imageHeight = gameFactoryImage.naturalHeight;
        const canvasRatio = width / height;
        const imageRatio = imageWidth / imageHeight;
        let sourceX = 0;
        let sourceY = 0;
        let sourceWidth = imageWidth;
        let sourceHeight = imageHeight;
        if (imageRatio > canvasRatio) {
            sourceWidth = imageHeight * canvasRatio;
            const margin = imageWidth - sourceWidth;
            sourceX = margin * (0.5 + Math.sin(game.scrollX * 0.0007) * 0.5);
        } else if (imageRatio < canvasRatio) {
            sourceHeight = imageWidth / canvasRatio;
            sourceY = Math.max(0, imageHeight - sourceHeight);
        }
        ctx.save();
        ctx.globalAlpha = 0.86;
        ctx.drawImage(
            gameFactoryImage,
            sourceX,
            sourceY,
            sourceWidth,
            sourceHeight,
            0,
            0,
            width,
            height
        );
        ctx.restore();
    }

    function gameDrawFactoryParallax(ctx, ground) {
        const span = game.width + 180;
        ctx.save();
        ctx.globalAlpha = 0.34;
        for (let i = 0; i < 7; i += 1) {
            const x = ((i * 211 - game.scrollX * 0.12) % span + span) % span - 90;
            const y = ground * (0.2 + (i % 3) * 0.13);
            const panelWidth = 42 + (i % 3) * 18;
            const panelHeight = 20 + (i % 2) * 10;
            ctx.fillStyle = '#101b47';
            gameRoundedRect(ctx, x, y, panelWidth, panelHeight, 4);
            ctx.fill();
            ctx.strokeStyle = i % 2 ? 'rgba(121, 139, 255, .75)' : 'rgba(85, 211, 211, .52)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x + 5, y + panelHeight * 0.5);
            ctx.lineTo(x + panelWidth - 8, y + panelHeight * 0.5);
            ctx.stroke();
        }
        ctx.restore();

        ctx.save();
        ctx.globalAlpha = 0.24;
        ctx.strokeStyle = '#6075dd';
        ctx.lineWidth = 1;
        for (let i = 0; i < 9; i += 1) {
            const x = ((i * 173 - game.scrollX * 0.18) % span + span) % span - 70;
            ctx.beginPath();
            ctx.moveTo(x, ground * 0.56);
            ctx.lineTo(x + 24, ground * 0.56);
            ctx.lineTo(x + 34, ground * 0.64);
            ctx.stroke();
        }
        ctx.restore();
    }

    function gameDrawBackground(ctx) {
        const width = game.width;
        const height = game.height;
        const ground = gameGroundY();
        const sky = ctx.createLinearGradient(0, 0, 0, height);
        sky.addColorStop(0, '#070810');
        sky.addColorStop(0.72, '#111328');
        sky.addColorStop(1, '#080911');
        ctx.fillStyle = sky;
        ctx.fillRect(0, 0, width, height);
        gameDrawFactoryBackdrop(ctx, width, height);
        gameDrawFactoryParallax(ctx, ground);

        ctx.save();
        ctx.globalAlpha = 0.58;
        for (let i = 0; i < 15; i += 1) {
            const span = width + 90;
            const x = ((i * 137 - game.scrollX * 0.24) % span + span) % span - 45;
            const y = 24 + ((i * 53) % Math.max(40, ground - 88));
            ctx.fillStyle = i % 3 === 0 ? '#9eb0ff' : '#5369d8';
            ctx.beginPath();
            ctx.arc(x, y, i % 3 === 0 ? 1.7 : 1, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();

        const floor = ctx.createLinearGradient(0, ground, 0, height);
        floor.addColorStop(0, '#20243a');
        floor.addColorStop(0.12, '#121522');
        floor.addColorStop(1, '#08090e');
        ctx.fillStyle = floor;
        ctx.fillRect(0, ground, width, height - ground);
        ctx.fillStyle = '#6677e5';
        ctx.globalAlpha = 0.52;
        ctx.fillRect(0, ground, width, 2);
        ctx.globalAlpha = 1;

        ctx.fillStyle = 'rgba(80, 105, 222, .3)';
        ctx.fillRect(0, ground + 4, width, Math.max(8, height * 0.04));
        for (let i = 0; i < 9; i += 1) {
            const span = width / 8;
            const x = ((i * span - game.scrollX) % width + width) % width;
            ctx.fillStyle = 'rgba(184, 197, 255, .28)';
            ctx.fillRect(x, ground + 7, Math.max(8, span * 0.34), 2);
        }

        ctx.strokeStyle = 'rgba(111, 138, 230, .22)';
        ctx.lineWidth = 1;
        for (let i = 0; i < 8; i += 1) {
            const span = width / 7;
            const x = ((i * span - game.scrollX) % width + width) % width;
            ctx.beginPath();
            ctx.moveTo(x, ground + 5);
            ctx.lineTo(x - height * 0.17, height);
            ctx.stroke();
        }
    }

    function gameDrawOrb(ctx, entity) {
        const y = entity.y + Math.sin(entity.phase) * 2.2;
        ctx.save();
        ctx.shadowColor = 'rgba(255, 194, 91, .82)';
        ctx.shadowBlur = 13;
        const glow = ctx.createRadialGradient(
            entity.x - entity.radius * 0.35,
            y - entity.radius * 0.4,
            entity.radius * 0.1,
            entity.x,
            y,
            entity.radius
        );
        glow.addColorStop(0, '#fff5c0');
        glow.addColorStop(0.34, '#f8c86d');
        glow.addColorStop(1, '#bd7138');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(entity.x, y, entity.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    function gameDrawChip(ctx, x, y, width, height, { hanging = false, label = '' } = {}) {
        const pinLength = Math.max(3, Math.min(7, width * 0.16));
        const pinCount = Math.max(2, Math.floor(height / Math.max(7, height * 0.2)));
        ctx.save();
        ctx.shadowColor = 'rgba(76, 123, 238, .52)';
        ctx.shadowBlur = 10;
        const body = ctx.createLinearGradient(x, y, x + width, y + height);
        body.addColorStop(0, '#5868a8');
        body.addColorStop(0.35, '#222a55');
        body.addColorStop(1, '#11162e');
        ctx.fillStyle = body;
        gameRoundedRect(ctx, x, y, width, height, Math.min(6, width * 0.16));
        ctx.fill();
        ctx.strokeStyle = '#98a8ff';
        ctx.globalAlpha = 0.72;
        ctx.lineWidth = 1;
        gameRoundedRect(ctx, x + 1, y + 1, width - 2, height - 2, Math.min(5, width * 0.13));
        ctx.stroke();
        ctx.globalAlpha = 1;

        ctx.fillStyle = '#090d20';
        gameRoundedRect(ctx, x + width * 0.22, y + height * 0.2, width * 0.56, height * 0.58, 3);
        ctx.fill();
        ctx.strokeStyle = 'rgba(92, 231, 228, .7)';
        ctx.lineWidth = 1;
        for (let i = 0; i < 3; i += 1) {
            const traceY = y + height * (0.34 + i * 0.15);
            ctx.beginPath();
            ctx.moveTo(x + width * 0.08, traceY);
            ctx.lineTo(x + width * 0.22, traceY);
            ctx.lineTo(x + width * 0.28, traceY + (i - 1) * 2);
            ctx.stroke();
        }
        if (label) {
            ctx.fillStyle = '#c2ccff';
            ctx.globalAlpha = 0.78;
            ctx.font = `bold ${Math.max(5, Math.min(8, width * 0.2))}px system-ui`;
            ctx.textAlign = 'center';
            ctx.fillText(label, x + width * 0.5, y + height * 0.62);
            ctx.globalAlpha = 1;
        }
        ctx.fillStyle = '#b8c8ff';
        for (let i = 0; i < pinCount; i += 1) {
            const pinY = y + height * ((i + 0.5) / pinCount) - 1;
            ctx.fillRect(x - pinLength, pinY, pinLength, 2);
            ctx.fillRect(x + width, pinY, pinLength, 2);
        }
        if (hanging) {
            ctx.fillStyle = '#c1ccff';
            const bottomPins = Math.max(2, Math.floor(width / 9));
            for (let i = 0; i < bottomPins; i += 1) {
                const pinX = x + width * ((i + 0.5) / bottomPins) - 1;
                ctx.fillRect(pinX, y + height, 2, pinLength);
            }
        }
        ctx.restore();
    }

    function gameDrawObstacle(ctx, entity) {
        if (entity.obstacleKind === 'aerial_spikes') {
            ctx.save();
            ctx.strokeStyle = 'rgba(160, 177, 255, .75)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(entity.x + entity.w * 0.5, 0);
            ctx.lineTo(entity.x + entity.w * 0.5, entity.y + entity.h * 0.16);
            ctx.stroke();
            gameDrawChip(ctx, entity.x, entity.y + entity.h * 0.16, entity.w, entity.h * 0.74, {
                hanging: true,
                label: 'IO',
            });
            ctx.restore();
            return;
        }

        if (entity.obstacleKind === 'gate') {
            gameDrawChip(ctx, entity.x, entity.y, entity.w, entity.h, { label: 'ESP' });
            return;
        }

        gameDrawChip(ctx, entity.x, entity.y, entity.w, entity.h, { label: 'IC' });
    }

    function gameDrawPlayer(ctx) {
        const player = game.player;
        const canFloat = game.phase === 'idle' || game.phase === 'ready' || game.phase === 'running';
        const bobAmplitude = game.flightActive ? 2.6 : (player.grounded ? 1.2 : 1.8);
        const bob = canFloat
            ? Math.sin(game.elapsed * (game.flightActive ? 7.5 : 10)) * bobAmplitude
            : 0;
        const rotation = game.phase === 'done'
            ? -0.34
            : Math.max(-0.18, Math.min(0.18, player.vy / (game.height * 4)));
        ctx.save();
        ctx.globalAlpha = 0.38;
        ctx.fillStyle = '#05060b';
        ctx.beginPath();
        ctx.ellipse(
            player.x + player.size * 0.5,
            gameGroundY() + 5,
            player.size * (player.grounded ? 0.42 : 0.3),
            player.size * 0.11,
            0,
            0,
            Math.PI * 2
        );
        ctx.fill();
        ctx.restore();

        if (game.flightActive && game.phase === 'running') {
            ctx.save();
            const thrust = ctx.createLinearGradient(0, player.y + player.size * 0.62, 0, player.y + player.size * 1.28);
            thrust.addColorStop(0, 'rgba(112, 133, 255, .44)');
            thrust.addColorStop(1, 'rgba(112, 133, 255, 0)');
            ctx.fillStyle = thrust;
            ctx.beginPath();
            ctx.ellipse(
                player.x + player.size * 0.5,
                player.y + player.size * 0.9,
                player.size * 0.24,
                player.size * 0.42,
                0,
                0,
                Math.PI * 2
            );
            ctx.fill();
            ctx.restore();
        }

        if (game.phase === 'running') {
            for (let i = 2; i > 0; i -= 1) {
                ctx.save();
                ctx.globalAlpha = 0.07 * (3 - i);
                ctx.translate(-i * player.size * 0.25, 0);
                if (gameGhostImage.complete && gameGhostImage.naturalWidth) {
                    ctx.drawImage(gameGhostImage, player.x, player.y + bob, player.size, player.size);
                }
                ctx.restore();
            }
        }

        ctx.save();
        ctx.translate(player.x + player.size / 2, player.y + player.size / 2 + bob);
        ctx.rotate(rotation);
        ctx.shadowColor = game.phase === 'done' ? 'rgba(255, 103, 109, .7)' : 'rgba(86, 111, 255, .65)';
        ctx.shadowBlur = game.phase === 'done' ? 20 : 15;
        if (gameGhostImage.complete && gameGhostImage.naturalWidth) {
            ctx.drawImage(gameGhostImage, -player.size / 2, -player.size / 2, player.size, player.size);
        } else {
            ctx.fillStyle = '#4b7bee';
            gameRoundedRect(ctx, -player.size / 2, -player.size / 2, player.size, player.size, player.size * 0.42);
            ctx.fill();
        }
        ctx.restore();
    }

    function gameDraw() {
        const ctx = game.ctx;
        if (!ctx) return;
        ctx.setTransform(game.dpr, 0, 0, game.dpr, 0, 0);
        ctx.clearRect(0, 0, game.width, game.height);
        gameDrawBackground(ctx);
        game.entities.forEach((entity) => {
            if (entity.kind === 'orb') gameDrawOrb(ctx, entity);
            else gameDrawObstacle(ctx, entity);
        });
        game.particles.forEach((particle) => {
            ctx.save();
            ctx.globalAlpha = Math.min(1, particle.life * 3);
            ctx.fillStyle = '#ffd078';
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, 2.2, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        });
        gameDrawPlayer(ctx);
        if (game.hitFlash > 0) {
            ctx.fillStyle = 'rgba(213, 72, 79, .18)';
            ctx.fillRect(0, 0, game.width, game.height);
        }
    }

    function gameFrame(now) {
        if (game.phase !== 'running') {
            game.raf = null;
            gameDraw();
            return;
        }
        const dt = Math.min(0.05, Math.max(0, (now - game.lastFrameAt) / 1000));
        game.lastFrameAt = now;
        gameUpdate(dt);
        gameDraw();
        if (game.phase === 'running') game.raf = requestAnimationFrame(gameFrame);
        else game.raf = null;
    }

    function gameStart() {
        const restartingFromGameOver = game.phase === 'done';
        reportGameAbandon('restart');
        clearTimeout(game.readyTimer);
        cancelAnimationFrame(game.raf);
        gameResizeCanvas();
        game.score = 0;
        game.orbs = 0;
        game.distance = 0;
        game.scrollX = 0;
        game.elapsed = 0;
        game.entities = [];
        game.particles = [];
        game.hitFlash = 0;
        game.manualFlight = false;
        if (restartingFromGameOver) gameResetPlayer();
        gameSetFlight(gameSensingActive());
        gameSpawnCourse();
        game.nextSpawn = Math.max(game.width * 0.58, Math.max(190, game.width * 0.32) * 1.9);
        gameUpdateStats();
        gameSetPhase('ready', 'GET READY');
        gameStartMusic();
        gameStartMotionSound();
        gamePlaySound('start');
        gameStartPreview();
        gameMsg('The Spectre is taking off…');
        $('.js-game-canvas').focus({ preventScroll: true });
        gameDraw();
        track('game_start', { input_mode: connectionInputMode() });
        game.readyTimer = setTimeout(() => {
            if (game.phase !== 'ready') return;
            gameSetPhase('running', game.flightActive ? 'FLY' : 'GLIDE');
            gameStopPreview();
            gameMsg('Move for the high lane. Stay quiet for the low lane. Orbs add 100 points.');
            game.lastFrameAt = performance.now();
            game.raf = requestAnimationFrame(gameFrame);
        }, GAME_START_DELAY_MS);
    }

    function gameDemoFlight(active, event) {
        if (conn.mode !== 'demo' || !['idle', 'ready', 'running'].includes(game.phase)) return;
        if (event) event.preventDefault();
        game.manualFlight = active;
        demoInputEnergy = active ? 1 : 0;
        gameSetFlight(active);
        gameStartPreview();
    }

    function gameInit() {
        const canvas = $('.js-game-canvas');
        $('.js-game-start').addEventListener('click', gameStart);
        $('.js-game-fullscreen').addEventListener('click', gameToggleFullscreen);
        $('.js-game-sound').addEventListener('click', gameToggleSound);
        canvas.addEventListener('pointerdown', (event) => {
            if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId);
            gameDemoFlight(true, event);
        });
        canvas.addEventListener('pointerup', (event) => gameDemoFlight(false, event));
        canvas.addEventListener('pointercancel', (event) => gameDemoFlight(false, event));
        document.addEventListener('keydown', (event) => {
            if (route !== 'tool-game' || (event.key !== ' ' && event.key !== 'ArrowUp')) return;
            if (document.activeElement !== canvas) return;
            gameDemoFlight(true, event);
        });
        document.addEventListener('keyup', (event) => {
            if (route !== 'tool-game' || (event.key !== ' ' && event.key !== 'ArrowUp')) return;
            if (document.activeElement !== canvas) return;
            gameDemoFlight(false, event);
        });
        window.addEventListener('resize', gameResizeCanvas);
        document.addEventListener('fullscreenchange', gameOnFullscreenChange);
        document.addEventListener('webkitfullscreenchange', gameOnFullscreenChange);
        gameGhostImage.addEventListener('load', gameDraw);
        gameRenderSoundControl();
        gameSyncFullscreenButton();
        gameResetPlayer();
    }
