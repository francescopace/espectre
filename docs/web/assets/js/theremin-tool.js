/*
 * ESPectre - Theremin tool
 *
 * Part of the website application shell.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

'use strict';

    /* ============================================================ theremin */

    const theremin = { ctx: null, osc: null, gain: null, raf: null, smoothed: 0, lastAt: 0 };

    function thereminStart() {
        if (theremin.ctx) return;
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
            toast('Web Audio is not available in this browser.');
            return;
        }
        theremin.ctx = new AudioCtx();
        theremin.osc = theremin.ctx.createOscillator();
        theremin.gain = theremin.ctx.createGain();
        theremin.osc.type = document.getElementById('th-wave').value;
        theremin.osc.frequency.value = 140;
        theremin.gain.gain.value = 0;
        theremin.osc.connect(theremin.gain).connect(theremin.ctx.destination);
        theremin.osc.start();
        theremin.lastAt = 0;
        $('.js-th-toggle').textContent = '⏹ Stop sound';
        const loop = () => {
            const nowMs = performance.now();
            const dt = theremin.lastAt ? Math.min(0.08, (nowMs - theremin.lastAt) / 1000) : 1 / 60;
            theremin.lastAt = nowMs;
            const f = energyFraction();
            const tau = evaluationIntervalMs() / 2000;
            const alpha = 1 - Math.exp(-dt / tau);
            theremin.smoothed += (f - theremin.smoothed) * alpha;
            const freq = 140 * Math.pow(2, theremin.smoothed * 2.6);
            const now = theremin.ctx.currentTime;
            theremin.osc.frequency.setTargetAtTime(freq, now, 0.05);
            theremin.gain.gain.setTargetAtTime(0.02 + theremin.smoothed * 0.35, now, 0.08);
            $('.js-th-freq').textContent = Math.round(freq);
            theremin.raf = requestAnimationFrame(loop);
        };
        loop();
    }

    function thereminStop() {
        if (!theremin.ctx) return;
        cancelAnimationFrame(theremin.raf);
        theremin.osc.stop();
        theremin.ctx.close();
        theremin.ctx = null;
        theremin.osc = null;
        theremin.gain = null;
        theremin.smoothed = 0;
        theremin.lastAt = 0;
        const toggle = $('.js-th-toggle');
        if (toggle) toggle.textContent = '▶ Start sound';
        const freq = $('.js-th-freq');
        if (freq) freq.textContent = '—';
    }

    function thereminInit() {
        $('.js-th-toggle').addEventListener('click', () => {
            const starting = !theremin.ctx;
            starting ? thereminStart() : thereminStop();
            track('theremin_configuration', {
                control: 'playback', setting_value: starting ? 'start' : 'stop'
            });
        });
        document.getElementById('th-wave').addEventListener('change', (event) => {
            if (theremin.osc) theremin.osc.type = event.target.value;
            track('theremin_configuration', {
                control: 'waveform', setting_value: event.target.value
            });
        });
    }
