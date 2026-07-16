/*
 * Shared vertical movement bar + draggable threshold marker.
 * Used by The Game and Configure.
 */

(function () {
    'use strict';

    const DEFAULTS = Object.freeze({
        scaleMax: 10,
        thresholdMin: 0,
        thresholdMax: 10,
        digits: 1
    });

    class ESPectreMovementBar {
        constructor(options = {}) {
            this.root = typeof options.root === 'string'
                ? document.querySelector(options.root)
                : (options.root || document.getElementById('movement-bar-vertical'));
            if (!this.root) {
                throw new Error('ESPectreMovementBar: root element not found');
            }

            this.fill = this.root.querySelector('[data-role="fill"], .movement-bar-fill, #movement-fill');
            this.marker = this.root.querySelector('[data-role="marker"], .threshold-marker, #threshold-marker');
            this.valueEl = this.root.querySelector('[data-role="value"], .threshold-value, #threshold-value');
            this.track = this.marker ? this.marker.parentElement : this.root.querySelector('.movement-bar-track');

            this.scaleMax = Number(options.scaleMax) || DEFAULTS.scaleMax;
            this.thresholdMin = Number.isFinite(options.thresholdMin) ? options.thresholdMin : DEFAULTS.thresholdMin;
            this.thresholdMax = Number.isFinite(options.thresholdMax) ? options.thresholdMax : DEFAULTS.thresholdMax;
            this.digits = Number.isFinite(options.digits) ? options.digits : DEFAULTS.digits;
            this.onThresholdCommit = options.onThresholdCommit || null;
            this.onThresholdChange = options.onThresholdChange || null;

            this.movement = 0;
            this.threshold = 1;
            this.dragging = false;
            this.interactive = options.interactive !== false;
            this._boundMove = (event) => this._handleDrag(event);
            this._boundEnd = () => this._stopDrag();

            this._bindDrag();
            this.setInteractive(this.interactive);
            this.setThreshold(this.threshold, { silent: true });
            this.setMovement(0);
        }

        get isDragging() {
            return this.dragging;
        }

        getThreshold() {
            return this.threshold;
        }

        setVisible(visible) {
            this.root.hidden = !visible;
            this.root.classList.toggle('is-hidden', !visible);
        }

        setInteractive(interactive) {
            this.interactive = Boolean(interactive);
            this.root.classList.toggle('is-readonly', !this.interactive);
            if (this.marker) {
                this.marker.style.pointerEvents = this.interactive ? 'auto' : 'none';
                this.marker.title = this.interactive
                    ? 'Drag to adjust motion threshold on device'
                    : 'Threshold (read-only)';
            }
            if (this.valueEl) {
                this.valueEl.style.pointerEvents = this.interactive ? 'auto' : 'none';
                this.valueEl.style.cursor = this.interactive ? 'ns-resize' : 'default';
            }
        }

        setScaleMax(scaleMax) {
            const parsed = Number(scaleMax);
            if (!Number.isFinite(parsed) || parsed <= 0) return;
            this.scaleMax = parsed;
            this.thresholdMax = parsed;
            this.setThreshold(this.threshold, { silent: true });
            this.setMovement(this.movement);
        }

        setMovement(value) {
            const parsed = Number(value);
            this.movement = Number.isFinite(parsed)
                ? Math.max(0, Math.min(this.scaleMax, parsed))
                : 0;
            if (this.fill) {
                this.fill.style.height = `${(this.movement / this.scaleMax) * 100}%`;
            }
        }

        setThreshold(value, options = {}) {
            if (this.dragging && !options.force) return this.threshold;
            const parsed = Number(value);
            if (!Number.isFinite(parsed)) return this.threshold;
            this.threshold = Math.max(
                this.thresholdMin,
                Math.min(this.thresholdMax, parsed)
            );
            this._renderThreshold();
            if (!options.silent && typeof this.onThresholdChange === 'function') {
                this.onThresholdChange(this.threshold);
            }
            return this.threshold;
        }

        _renderThreshold() {
            if (this.marker) {
                const percentage = (this.threshold / this.scaleMax) * 100;
                this.marker.style.bottom = `${percentage}%`;
            }
            if (this.valueEl) {
                this.valueEl.textContent = this.threshold.toFixed(this.digits);
            }
        }

        _bindDrag() {
            if (!this.marker) return;
            const start = (event) => this._startDrag(event);
            this.marker.addEventListener('mousedown', start);
            this.marker.addEventListener('touchstart', start, { passive: false });
            if (this.valueEl) {
                this.valueEl.addEventListener('mousedown', start);
                this.valueEl.addEventListener('touchstart', start, { passive: false });
            }
        }

        _startDrag(event) {
            if (!this.interactive || this.dragging) return;
            if (event.cancelable) event.preventDefault();
            this.dragging = true;
            this.marker.classList.add('dragging');
            document.addEventListener('mousemove', this._boundMove);
            document.addEventListener('mouseup', this._boundEnd);
            document.addEventListener('touchmove', this._boundMove, { passive: false });
            document.addEventListener('touchend', this._boundEnd);
            document.addEventListener('touchcancel', this._boundEnd);
        }

        _handleDrag(event) {
            if (!this.dragging || !this.track) return;
            if (event.cancelable) event.preventDefault();

            const rect = this.track.getBoundingClientRect();
            const clientY = event.touches ? event.touches[0].clientY : event.clientY;
            const relativeY = rect.bottom - clientY;
            const percentage = Math.max(0, Math.min(100, (relativeY / rect.height) * 100));
            const raw = (percentage / 100) * this.scaleMax;
            const step = Math.pow(10, -this.digits);
            const rounded = Math.round(raw / step) * step;
            this.threshold = Math.max(
                this.thresholdMin,
                Math.min(this.thresholdMax, rounded)
            );
            this._renderThreshold();
            if (typeof this.onThresholdChange === 'function') {
                this.onThresholdChange(this.threshold);
            }
        }

        async _stopDrag() {
            if (!this.dragging) return;
            this.dragging = false;
            if (this.marker) this.marker.classList.remove('dragging');
            document.removeEventListener('mousemove', this._boundMove);
            document.removeEventListener('mouseup', this._boundEnd);
            document.removeEventListener('touchmove', this._boundMove);
            document.removeEventListener('touchend', this._boundEnd);
            document.removeEventListener('touchcancel', this._boundEnd);

            if (typeof this.onThresholdCommit === 'function') {
                try {
                    await this.onThresholdCommit(this.threshold);
                } catch (error) {
                    console.warn('Failed to commit threshold:', error);
                }
            }
        }
    }

    window.ESPectreMovementBar = ESPectreMovementBar;
}());
