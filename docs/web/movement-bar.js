/*
 * Shared vertical movement bar + draggable threshold marker.
 * Used by The Game and Configure.
 *
 * With autoScale (default), the display range follows the threshold so the
 * marker stays near mid-height when idle. Drag freezes the scale; release
 * re-anchors it. This keeps small thresholds (e.g. 0.01) usable.
 */

(function () {
    'use strict';

    const DEFAULTS = Object.freeze({
        scaleMax: 10,
        thresholdMin: 0,
        thresholdMax: 10,
        digits: 1,
        autoScale: true,
        // Idle marker position as a fraction of bar height (0 = bottom, 1 = top).
        thresholdAnchor: 0.5,
        scaleFloor: 1e-4
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

            this.thresholdMin = Number.isFinite(options.thresholdMin) ? options.thresholdMin : DEFAULTS.thresholdMin;
            this.thresholdMax = Number.isFinite(options.thresholdMax) ? options.thresholdMax : DEFAULTS.thresholdMax;
            this.autoScale = options.autoScale !== false;
            this.thresholdAnchor = Number.isFinite(options.thresholdAnchor)
                ? options.thresholdAnchor
                : DEFAULTS.thresholdAnchor;
            this.scaleFloor = Number.isFinite(options.scaleFloor) && options.scaleFloor > 0
                ? options.scaleFloor
                : DEFAULTS.scaleFloor;
            // With autoScale, digits follow the zoomed range unless autoDigits is forced off.
            this.autoDigits = this.autoScale && options.autoDigits !== false;
            this.digits = Number.isFinite(options.digits) ? options.digits : DEFAULTS.digits;
            this.onThresholdCommit = options.onThresholdCommit || null;
            this.onThresholdChange = options.onThresholdChange || null;

            this.movement = 0;
            this.threshold = 1;
            this.scaleMax = Number(options.scaleMax) || DEFAULTS.scaleMax;
            this.dragging = false;
            this.interactive = options.interactive !== false;
            this._boundMove = (event) => this._handleDrag(event);
            this._boundEnd = () => this._stopDrag();

            this._bindDrag();
            this.setInteractive(this.interactive);
            this.setThreshold(this.threshold, { silent: true });
            this.setMovement(0);
        }

        _renderMovement() {
            if (!this.fill || !(this.scaleMax > 0)) return;
            const display = Math.max(0, Math.min(this.scaleMax, this.movement));
            this.fill.style.height = `${(display / this.scaleMax) * 100}%`;
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

        /**
         * Sets the hard threshold ceiling (and fixed scale when autoScale is off).
         * Configure uses this when switching classic (10) vs ML (1).
         */
        setScaleMax(scaleMax) {
            const parsed = Number(scaleMax);
            if (!Number.isFinite(parsed) || parsed <= 0) return;
            this.thresholdMax = parsed;
            if (!this.autoScale) {
                this.scaleMax = parsed;
            }
            this.setThreshold(this.threshold, { silent: true });
        }

        setMovement(value) {
            const parsed = Number(value);
            // Keep the raw reading so zoom changes do not permanently clip it.
            this.movement = Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
            this._renderMovement();
        }

        setThreshold(value, options = {}) {
            if (this.dragging && !options.force) return this.threshold;
            const parsed = Number(value);
            if (!Number.isFinite(parsed)) return this.threshold;
            this.threshold = Math.max(
                this.thresholdMin,
                Math.min(this.thresholdMax, parsed)
            );
            this._syncDisplayScale();
            this._renderThreshold();
            this._renderMovement();
            if (!options.silent && typeof this.onThresholdChange === 'function') {
                this.onThresholdChange(this.threshold);
            }
            return this.threshold;
        }

        _syncDisplayScale() {
            if (!this.autoScale || this.dragging) return;

            const anchor = Math.min(0.9, Math.max(0.1, this.thresholdAnchor));
            let desired;
            if (this.threshold <= 0) {
                desired = this.thresholdMax;
            } else {
                desired = this.threshold / anchor;
            }

            // Zoom in for small thresholds; never exceed the hard ceiling.
            this.scaleMax = Math.min(
                this.thresholdMax,
                Math.max(this.scaleFloor, desired, this.threshold)
            );
            this._updateDigitsFromScale();
        }

        _updateDigitsFromScale() {
            if (!this.autoDigits) return;
            // Aim for ~100 discrete steps across the visible scale.
            const stepTarget = this.scaleMax / 100;
            if (!(stepTarget > 0) || stepTarget >= 1) {
                this.digits = stepTarget >= 1 ? 0 : DEFAULTS.digits;
                return;
            }
            this.digits = Math.min(6, Math.max(1, Math.ceil(-Math.log10(stepTarget))));
        }

        _renderThreshold() {
            if (this.marker) {
                const percentage = this.scaleMax > 0
                    ? (this.threshold / this.scaleMax) * 100
                    : 0;
                this.marker.style.bottom = `${Math.max(0, Math.min(100, percentage))}%`;
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

            // Re-anchor scale so the new threshold returns near mid-height.
            this._syncDisplayScale();
            this._renderThreshold();
            this._renderMovement();

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
