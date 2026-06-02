/*
 * ESPectre - The Game
 * 
 * A reaction game powered by ESPectre WiFi motion detection.
 * Stay still. Move fast. React to survive.
 * 
 * Communication: Web Bluetooth (desktop Chrome/Edge)
 * Protocol:
 *   - telemetry notify: [float32 movement, float32 threshold]
 *   - sysinfo notify: text "key=value" lines + "END"
 *   - control write: REQ_SYSINFO, SET_THRESHOLD:X.XX
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

// ==================== GAME STATE ====================

const GameState = {
    IDLE: 'idle',
    WAITING: 'waiting',      // Enemy spawning, player must stay still
    TRIGGER: 'trigger',      // MOVE! Player must move
    WIN: 'win',
    LOSE: 'lose',
    CHEATER: 'cheater',
    GAME_OVER: 'game_over'
};

// Enemy types with increasing difficulty
const EnemyTypes = [
    { name: 'WISP', maxTime: 800, points: 100, hp: 1, waves: [1, 2, 3] },
    { name: 'SHADE', maxTime: 600, points: 200, hp: 2, waves: [4, 5, 6] },
    { name: 'PHANTOM', maxTime: 450, points: 350, hp: 2, waves: [7, 8, 9] },
    { name: 'GLITCH', maxTime: 350, points: 500, hp: 3, waves: [10, 11, 12] },
    { name: 'VOID', maxTime: 250, points: 750, hp: 3, waves: [13, 14, 15] }
];

// Game progression constants
const MAX_WAVE = 15;
const SPECTRES_PER_WAVE = 3;
const MOVEMENT_SCALE_MAX = 10.0;
const THRESHOLD_MIN = 0.0;
const THRESHOLD_MAX = 10.0;

// Hit strength categories based on power (movement / threshold * 0.3)
const HitStrength = {
    NONE: { name: 'none', minPower: 0, damage: 0 },
    WEAK: { name: 'weak', minPower: 0.5, damage: 1 },
    NORMAL: { name: 'normal', minPower: 1.0, damage: 1 },
    STRONG: { name: 'strong', minPower: 2.0, damage: 2 },
    CRITICAL: { name: 'critical', minPower: 3.0, damage: 3 }
};

const BLE_ESPECTRE_SERVICE_UUID = 'd33ff46b-2203-4775-bc6f-b3a2c36af8f0';
const BLE_ESPECTRE_TELEMETRY_UUID = '119d5cac-48da-4bd9-bfc3-169805868258';
const BLE_ESPECTRE_SYSINFO_UUID = 'c8c89ffa-c401-461f-9ffc-942fa04adfe3';
const BLE_ESPECTRE_CONTROL_UUID = '33ed9214-a8d7-40e8-82d1-c82747dcdc71';

// ==================== GAME CLASS ====================

class ESPectreGame {
    constructor() {
        // Game state
        this.state = GameState.IDLE;
        this.inputMode = 'mouse'; // 'mouse', 'touch', or 'bluetooth'
        this.isTouching = false;  // Touch state for mobile
        
        // Web Bluetooth connection
        this.bleDevice = null;
        this.bleGattServer = null;
        this.bleService = null;
        this.bleTelemetryChar = null;
        this.bleSysinfoChar = null;
        this.bleControlChar = null;
        this.isDisconnecting = false;
        
        // Stats
        this.score = 0;
        this.streak = 0;
        this.maxStreak = 0;
        this.wave = 1;
        this.spectresDefeated = 0;
        this.bestTime = null;
        this.reactionTimes = [];
        this.lives = 3;
        this.maxLives = 3;
        
        // Current round
        this.currentEnemy = null;
        this.enemyCurrentHp = 0;
        this.triggerTime = null;
        this.movement = 0;
        this.threshold = 1.0;  // From ESP32 config
        this.lastHitStrength = null;
        this.inputDecayTimer = null;  // Decay timer for mouse/touch input
        
        // System info from ESP32
        this.systemInfo = {};
        
        // Game timers (for cleanup on disconnect/home)
        this.gameTimers = [];
        
        // Power stats
        this.totalPower = 0;
        this.hitCount = 0;
        this.criticalHits = 0;
        this.strongHits = 0;
        
        // Thresholds (relative to ESP32 threshold)
        this.cheatMultiplier = 1.0;  // 100% of threshold = cheating (moving before trigger)
        this.moveMultiplier = 1.2;   // 120% of threshold = valid move (slightly higher to avoid false positives)
        
        // DOM elements
        this.screens = {
            connect: document.getElementById('screen-connect'),
            game: document.getElementById('screen-game'),
            results: document.getElementById('screen-results')
        };
        
        this.elements = {
            btnConnect: document.getElementById('btn-connect'),
            btnMouse: document.getElementById('btn-mouse'),
            connectionStatus: document.getElementById('connection-status'),
            wave: document.getElementById('wave'),
            waveProgress: document.getElementById('wave-progress'),
            bestTime: document.getElementById('best-time'),
            score: document.getElementById('score'),
            enemy: document.getElementById('enemy'),
            enemyName: document.getElementById('enemy-name'),
            enemyHpBar: document.getElementById('enemy-hp-bar'),
            enemyHpFill: document.getElementById('enemy-hp-fill'),
            enemyHpText: document.getElementById('enemy-hp-text'),
            hitStrength: document.getElementById('hit-strength'),
            gameMessage: document.getElementById('game-message'),
            reactionTime: document.getElementById('reaction-time'),
            // Vertical movement bar (always visible)
            movementFill: document.getElementById('movement-fill'),
            // USB button in header
            btnUsb: document.getElementById('btn-usb'),
            // Life ghost display (in header, from components.js)
            lifeGhost: document.getElementById('life-ghost'),
            // USB ready elements
            connectionPre: document.getElementById('connection-pre'),
            connectionReady: document.getElementById('connection-ready'),
            dividerMouse: document.getElementById('divider-mouse'),
            mouseHint: document.getElementById('mouse-hint'),
            thresholdMarker: document.getElementById('threshold-marker'),
            thresholdValue: document.getElementById('threshold-value'),
            usbDeviceName: document.getElementById('usb-device-name'),
            btnStartGame: document.getElementById('btn-start-game'),
            btnDisconnect: document.getElementById('btn-disconnect'),
            btnPlayAgain: document.getElementById('btn-play-again'),
            btnShare: document.getElementById('btn-share'),
            btnMute: document.getElementById('btn-mute'),
            // System info elements
            systemInfo: document.getElementById('system-info'),
            infoThreshold: document.getElementById('info-threshold'),
            infoWindow: document.getElementById('info-window'),
            infoSubcarriers: document.getElementById('info-subcarriers'),
            infoLowpass: document.getElementById('info-lowpass'),
            infoHampel: document.getElementById('info-hampel'),
            infoTraffic: document.getElementById('info-traffic'),
            // Connection box and mobile hint
            connectionBox: document.getElementById('connection-box'),
            mobileUsbHint: document.getElementById('mobile-usb-hint')
        };
        
        // Audio system
        this.audioContext = null;
        this.audioEnabled = false;  // Start muted, user enables with button
        this.audioInitialized = false;
        
        this.init();
    }
    
    init() {
        // Enable scrolling on initial load (connection screen should be scrollable on mobile)
        this.enableTouchGestures();
        
        // Event listeners
        this.elements.btnConnect.addEventListener('click', () => this.connect());
        this.elements.btnMouse.addEventListener('click', () => this.startMouseMode());
        
        // Touch mode button (only if element exists)
        this.elements.btnTouch = document.getElementById('btn-touch');
        if (this.elements.btnTouch) {
            this.elements.btnTouch.addEventListener('click', () => this.startTouchMode());
        }
        this.elements.btnStartGame.addEventListener('click', () => this.startGame());
        this.elements.btnDisconnect.addEventListener('click', () => this.disconnect());
        this.elements.btnPlayAgain.addEventListener('click', () => this.restart());
        this.elements.btnShare.addEventListener('click', () => this.share());
        
        // Mute button
        if (this.elements.btnMute) {
            this.elements.btnMute.addEventListener('click', () => this.toggleMute());
        }
        
        // USB button (connect/disconnect)
        if (this.elements.btnUsb) {
            this.elements.btnUsb.addEventListener('click', () => this.toggleUsb());
        }
        
        // Threshold marker drag
        this.initThresholdMarker();
        
        // Mouse input (for keyboard/mouse fallback mode)
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.lastMouseTime = 0;
        document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        
        // Touch input (for mobile)
        this.lastTouchX = 0;
        this.lastTouchY = 0;
        this.lastTouchTime = 0;
        this.touchDetected = false;  // Flag to detect touch capability at runtime
        document.addEventListener('touchstart', (e) => {
            // Dynamically detect touch and show swipe button if needed
            if (!this.touchDetected) {
                this.touchDetected = true;
                this.checkBrowserSupport();  // Re-run to show correct buttons
            }
            this.handleTouchStart(e);
        }, { passive: true });
        document.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: true });
        document.addEventListener('touchend', (e) => this.handleTouchEnd(e), { passive: true });
        document.addEventListener('touchcancel', (e) => this.handleTouchEnd(e), { passive: true });
        
        // Escape key to go home
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Escape' && this.state !== GameState.IDLE) {
                this.goHome();
            }
        });
          
        // Check browser support
        this.checkBrowserSupport();
        
        // Cosmic background
        // Cosmic background is loaded from ../cosmic-bg.js
    }
    
    checkBrowserSupport() {
        // Show touch button on mobile, mouse button on desktop
        // Also check touchDetected flag for runtime detection (e.g., Chrome DevTools device mode)
        const isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0 || this.touchDetected;
        const hasBluetooth = 'bluetooth' in navigator;
        
        // Update header controls (hide button if not supported or on mobile)
        this.updateHeaderControls(hasBluetooth, isMobile);
        
        if (isMobile) {
            // Mobile: hide USB connection box entirely, show swipe as primary
            if (this.elements.connectionBox) {
                this.elements.connectionBox.classList.add('mobile-hidden');
            }
            if (this.elements.dividerMouse) this.elements.dividerMouse.style.display = 'none';
            if (this.elements.btnMouse) this.elements.btnMouse.style.display = 'none';
            if (this.elements.mouseHint) this.elements.mouseHint.style.display = 'none';
            if (this.elements.btnTouch) this.elements.btnTouch.style.display = '';
            // Show mobile hint about USB option on desktop
            if (this.elements.mobileUsbHint) this.elements.mobileUsbHint.style.display = '';
        } else {
            // Desktop: show USB connection box
            if (this.elements.connectionBox) {
                this.elements.connectionBox.classList.remove('mobile-hidden');
            }
            if (!hasBluetooth) {
                this.showConnectionStatus(
                    'Web Bluetooth not supported. Use Chrome or Edge desktop.',
                    'error'
                );
                this.elements.btnConnect.disabled = true;
            }
            if (this.elements.btnMouse) this.elements.btnMouse.style.display = '';
            if (this.elements.mouseHint) this.elements.mouseHint.style.display = '';
            if (this.elements.btnTouch) this.elements.btnTouch.style.display = 'none';
            if (this.elements.mobileUsbHint) this.elements.mobileUsbHint.style.display = 'none';
        }
    }
    
    updateHeaderControls(hasBluetooth, isMobile) {
        // Hide device button if Web Bluetooth not supported or on mobile
        if (this.elements.btnUsb) {
            this.elements.btnUsb.style.display = (hasBluetooth && !isMobile) ? '' : 'none';
        }
    }
    
    // ==================== THRESHOLD MARKER ====================
    
    initThresholdMarker() {
        const marker = this.elements.thresholdMarker;
        const valueEl = this.elements.thresholdValue;
        if (!marker) return;
        
        this.isDraggingThreshold = false;
        this.thresholdMin = THRESHOLD_MIN;
        this.thresholdMax = THRESHOLD_MAX;
        
        // Update marker position based on current threshold
        this.updateThresholdMarkerPosition();
        
        // Mouse events on marker
        marker.addEventListener('mousedown', (e) => this.startThresholdDrag(e));
        document.addEventListener('mousemove', (e) => this.handleThresholdDrag(e));
        document.addEventListener('mouseup', () => this.stopThresholdDrag());
        
        // Touch events on marker
        marker.addEventListener('touchstart', (e) => this.startThresholdDrag(e), { passive: false });
        document.addEventListener('touchmove', (e) => this.handleThresholdDrag(e), { passive: false });
        document.addEventListener('touchend', () => this.stopThresholdDrag());
        
        // Also allow dragging from the value label
        if (valueEl) {
            valueEl.addEventListener('mousedown', (e) => this.startThresholdDrag(e));
            valueEl.addEventListener('touchstart', (e) => this.startThresholdDrag(e), { passive: false });
        }
    }
    
    startThresholdDrag(e) {
        if (e.cancelable) e.preventDefault();
        this.isDraggingThreshold = true;
        this.elements.thresholdMarker.classList.add('dragging');
    }
    
    handleThresholdDrag(e) {
        if (!this.isDraggingThreshold) return;
        if (e.cancelable) e.preventDefault();
        
        const track = this.elements.thresholdMarker.parentElement;
        const rect = track.getBoundingClientRect();
        
        // Get Y position (touch or mouse)
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        
        // Calculate position as percentage from bottom (0% = bottom, 100% = top)
        const relativeY = rect.bottom - clientY;
        const percentage = Math.max(0, Math.min(100, (relativeY / rect.height) * 100));
        
        // Map percentage to threshold value (0-100% → 0-10, same scale as movement bar)
        // Clamp to valid range (thresholdMin to thresholdMax)
        const newThreshold = (percentage / 100) * this.thresholdMax;
        this.threshold = Math.max(this.thresholdMin, Math.min(this.thresholdMax, Math.round(newThreshold * 10) / 10));
        
        // Update UI while dragging
        this.updateThresholdMarkerPosition();
    }
    
    stopThresholdDrag() {
        if (!this.isDraggingThreshold) return;
        this.isDraggingThreshold = false;
        this.elements.thresholdMarker.classList.remove('dragging');
        this.sendThresholdToDevice();
    }
    
    updateThresholdMarkerPosition() {
        const marker = this.elements.thresholdMarker;
        const valueEl = this.elements.thresholdValue;
        if (!marker) return;
        
        // Map threshold to percentage (0-100) using same scale as movement bar (0-10).
        const percentage = (this.threshold / this.thresholdMax) * 100;
        marker.style.bottom = `${percentage}%`;
        
        if (valueEl) {
            valueEl.textContent = this.threshold.toFixed(1);
        }
        
        // Also update system info threshold display
        if (this.elements.infoThreshold) {
            this.elements.infoThreshold.textContent = this.threshold.toFixed(2);
        }
    }

    async sendThresholdToDevice() {
        try {
            await this.sendBleControlCommand(`SET_THRESHOLD:${this.threshold.toFixed(2)}`);
        } catch (e) {
            console.warn('Failed to send threshold command:', e);
        }
    }

    // ==================== SCREENS ====================
    
    showScreen(screenName) {
        Object.keys(this.screens).forEach(name => {
            this.screens[name].classList.remove('active');
        });
        this.screens[screenName].classList.add('active');
    }
    
    // ==================== BLUETOOTH CONNECTION ====================
    
    async connect() {
        try {
            await this.connectBluetooth();
        } catch (e) {
            console.error('Bluetooth connection failed:', e);
            let errorType = 'unknown';
            if (e.message.includes('No device selected')) {
                this.showConnectionStatus(
                    'No ESPectre device found. Make sure your ESP32 is flashed with ESPectre firmware.',
                    'error'
                );
                errorType = 'no_device_selected';
            } else {
                this.showConnectionStatus('Connection failed: ' + e.message, 'error');
                errorType = e.message.substring(0, 50);
            }
            trackEvent('ble_connect_fail', { error: errorType });
        }
    }
    
    async connectBluetooth() {
        this.showConnectionStatus('Requesting Bluetooth device...', 'connecting');

        this.bleDevice = await navigator.bluetooth.requestDevice({
            filters: [{ services: [BLE_ESPECTRE_SERVICE_UUID] }]
        });
        this.bleDevice.addEventListener('gattserverdisconnected', () => this.handleBluetoothDisconnect());

        this.bleGattServer = await this.bleDevice.gatt.connect();
        this.bleService = await this.bleGattServer.getPrimaryService(BLE_ESPECTRE_SERVICE_UUID);
        this.bleTelemetryChar = await this.bleService.getCharacteristic(BLE_ESPECTRE_TELEMETRY_UUID);
        this.bleSysinfoChar = await this.bleService.getCharacteristic(BLE_ESPECTRE_SYSINFO_UUID);
        this.bleControlChar = await this.bleService.getCharacteristic(BLE_ESPECTRE_CONTROL_UUID);

        await this.bleTelemetryChar.startNotifications();
        this.bleTelemetryChar.addEventListener('characteristicvaluechanged', (event) =>
            this.handleTelemetryNotification(event)
        );
        await this.bleSysinfoChar.startNotifications();
        this.bleSysinfoChar.addEventListener('characteristicvaluechanged', (event) =>
            this.handleSysinfoNotification(event)
        );

        this.inputMode = 'bluetooth';
        this.showConnectionStatus('Connected via Bluetooth!', 'connected');
        trackEvent('ble_connect');
        await this.sendBleControlCommand('REQ_SYSINFO');
        this.showUsbReady('ESP32 (Bluetooth)');
    }
    
    showUsbReady(deviceName) {
        // Switch connection box from pre-connection to ready state
        if (this.elements.connectionPre) {
            this.elements.connectionPre.classList.add('hidden');
        }
        if (this.elements.connectionReady) {
            this.elements.connectionReady.classList.remove('hidden');
        }
        if (this.elements.usbDeviceName) {
            this.elements.usbDeviceName.textContent = deviceName;
        }
        // Update USB button in header
        this.updateUsbButton();
        
        // Hide mouse mode section
        if (this.elements.dividerMouse) {
            this.elements.dividerMouse.style.display = 'none';
        }
        if (this.elements.btnMouse) {
            this.elements.btnMouse.style.display = 'none';
        }
        if (this.elements.mouseHint) {
            this.elements.mouseHint.style.display = 'none';
        }
    }
    
    handleTelemetryNotification(event) {
        const dv = event.target.value;
        if (!dv || dv.byteLength < 8) return;
        const movement = dv.getFloat32(0, true);
        const threshold = dv.getFloat32(4, true);
        if (!Number.isFinite(movement) || !Number.isFinite(threshold)) return;
        if (!this.isDraggingThreshold) {
            this.threshold = threshold;
            this.updateThresholdMarkerPosition();
        }
        this.updateInput(movement);
    }
    
    handleSysinfoNotification(event) {
        const value = event.target.value;
        if (!value) return;
        const line = new TextDecoder().decode(value.buffer);
        if (!line) return;
        const trimmed = line.trim();
        if (trimmed === 'END') {
            this.handleSystemInfo('END', '');
            return;
        }
        const equalIdx = trimmed.indexOf('=');
        if (equalIdx <= 0) return;
        const key = trimmed.slice(0, equalIdx).trim();
        const data = trimmed.slice(equalIdx + 1).trim();
        this.handleSystemInfo(key, data);
    }

    async sendBleControlCommand(command) {
        if (!this.bleControlChar) return;
        const payload = new TextEncoder().encode(command);
        if (typeof this.bleControlChar.writeValueWithResponse === 'function') {
            await this.bleControlChar.writeValueWithResponse(payload);
            return;
        }
        if (typeof this.bleControlChar.writeValueWithoutResponse === 'function') {
            await this.bleControlChar.writeValueWithoutResponse(payload);
            return;
        }
        // Legacy fallback for older browser implementations.
        await this.bleControlChar.writeValue(payload);
    }

    async handleBluetoothDisconnect() {
        if (this.bleDevice && this.bleDevice.gatt && this.bleDevice.gatt.connected) {
            return;
        }
        await this.disconnect();
    }
    
    handleSystemInfo(key, value) {
        this.systemInfo[key] = value;
        
        // Update UI based on key
        switch (key) {
            case 'chip':
                // Update device name in ready status (uppercase chip name)
                if (this.elements.usbDeviceName) {
                    this.elements.usbDeviceName.textContent = value.toUpperCase() + ' Connected';
                }
                // Track device type
                trackEvent('ble_device_info', { chip: value.toUpperCase() });
                break;
            case 'threshold':
                if (this.elements.infoThreshold) {
                    this.elements.infoThreshold.textContent = value;
                }
                // Update local threshold and marker position
                const thresholdValue = parseFloat(value);
                if (!isNaN(thresholdValue)) {
                    this.threshold = thresholdValue;
                    this.updateThresholdMarkerPosition();
                }
                break;
            case 'window':
                if (this.elements.infoWindow) {
                    this.elements.infoWindow.textContent = value + ' pkts';
                }
                break;
            case 'subcarriers':
                if (this.elements.infoSubcarriers) {
                    this.elements.infoSubcarriers.textContent = value.toUpperCase();
                }
                break;
            case 'lowpass':
                if (this.elements.infoLowpass) {
                    this.elements.infoLowpass.textContent = value.toUpperCase();
                }
                break;
            case 'lowpass_cutoff':
                if (this.elements.infoLowpass) {
                    this.elements.infoLowpass.textContent = value + ' Hz';
                }
                break;
            case 'hampel':
                if (this.elements.infoHampel) {
                    this.elements.infoHampel.textContent = value.toUpperCase();
                }
                break;
            case 'hampel_window':
                // Append to existing hampel info
                if (this.elements.infoHampel && this.systemInfo['hampel'] === 'on') {
                    this.elements.infoHampel.textContent = value + ' pkts';
                }
                break;
            case 'traffic_rate':
                if (this.elements.infoTraffic) {
                    this.elements.infoTraffic.textContent = value + ' pps';
                }
                break;
            case 'END':
                // Show the system info panel
                if (this.elements.systemInfo) {
                    this.elements.systemInfo.classList.remove('hidden');
                }
                break;
        }
    }
    
    showConnectionStatus(message, type) {
        this.elements.connectionStatus.textContent = message;
        this.elements.connectionStatus.className = 'connection-status ' + type;
    }
    
    // ==================== INPUT ====================
    
    startMouseMode() {
        this.inputMode = 'mouse';
        this.startGame();
    }
    
    startTouchMode() {
        this.inputMode = 'touch';
        
        // Reset touch tracking
        this.lastTouchX = 0;
        this.lastTouchY = 0;
        this.lastTouchTime = 0;
        
        this.startGame();
    }
    
    disableTouchGestures() {
        // No-op: scroll control removed for simplicity
    }
    
    enableTouchGestures() {
        // Reset any stuck drag state that could block scrolling
        this.isDraggingThreshold = false;
        if (this.elements.thresholdMarker) {
            this.elements.thresholdMarker.classList.remove('dragging');
        }
    }
    
    handleMouseMove(e) {
        // Ignore mouse input when device stream is connected
        if (this.inputMode === 'bluetooth') return;
        
        const now = performance.now();
        const dt = now - this.lastMouseTime;
        
        if (dt > 0 && this.lastMouseTime > 0) {
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const velocity = (distance / dt) * 0.3;
            const movement = Math.min(velocity, MOVEMENT_SCALE_MAX);
            
            this.updateInput(movement);
            
            if (this.inputDecayTimer) clearTimeout(this.inputDecayTimer);
            this.inputDecayTimer = setTimeout(() => this.updateInput(0), 50);
        }
        
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.lastMouseTime = now;
    }
    
    handleTouchStart(e) {
        if (e.touches.length > 0) {
            this.lastTouchX = e.touches[0].clientX;
            this.lastTouchY = e.touches[0].clientY;
            this.lastTouchTime = performance.now();
            this.isTouching = true;
        }
    }
    
    handleTouchMove(e) {
        // Ignore touch input when device stream is connected
        if (this.inputMode === 'bluetooth') return;
        
        if (e.touches.length === 0) return;
        
        const now = performance.now();
        const dt = now - this.lastTouchTime;
        
        if (dt > 0 && this.lastTouchTime > 0) {
            const dx = e.touches[0].clientX - this.lastTouchX;
            const dy = e.touches[0].clientY - this.lastTouchY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const velocity = (distance / dt) * 0.3;
            const movement = Math.min(velocity, MOVEMENT_SCALE_MAX);
            
            this.updateInput(movement);
            
            if (this.inputDecayTimer) clearTimeout(this.inputDecayTimer);
            this.inputDecayTimer = setTimeout(() => this.updateInput(0), 50);
        }
        
        this.lastTouchX = e.touches[0].clientX;
        this.lastTouchY = e.touches[0].clientY;
        this.lastTouchTime = now;
    }
    
    handleTouchEnd(e) {
        // Ignore touch input when device stream is connected
        if (this.inputMode === 'bluetooth') return;
        
        this.isTouching = false;
        this.updateInput(0);
        if (this.inputDecayTimer) {
            clearTimeout(this.inputDecayTimer);
            this.inputDecayTimer = null;
        }
    }
    
    /**
     * Unified input handler - updates movement bar and audio for all sources
     */
    updateInput(value) {
        this.movement = Math.max(0, Math.min(MOVEMENT_SCALE_MAX, value));
        
        // Update vertical movement bar (height-based)
        const percent = (this.movement / MOVEMENT_SCALE_MAX) * 100;
        if (this.elements.movementFill) {
            this.elements.movementFill.style.height = percent + '%';
        }
        
        // Update audio
        this.updateAudioMovement(this.movement);
        
        // Check game logic if in active game state
        if (this.state === GameState.WAITING || this.state === GameState.TRIGGER) {
            this.checkMovement();
        }
    }
    
    checkMovement() {
        const cheatThreshold = this.threshold * this.cheatMultiplier;
        const moveThreshold = this.threshold * this.moveMultiplier;
        
        if (this.state === GameState.WAITING && this.movement > cheatThreshold) {
            this.handleCheater();
        } else if (this.state === GameState.TRIGGER && this.movement > moveThreshold) {
            this.handleReaction();
        }
    }
    
    // ==================== GAME LOGIC ====================
    
    startGame() {
        // Block scroll during gameplay (for all input modes)
        this.disableTouchGestures();
        
        // Reset stats
        this.score = 0;
        this.streak = 0;
        this.maxStreak = 0;
        this.wave = 1;
        this.spectresDefeated = 0;
        this.bestTime = null;
        this.reactionTimes = [];
        this.lives = this.maxLives;
        
        // Reset power stats
        this.totalPower = 0;
        this.hitCount = 0;
        this.criticalHits = 0;
        this.strongHits = 0;
        
        // Track game start
        trackEvent('game_start', { input_mode: this.inputMode });
        
        // Play game start sound
        this.playSound('gamestart');
        
        this.updateHUD();
        this.showScreen('game');
        
        // Reset mouse tracking
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.lastMouseTime = 0;
        
        // Start first round
        this.startRound();
    }
    
    startRound() {
        this.state = GameState.IDLE;
        this.movement = 0;
        this.lastHitStrength = null;
        
        // Hide enemy and message
        this.elements.enemy.classList.remove('visible', 'attacking', 'dissolved', 'dissolved-strong', 'dissolved-critical', 'corrupt');
        this.elements.reactionTime.classList.add('hidden');
        
        // Hide HP bar and hit strength
        if (this.elements.enemyHpBar) {
            this.elements.enemyHpBar.classList.add('hidden');
        }
        if (this.elements.hitStrength) {
            this.elements.hitStrength.classList.add('hidden');
        }
        
        // Show "GET READY"
        this.showMessage('GET READY...', '');
        
        // Spawn enemy after short delay
        setTimeout(() => this.spawnEnemy(), 1000);
    }
    
    spawnEnemy() {
        // Pick enemy based on wave
        this.currentEnemy = this.getEnemyForWave(this.wave);
        this.enemyCurrentHp = this.currentEnemy.hp;
        
        // Update enemy display (but keep hidden until MOVE)
        this.elements.enemyName.textContent = this.currentEnemy.name;
        // Enemy stays hidden during WAITING phase
        
        // Update HP bar (hidden for now)
        this.updateEnemyHpBar();
        
        // Hide hit strength indicator
        if (this.elements.hitStrength) {
            this.elements.hitStrength.classList.add('hidden');
        }
        
        // Enter waiting state
        this.state = GameState.WAITING;
        this.showMessage('STAY STILL...', '');
        
        // Random delay before trigger (2-5 seconds)
        const delay = 2000 + Math.random() * 3000;
        
        const timerId = setTimeout(() => {
            if (this.state === GameState.WAITING) {
                this.trigger();
            }
        }, delay);
        this.gameTimers.push(timerId);
    }
    
    updateEnemyHpBar() {
        if (!this.elements.enemyHpBar || !this.currentEnemy) return;
        
        const maxHp = this.currentEnemy.hp;
        const currentHp = this.enemyCurrentHp;
        const percent = (currentHp / maxHp) * 100;
        
        // Show HP bar only for multi-HP enemies
        if (maxHp > 1) {
            this.elements.enemyHpBar.classList.remove('hidden');
            this.elements.enemyHpFill.style.width = percent + '%';
            this.elements.enemyHpText.textContent = currentHp + '/' + maxHp;
            
            // Color based on HP
            this.elements.enemyHpFill.classList.remove('low', 'critical');
            if (percent <= 33) {
                this.elements.enemyHpFill.classList.add('critical');
            } else if (percent <= 66) {
                this.elements.enemyHpFill.classList.add('low');
            }
        } else {
            this.elements.enemyHpBar.classList.add('hidden');
        }
    }
    
    getHitStrength(power) {
        if (power >= HitStrength.CRITICAL.minPower) return HitStrength.CRITICAL;
        if (power >= HitStrength.STRONG.minPower) return HitStrength.STRONG;
        if (power >= HitStrength.NORMAL.minPower) return HitStrength.NORMAL;
        if (power >= HitStrength.WEAK.minPower) return HitStrength.WEAK;
        return HitStrength.NONE;
    }
    
    getEnemyForWave(wave) {
        // Find appropriate enemy for this wave
        for (let i = EnemyTypes.length - 1; i >= 0; i--) {
            if (wave >= EnemyTypes[i].waves[0]) {
                return { ...EnemyTypes[i] };
            }
        }
        return { ...EnemyTypes[0] };
    }
    
    trigger() {
        this.state = GameState.TRIGGER;
        this.triggerTime = performance.now();
        
        // Show enemy and trigger visual/audio feedback
        this.elements.enemy.classList.add('visible', 'attacking');
        this.showMessage('MOVE!', 'move');
        this.flashScreen('move');
        this.playSound('move');
        this.playSound('spawn');
        
        // Timeout - player too slow
        const timerId = setTimeout(() => {
            if (this.state === GameState.TRIGGER) {
                this.handleTooSlow();
            }
        }, this.currentEnemy.maxTime);
        this.gameTimers.push(timerId);
    }
    
    handleReaction() {
        const reactionTime = Math.round(performance.now() - this.triggerTime);
        this.reactionTimes.push(reactionTime);
        
        if (reactionTime <= this.currentEnemy.maxTime) {
            // Calculate power: movement / (threshold * moveMultiplier)
            const power = this.movement / (this.threshold * this.moveMultiplier);
            const hitStrength = this.getHitStrength(power);
            this.lastHitStrength = hitStrength;
            
            // Track power stats
            this.totalPower += power;
            this.hitCount++;
            if (hitStrength === HitStrength.CRITICAL) this.criticalHits++;
            if (hitStrength === HitStrength.STRONG) this.strongHits++;
            
            // Calculate damage (power >= HP = one-shot)
            let damage = hitStrength.damage;
            if (power >= this.enemyCurrentHp) {
                damage = this.enemyCurrentHp; // One-shot!
            }
            
            // Apply damage
            this.enemyCurrentHp = Math.max(0, this.enemyCurrentHp - damage);
            this.updateEnemyHpBar();
            
            // Show hit strength indicator
            this.showHitStrength(hitStrength, power);
            
            if (this.enemyCurrentHp <= 0) {
                // Enemy defeated!
                this.handleWin(reactionTime, power, hitStrength);
            } else {
                // Enemy survives, need another hit
                this.handlePartialHit(reactionTime, damage, hitStrength);
            }
        } else {
            this.handleTooSlow();
        }
    }
    
    showHitStrength(hitStrength, power) {
        if (!this.elements.hitStrength) return;
        
        const el = this.elements.hitStrength;
        el.textContent = hitStrength.name.toUpperCase() + '!';
        el.className = 'hit-strength ' + hitStrength.name;
        el.classList.remove('hidden');
        
        // Add power value for strong/critical
        if (hitStrength === HitStrength.STRONG || hitStrength === HitStrength.CRITICAL) {
            el.textContent += ' (' + power.toFixed(1) + 'x)';
        }
    }
    
    handlePartialHit(reactionTime, damage, hitStrength) {
        // Show feedback but don't end the round
        this.showMessage('HIT! -' + damage + ' HP', 'partial-hit');
        this.showReactionTime(reactionTime);
        
        // Flash and sound based on hit strength
        this.flashScreen(hitStrength.name);
        this.playSound(hitStrength.name);
        
        // Brief shake for strong/critical
        if (hitStrength === HitStrength.STRONG || hitStrength === HitStrength.CRITICAL) {
            document.querySelector('.game-arena').classList.add('shake');
            setTimeout(() => {
                document.querySelector('.game-arena').classList.remove('shake');
            }, 200);
        }
        
        // Continue the round - trigger again after delay
        this.state = GameState.WAITING;
        this.showMessage('AGAIN!', '');
        
        // Shorter delay for multi-hit rounds (1-2.5 seconds)
        const delay = 1000 + Math.random() * 1500;
        
        const timerId = setTimeout(() => {
            if (this.state === GameState.WAITING) {
                this.trigger();
            }
        }, delay);
        this.gameTimers.push(timerId);
    }
    
    handleWin(reactionTime, power, hitStrength) {
        this.state = GameState.WIN;
        
        // Update stats
        this.streak++;
        this.maxStreak = Math.max(this.maxStreak, this.streak);
        this.spectresDefeated++;
        
        // Calculate score with power bonus
        const timeBonus = Math.max(0, this.currentEnemy.maxTime - reactionTime);
        const streakMultiplier = 1 + (this.streak - 1) * 0.1;
        
        // Power bonus: extra points based on hit strength
        let powerBonus = 0;
        if (hitStrength === HitStrength.NORMAL) powerBonus = 50;
        if (hitStrength === HitStrength.STRONG) powerBonus = 150;
        if (hitStrength === HitStrength.CRITICAL) powerBonus = 300;
        
        const points = Math.round((this.currentEnemy.points + timeBonus + powerBonus) * streakMultiplier);
        this.score += points;
        
        // Update best time
        if (!this.bestTime || reactionTime < this.bestTime) {
            this.bestTime = reactionTime;
        }
        
        // Visual feedback based on hit strength
        let message = 'DISSOLVED!';
        let messageClass = 'win';
        
        if (hitStrength === HitStrength.CRITICAL) {
            message = 'OBLITERATED!';
            messageClass = 'win critical';
        } else if (hitStrength === HitStrength.STRONG) {
            message = 'SHATTERED!';
            messageClass = 'win strong';
        }
        
        this.showMessage(message, messageClass);
        this.elements.enemy.classList.remove('attacking');
        
        // Different dissolve animation based on strength
        if (hitStrength === HitStrength.CRITICAL) {
            this.elements.enemy.classList.add('dissolved-critical');
        } else if (hitStrength === HitStrength.STRONG) {
            this.elements.enemy.classList.add('dissolved-strong');
        } else {
            this.elements.enemy.classList.add('dissolved');
        }
        
        this.showReactionTime(reactionTime);
        
        // Stronger flash for powerful hits
        this.flashScreen(hitStrength.name);
        
        // Play hit sound based on strength
        this.playSound(hitStrength.name);
        
        // Screen shake for strong/critical
        if (hitStrength === HitStrength.STRONG || hitStrength === HitStrength.CRITICAL) {
            document.querySelector('.game-arena').classList.add('shake');
            setTimeout(() => {
                document.querySelector('.game-arena').classList.remove('shake');
            }, 300);
        }
        
        this.updateHUD();
        
        // Progress wave every 3 wins
        if (this.spectresDefeated % 3 === 0) {
            this.wave++;
        }
        
        // Next round
        const timerId = setTimeout(() => this.startRound(), 1500);
        this.gameTimers.push(timerId);
    }
    
    handleTooSlow() {
        this.state = GameState.LOSE;
        
        // Visual and audio feedback
        this.showMessage('TOO SLOW!', 'lose');
        this.elements.enemy.classList.remove('attacking');
        this.elements.enemy.classList.add('corrupt');
        this.flashScreen('lose');
        this.playSound('lose');
        
        // Lose a life
        this.loseLife();
        
        if (this.lives <= 0) {
            // Game over
            const timerId = setTimeout(() => this.gameOver(), 1500);
            this.gameTimers.push(timerId);
        } else {
            // Continue from current wave
            const timerId = setTimeout(() => this.continueFromWave(), 1500);
            this.gameTimers.push(timerId);
        }
    }
    
    handleCheater() {
        this.state = GameState.CHEATER;
        
        // Visual and audio feedback
        this.showMessage('CHEATER!', 'cheater');
        this.elements.enemy.classList.add('corrupt');
        this.flashScreen('lose');
        this.playSound('cheater');
        document.querySelector('.game-arena').classList.add('shake');
        
        setTimeout(() => {
            document.querySelector('.game-arena').classList.remove('shake');
        }, 300);
        
        // Lose a life
        this.loseLife();
        
        if (this.lives <= 0) {
            // Game over
            const timerId = setTimeout(() => this.gameOver(), 1500);
            this.gameTimers.push(timerId);
        } else {
            // Continue from current wave
            const timerId = setTimeout(() => this.continueFromWave(), 1500);
            this.gameTimers.push(timerId);
        }
    }
    
    continueFromWave() {
        // Reset streak but keep score, wave, and spectresDefeated
        this.streak = 0;
        this.updateHUD();
        
        // Clear enemy state
        this.elements.enemy.classList.remove('attacking', 'corrupt');
        
        // Start a new round at the current wave
        this.startRound();
    }
    
    async gameOver() {
        this.state = GameState.GAME_OVER;
        
        // Play game over sound
        this.playSound('gameover');
        
        // Re-enable touch gestures for scrolling on results screen
        this.enableTouchGestures();
        
        // Keep Bluetooth telemetry active to show live movement on game over screen.
        
        // Update results
        document.getElementById('result-wave').textContent = this.wave;
        document.getElementById('result-spectres').textContent = this.spectresDefeated;
        document.getElementById('result-best').textContent = this.bestTime ? this.bestTime + 'ms' : '---';
        document.getElementById('result-streak').textContent = this.maxStreak;
        document.getElementById('result-score').textContent = this.score;
        
        // Update power stats
        const avgPower = this.hitCount > 0 ? (this.totalPower / this.hitCount) : 0;
        document.getElementById('result-avg-power').textContent = avgPower.toFixed(1) + 'x';
        document.getElementById('result-critical').textContent = this.criticalHits;
        
        // Track game over with all stats
        trackEvent('game_over', {
            input_mode: this.inputMode,
            score: this.score,
            wave: this.wave,
            spectres: this.spectresDefeated,
            max_streak: this.maxStreak,
            best_time: this.bestTime || 0,
            critical_hits: this.criticalHits,
            avg_power: avgPower.toFixed(1)
        });
        
        this.showScreen('results');
    }
    
    // ==================== UI HELPERS ====================
    
    showMessage(text, className) {
        const messageEl = this.elements.gameMessage;
        messageEl.querySelector('.message-text').textContent = text;
        messageEl.className = 'game-message ' + className;
    }
    
    showReactionTime(ms) {
        const el = this.elements.reactionTime;
        el.querySelector('.reaction-value').textContent = ms;
        el.classList.remove('hidden');
    }
    
    flashScreen(type) {
        const flash = document.createElement('div');
        flash.className = 'screen-flash ' + type;
        document.body.appendChild(flash);
        
        setTimeout(() => flash.remove(), 200);
    }
    
    updateHUD() {
        // Wave shows current/max
        this.elements.wave.textContent = this.wave + '/' + MAX_WAVE;
        // Wave progress shows spectres defeated in current wave
        const waveProgress = this.spectresDefeated % SPECTRES_PER_WAVE;
        if (this.elements.waveProgress) {
            this.elements.waveProgress.textContent = waveProgress + '/' + SPECTRES_PER_WAVE;
        }
        this.elements.bestTime.textContent = this.bestTime ? this.bestTime + 'ms' : '---';
        this.elements.score.textContent = this.score;
        this.updateLivesDisplay();
    }
    
    updateLivesDisplay() {
        if (this.elements.lifeGhost) {
            this.elements.lifeGhost.dataset.lives = this.lives;
        }
    }
    
    loseLife() {
        if (this.lives > 0) {
            this.lives--;
            
            // Trigger hit animation
            if (this.elements.lifeGhost) {
                this.elements.lifeGhost.classList.add('losing');
                setTimeout(() => {
                    this.elements.lifeGhost.classList.remove('losing');
                }, 600);
            }
            
            this.updateLivesDisplay();
        }
    }
    
    // ==================== ACTIONS ====================
    
    clearGameTimers() {
        // Cancel all pending game timers
        for (const timerId of this.gameTimers) {
            clearTimeout(timerId);
        }
        this.gameTimers = [];
    }
    
    resetGameVisuals() {
        // Reset enemy visual state
        if (this.elements.enemy) {
            this.elements.enemy.classList.remove('visible', 'attacking', 'dissolved', 'dissolved-strong', 'dissolved-critical', 'corrupt');
        }
        
        // Reset life ghost to home state (no data-lives = white)
        if (this.elements.lifeGhost) {
            this.elements.lifeGhost.removeAttribute('data-lives');
            this.elements.lifeGhost.classList.remove('losing');
        }
        
        // Hide HP bar
        if (this.elements.enemyHpBar) {
            this.elements.enemyHpBar.classList.add('hidden');
        }
        
        // Hide hit strength indicator
        if (this.elements.hitStrength) {
            this.elements.hitStrength.classList.add('hidden');
        }
        
        // Hide reaction time
        if (this.elements.reactionTime) {
            this.elements.reactionTime.classList.add('hidden');
        }
    }
    
    async restart() {
        this.startGame();
    }
    
    share() {
        const avgPower = this.hitCount > 0 ? (this.totalPower / this.hitCount).toFixed(1) : '0.0';
        const text = `ESPectre - The Game\n\n` +
            `Score: ${this.score}\n` +
            `Wave: ${this.wave}/${MAX_WAVE}\n` +
            `Spectres Dissolved: ${this.spectresDefeated}\n` +
            `Best Reaction: ${this.bestTime ? this.bestTime + 'ms' : '---'}\n` +
            `Avg Power: ${avgPower}x\n\n` +
            `Play at https://espectre.dev/game`;
        
        // Track share action
        trackEvent('share', {
            score: this.score,
            wave: this.wave,
            method: navigator.share ? 'native' : 'clipboard'
        });
        
        if (navigator.share) {
            navigator.share({ text });
        } else {
            navigator.clipboard.writeText(text).then(() => {
                this.elements.btnShare.innerHTML = '<i class="fas fa-check"></i> Copied!';
                setTimeout(() => {
                    this.elements.btnShare.innerHTML = '<i class="fas fa-share"></i> Share';
                }, 2000);
            });
        }
    }
    
    async goHome() {
        // Cancel any pending game timers
        this.clearGameTimers();
        
        // Reset all game visual elements
        this.resetGameVisuals();
        
        // Go back to connect screen but keep device connection active
        this.state = GameState.IDLE;
        this.showScreen('connect');
        
        // If still connected via Bluetooth, show the ready state
        if (this.bleGattServer && this.bleGattServer.connected && this.inputMode === 'bluetooth') {
            // Keep connection-ready visible, movement bar stays visible
            return;
        }
        
        // Mouse/touch mode: reset to pre-connection state
        this.showConnectionStatus('', '');
        
        // Reset connection box to pre-connection state
        if (this.elements.connectionPre) {
            this.elements.connectionPre.classList.remove('hidden');
        }
        if (this.elements.connectionReady) {
            this.elements.connectionReady.classList.add('hidden');
        }
        // Update USB button in header
        this.updateUsbButton();
        
        // Show mouse/touch mode section again (based on device type)
        if (this.elements.dividerMouse) {
            this.elements.dividerMouse.style.display = '';
        }
        
        // Re-run device detection to show correct buttons
        this.checkBrowserSupport();
        
        // Reset input mode and restore gestures
        this.inputMode = 'mouse';
        this.isTouching = false;
        this.enableTouchGestures();
    }
    
    async disconnect() {
        if (this.isDisconnecting) return;
        this.isDisconnecting = true;

        // Cancel any pending game timers
        this.clearGameTimers();
        
        // Reset all game visual elements
        this.resetGameVisuals();
        
        if (this.bleTelemetryChar) {
            try {
                await this.bleTelemetryChar.stopNotifications();
            } catch (e) {}
            this.bleTelemetryChar = null;
        }
        if (this.bleSysinfoChar) {
            try {
                await this.bleSysinfoChar.stopNotifications();
            } catch (e) {}
            this.bleSysinfoChar = null;
        }
        this.bleControlChar = null;
        this.bleService = null;

        if (this.bleGattServer && this.bleGattServer.connected) {
            try {
                this.bleGattServer.disconnect();
            } catch (e) {}
        }
        this.bleGattServer = null;
        this.bleDevice = null;
        
        this.state = GameState.IDLE;
        this.inputMode = 'mouse';
        this.showScreen('connect');
        this.showConnectionStatus('Disconnected', '');
        
        // Reset connection box to pre-connection state
        if (this.elements.connectionPre) {
            this.elements.connectionPre.classList.remove('hidden');
        }
        if (this.elements.connectionReady) {
            this.elements.connectionReady.classList.add('hidden');
        }
        // Update USB button in header
        this.updateUsbButton();

        // Hide system info panel and clear data
        if (this.elements.systemInfo) {
            this.elements.systemInfo.classList.add('hidden');
        }
        this.systemInfo = {};
        
        // Show mouse mode section again
        if (this.elements.dividerMouse) {
            this.elements.dividerMouse.style.display = '';
        }
        if (this.elements.btnMouse) {
            this.elements.btnMouse.style.display = '';
        }
        if (this.elements.mouseHint) {
            this.elements.mouseHint.style.display = '';
        }
        this.isDisconnecting = false;
    }
    
    // ==================== AUDIO SYSTEM ====================
    
    initAudio() {
        if (this.audioInitialized) return;
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioInitialized = true;
            
            // Resume AudioContext if suspended (browser autoplay policy)
            if (this.audioContext.state === 'suspended') {
                this.audioContext.resume();
            }
            
            // Initialize ambient system
            this.initAmbientAudio();
        } catch (e) {
            console.warn('Web Audio not available:', e);
            this.audioEnabled = false;
        }
    }
    
    // ==================== AMBIENT AUDIO SYSTEM ====================
    
    initAmbientAudio() {
        if (!this.audioContext) return;
        
        const ctx = this.audioContext;
        
        // Create master gain for ambient sounds
        this.ambientMasterGain = ctx.createGain();
        this.ambientMasterGain.gain.value = 0;
        this.ambientMasterGain.connect(ctx.destination);
        
        // Create a low-pass filter for ethereal effect
        this.ambientFilter = ctx.createBiquadFilter();
        this.ambientFilter.type = 'lowpass';
        this.ambientFilter.frequency.value = 2000;
        this.ambientFilter.Q.value = 1;
        this.ambientFilter.connect(this.ambientMasterGain);
        
        // Ethereal oscillator (reacts to movement only)
        this.etherealOsc = ctx.createOscillator();
        this.etherealGain = ctx.createGain();
        this.etherealOsc.type = 'sine';
        this.etherealOsc.frequency.value = 220;
        this.etherealGain.gain.value = 0; // Starts silent, only sounds on movement
        this.etherealOsc.connect(this.etherealGain);
        this.etherealGain.connect(this.ambientFilter);
        this.etherealOsc.start();
        
        // LFO for vibrato on movement sound
        this.lfo = ctx.createOscillator();
        this.lfoGain = ctx.createGain();
        this.lfo.type = 'sine';
        this.lfo.frequency.value = 5; // Vibrato speed
        this.lfoGain.gain.value = 8; // Vibrato depth in Hz
        this.lfo.connect(this.lfoGain);
        this.lfoGain.connect(this.etherealOsc.frequency);
        this.lfo.start();
        
        // Ambient state
        this.ambientActive = false;
        this.ambientTargetVolume = 0.4; // Volume when there's movement
        this.lastAmbientUpdate = 0;
        
        console.log('Ambient audio initialized (movement-reactive only)');
    }
    
    startAmbient() {
        if (!this.ambientMasterGain) {
            console.warn('Ambient not initialized');
            return;
        }
        
        if (this.ambientActive) {
            console.log('Ambient already active');
            return;
        }
        
        // Resume audio context if needed
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume().then(() => {
                console.log('AudioContext resumed');
            });
        }
        
        this.ambientActive = true;
        const ctx = this.audioContext;
        const now = ctx.currentTime;
        
        console.log('Starting ambient audio, volume:', this.ambientTargetVolume);
        
        // Fade in ambient (faster: 1 second)
        this.ambientMasterGain.gain.cancelScheduledValues(now);
        this.ambientMasterGain.gain.setValueAtTime(0.001, now);
        this.ambientMasterGain.gain.exponentialRampToValueAtTime(this.ambientTargetVolume, now + 1);
    }
    
    stopAmbient() {
        if (!this.ambientMasterGain || !this.ambientActive) return;
        
        this.ambientActive = false;
        const ctx = this.audioContext;
        const now = ctx.currentTime;
        
        // Fade out ambient
        this.ambientMasterGain.gain.cancelScheduledValues(now);
        this.ambientMasterGain.gain.setValueAtTime(this.ambientMasterGain.gain.value, now);
        this.ambientMasterGain.gain.linearRampToValueAtTime(0, now + 1);
    }
    
    /**
     * Update audio based on movement input (mouse/touch/bluetooth).
     * This is called for all input sources and provides theremin-like audio feedback.
     * Initializes and starts ambient audio on first movement.
     */
    updateAudioMovement(movement) {
        // Only process if audio is enabled and initialized
        if (!this.audioEnabled || !this.audioInitialized) return;
        
        // Update the ambient sound
        this.updateAmbient(movement);
    }
    
    updateAmbient(movement) {
        if (!this.ambientActive || !this.audioContext || !this.audioEnabled) return;
        if (!this.etherealOsc || !this.etherealGain) return;
        
        const ctx = this.audioContext;
        const now = ctx.currentTime;
        
        // Throttle updates (60fps max)
        if (now - this.lastAmbientUpdate < 0.016) return;
        this.lastAmbientUpdate = now;
        
        // Normalize movement (0-1 range, capped at 2x threshold)
        const normalizedMovement = Math.min(movement / (this.threshold * 2), 1);
        
        // Only produce sound when there's movement
        // Volume scales with movement intensity
        const targetVolume = normalizedMovement * this.ambientTargetVolume;
        this.etherealGain.gain.setTargetAtTime(targetVolume, now, 0.05);
        
        // Frequency rises with movement: 220Hz (A3) to 880Hz (A5)
        const targetFreq = 220 + normalizedMovement * 660;
        this.etherealOsc.frequency.setTargetAtTime(targetFreq, now, 0.08);
        
        // Vibrato speed increases with movement intensity
        const vibratoSpeed = 3 + normalizedMovement * 8;
        this.lfo.frequency.setTargetAtTime(vibratoSpeed, now, 0.1);
        
        // Filter brightness based on movement intensity (brighter = more movement)
        const filterFreq = 1000 + normalizedMovement * 3000;
        this.ambientFilter.frequency.setTargetAtTime(filterFreq, now, 0.15);
    }
    
    toggleMute() {
        // Initialize audio on first unmute (user gesture required by browsers)
        if (!this.audioInitialized) {
            this.initAudio();
        }
        
        this.audioEnabled = !this.audioEnabled;
        
        // Update button appearance
        if (this.elements.btnMute) {
            const icon = this.elements.btnMute.querySelector('i');
            if (icon) {
                icon.className = this.audioEnabled ? 'fas fa-volume-up' : 'fas fa-volume-mute';
            }
            this.elements.btnMute.classList.toggle('unmuted', this.audioEnabled);
            this.elements.btnMute.title = this.audioEnabled ? 'Mute sound' : 'Enable sound';
        }
        
        // Start ambient if enabling for first time
        if (this.audioEnabled && !this.ambientActive) {
            this.startAmbient();
        }
        
        // Mute/unmute ambient
        if (this.ambientMasterGain && this.audioContext) {
            const now = this.audioContext.currentTime;
            this.ambientMasterGain.gain.setTargetAtTime(
                this.audioEnabled && this.ambientActive ? this.ambientTargetVolume : 0,
                now,
                0.1
            );
        }
    }
    
    toggleUsb() {
        // If connected, disconnect; otherwise connect
        if (this.bleGattServer && this.bleGattServer.connected) {
            this.disconnect();
        } else {
            this.connect();
        }
    }
    
    updateUsbButton() {
        if (this.elements.btnUsb) {
            const isConnected = !!(this.bleGattServer && this.bleGattServer.connected);
            this.elements.btnUsb.classList.toggle('connected', isConnected);
            this.elements.btnUsb.title = isConnected ? 'Disconnect Bluetooth' : 'Connect Bluetooth';
        }
    }
    
    playSound(type) {
        if (!this.audioEnabled || !this.audioContext) return;
        
        // Resume audio context if suspended (browser autoplay policy)
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
        
        switch (type) {
            case 'move':
                // Sharp, attention-grabbing alert tone
                this.playTone(880, 0.1, 'square', 0.3); // High A
                setTimeout(() => this.playTone(1320, 0.15, 'square', 0.25), 50); // High E
                break;
                
            case 'weak':
                // Soft blip
                this.playTone(440, 0.1, 'sine', 0.15);
                break;
                
            case 'normal':
                // Satisfying hit
                this.playTone(523, 0.08, 'triangle', 0.2); // C5
                this.playTone(659, 0.12, 'triangle', 0.15); // E5
                break;
                
            case 'strong':
                // Powerful impact
                this.playTone(330, 0.05, 'sawtooth', 0.3); // Bass hit
                setTimeout(() => this.playTone(659, 0.1, 'triangle', 0.25), 30);
                setTimeout(() => this.playTone(880, 0.15, 'triangle', 0.2), 60);
                break;
                
            case 'critical':
                // Epic critical hit - rising arpeggio
                this.playTone(262, 0.05, 'sawtooth', 0.35); // C4
                setTimeout(() => this.playTone(330, 0.05, 'sawtooth', 0.3), 40);
                setTimeout(() => this.playTone(392, 0.05, 'triangle', 0.3), 80);
                setTimeout(() => this.playTone(523, 0.08, 'triangle', 0.25), 120);
                setTimeout(() => this.playTone(659, 0.1, 'triangle', 0.2), 160);
                setTimeout(() => this.playTone(784, 0.15, 'sine', 0.2), 200);
                break;
                
            case 'lose':
                // Descending sad tone
                this.playTone(440, 0.15, 'sine', 0.2);
                setTimeout(() => this.playTone(392, 0.15, 'sine', 0.18), 150);
                setTimeout(() => this.playTone(330, 0.2, 'sine', 0.15), 300);
                break;
                
            case 'cheater':
                // Harsh buzzer
                this.playTone(150, 0.3, 'sawtooth', 0.25);
                this.playTone(155, 0.3, 'sawtooth', 0.25);
                break;
                
            case 'spawn':
                // Ethereal whoosh
                this.playTone(200, 0.3, 'sine', 0.1);
                setTimeout(() => this.playTone(300, 0.2, 'sine', 0.08), 100);
                break;
                
            case 'gameover':
                // Dramatic descending minor chord sequence
                this.playTone(392, 0.4, 'sine', 0.25);      // G4
                this.playTone(466, 0.4, 'sine', 0.2);       // Bb4 (minor third)
                setTimeout(() => {
                    this.playTone(330, 0.4, 'sine', 0.25);  // E4
                    this.playTone(392, 0.4, 'sine', 0.2);   // G4
                }, 300);
                setTimeout(() => {
                    this.playTone(262, 0.5, 'sine', 0.25);  // C4
                    this.playTone(311, 0.5, 'sine', 0.2);   // Eb4 (minor)
                }, 600);
                setTimeout(() => {
                    this.playTone(196, 0.8, 'sine', 0.3);   // G3 - final low note
                    this.playTone(233, 0.8, 'sine', 0.15);  // Bb3
                }, 900);
                break;
                
            case 'gamestart':
                // Energetic ascending fanfare
                this.playTone(262, 0.1, 'triangle', 0.25);  // C4
                setTimeout(() => this.playTone(330, 0.1, 'triangle', 0.25), 80);   // E4
                setTimeout(() => this.playTone(392, 0.1, 'triangle', 0.25), 160);  // G4
                setTimeout(() => this.playTone(523, 0.2, 'triangle', 0.3), 240);   // C5 - hold
                setTimeout(() => {
                    // Final power chord
                    this.playTone(523, 0.3, 'sawtooth', 0.2);  // C5
                    this.playTone(659, 0.3, 'triangle', 0.15); // E5
                    this.playTone(784, 0.3, 'sine', 0.1);      // G5
                }, 400);
                break;
        }
    }
    
    playTone(frequency, duration, type = 'sine', volume = 0.2) {
        if (!this.audioContext) return;
        
        const ctx = this.audioContext;
        const oscillator = ctx.createOscillator();
        const gainNode = ctx.createGain();
        
        oscillator.type = type;
        oscillator.frequency.setValueAtTime(frequency, ctx.currentTime);
        
        gainNode.gain.setValueAtTime(0, ctx.currentTime);
        gainNode.gain.linearRampToValueAtTime(volume, ctx.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
        
        oscillator.connect(gainNode);
        gainNode.connect(ctx.destination);
        
        oscillator.start(ctx.currentTime);
        oscillator.stop(ctx.currentTime + duration);
    }
    
}

// ==================== INIT ====================

document.addEventListener('DOMContentLoaded', () => {
    window.game = new ESPectreGame();
});
