/*
 * ESPectre - The Game
 * 
 * A reaction game powered by ESPectre WiFi motion detection.
 * Stay still. Move fast. React to survive.
 * 
 * Communication: USB Serial (all ESP32 variants)
 * Protocol: "G<movement>,<threshold>\n" from ESP32
 *           "START\n" / "STOP\n" / "T:X.XX\n" from browser
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

// Hit strength categories based on power (movement / threshold * 0.3)
const HitStrength = {
    NONE: { name: 'none', minPower: 0, damage: 0 },
    WEAK: { name: 'weak', minPower: 0.5, damage: 1 },
    NORMAL: { name: 'normal', minPower: 1.0, damage: 1 },
    STRONG: { name: 'strong', minPower: 2.0, damage: 2 },
    CRITICAL: { name: 'critical', minPower: 3.0, damage: 3 }
};

// USB VID/PID for ESP32 variants
const ESP32_USB_FILTERS = [
    // Espressif USB JTAG/Serial (ESP32-C3, C5, C6)
    { usbVendorId: 0x303A, usbProductId: 0x1001 },
    // ESP32-S2/S3 CDC
    { usbVendorId: 0x303A, usbProductId: 0x0002 },
    // CP210x (common USB-UART bridge)
    { usbVendorId: 0x10C4, usbProductId: 0xEA60 },
    // CH340 (common USB-UART bridge)
    { usbVendorId: 0x1A86, usbProductId: 0x7523 },
    // FTDI (common USB-UART bridge)
    { usbVendorId: 0x0403, usbProductId: 0x6001 }
];

// ==================== GAME CLASS ====================

class ESPectreGame {
    constructor() {
        // Game state
        this.state = GameState.IDLE;
        this.inputMode = 'mouse'; // 'mouse' or 'serial'
        
        // USB Serial connection
        this.serialPort = null;
        this.serialReader = null;
        this.serialWriter = null;
        this.readBuffer = '';
        this.readableStreamClosed = null;
        this.writableStreamClosed = null;
        this.pingInterval = null;  // Keep-alive ping timer
        
        // Stats
        this.score = 0;
        this.streak = 0;
        this.maxStreak = 0;
        this.wave = 1;
        this.spectresDefeated = 0;
        this.bestTime = null;
        this.reactionTimes = [];
        
        // Current round
        this.currentEnemy = null;
        this.enemyCurrentHp = 0;
        this.triggerTime = null;
        this.movement = 0;
        this.threshold = 1.0;  // From ESP32 config
        this.lastHitStrength = null;
        this.isDraggingThreshold = false;  // Block threshold updates during slider drag
        
        // System info from ESP32
        this.systemInfo = {};
        
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
            inputMode: document.getElementById('input-mode'),
            streak: document.getElementById('streak'),
            wave: document.getElementById('wave'),
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
            // Global movement bar (single instance for all screens)
            globalMovementBar: document.getElementById('global-movement-bar'),
            movementFill: document.getElementById('movement-fill'),
            movementThresholdMarker: document.getElementById('movement-threshold-marker'),
            movementValue: document.getElementById('movement-value'),
            thresholdValue: document.getElementById('threshold-value'),
            // USB ready elements
            connectionPre: document.getElementById('connection-pre'),
            connectionReady: document.getElementById('connection-ready'),
            dividerMouse: document.getElementById('divider-mouse'),
            mouseHint: document.getElementById('mouse-hint'),
            usbDeviceName: document.getElementById('usb-device-name'),
            btnStartGame: document.getElementById('btn-start-game'),
            btnDisconnect: document.getElementById('btn-disconnect'),
            btnPlayAgain: document.getElementById('btn-play-again'),
            btnShare: document.getElementById('btn-share'),
            btnHome: document.getElementById('btn-home'),
            btnMute: document.getElementById('btn-mute'),
            // System info elements
            systemInfo: document.getElementById('system-info'),
            infoThreshold: document.getElementById('info-threshold'),
            infoWindow: document.getElementById('info-window'),
            infoSubcarriers: document.getElementById('info-subcarriers'),
            infoLowpass: document.getElementById('info-lowpass'),
            infoHampel: document.getElementById('info-hampel'),
            infoTraffic: document.getElementById('info-traffic')
        };
        
        // Audio system
        this.audioContext = null;
        this.audioEnabled = true;
        this.audioInitialized = false;
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.elements.btnConnect.addEventListener('click', () => this.connect());
        this.elements.btnMouse.addEventListener('click', () => this.startMouseMode());
        this.elements.btnStartGame.addEventListener('click', () => this.startGame());
        this.elements.btnDisconnect.addEventListener('click', () => this.disconnect());
        this.elements.btnPlayAgain.addEventListener('click', () => this.restart());
        this.elements.btnShare.addEventListener('click', () => this.share());
        this.elements.btnHome.addEventListener('click', () => this.goHome());
        
        // Mute button
        if (this.elements.btnMute) {
            this.elements.btnMute.addEventListener('click', () => this.toggleMute());
        }
        
        // Mouse input (for keyboard/mouse fallback mode)
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        this.lastMouseTime = 0;
        this.mouseVelocityDecay = null;
        document.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        
        // Escape key to go home
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Escape' && this.state !== GameState.IDLE) {
                this.goHome();
            }
        });
        
        // Stop streaming when page is closed/navigated away
        // Using both beforeunload and pagehide for maximum browser compatibility
        window.addEventListener('beforeunload', () => this.stopStreamingSync());
        window.addEventListener('pagehide', () => this.stopStreamingSync());
        
        // Threshold slider drag
        this.initThresholdSlider();
        
        // Check browser support
        this.checkBrowserSupport();
        
        // Cosmic background
        // Cosmic background is loaded from ../cosmic-bg.js
    }
    
    checkBrowserSupport() {
        if (!('serial' in navigator)) {
            this.showConnectionStatus(
                'Web Serial not supported. Use Chrome or Edge.',
                'error'
            );
            this.elements.btnConnect.disabled = true;
        }
    }
    
    initThresholdSlider() {
        const marker = this.elements.movementThresholdMarker;
        if (!marker) return;
        
        let isDragging = false;
        
        const startDrag = (e) => {
            // Only allow dragging when connected (not during active game phases)
            if (this.state !== GameState.IDLE) return;
            
            isDragging = true;
            this.isDraggingThreshold = true;  // Block incoming threshold updates
            marker.classList.add('dragging');
            e.preventDefault();
        };
        
        const doDrag = (e) => {
            if (!isDragging) return;
            
            const bar = marker.parentElement;
            const rect = bar.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            
            // Calculate position as percentage (clamp to 2-98% for visual margin)
            let percent = ((clientX - rect.left) / rect.width) * 100;
            percent = Math.max(2, Math.min(98, percent));
            
            // Convert percent to threshold value: 0% = 0.5, 100% = 5.0
            const minThreshold = 0.5;
            const maxThreshold = 5.0;
            const newThreshold = minThreshold + (percent / 100) * (maxThreshold - minThreshold);
            
            this.threshold = newThreshold;
            
            // Directly update marker position during drag (don't rely on updateMovement)
            marker.style.left = percent + '%';
            
            // Update threshold value displays (movement bar and system info panel)
            if (this.elements.thresholdValue) {
                this.elements.thresholdValue.textContent = this.threshold.toFixed(2);
            }
            if (this.elements.infoThreshold) {
                this.elements.infoThreshold.textContent = this.threshold.toFixed(2);
            }
        };
        
        const endDrag = async () => {
            if (!isDragging) return;
            isDragging = false;
            marker.classList.remove('dragging');
            
            // Send threshold to ESP32 (format: T:X.XX)
            if (this.inputMode === 'serial') {
                await this.sendSerialCommand('T:' + this.threshold.toFixed(2));
            }
            
            // Re-enable threshold updates after a short delay
            // (allows the ESP32 to acknowledge the new threshold)
            setTimeout(() => {
                this.isDraggingThreshold = false;
            }, 500);
        };
        
        // Mouse events
        marker.addEventListener('mousedown', startDrag);
        document.addEventListener('mousemove', doDrag);
        document.addEventListener('mouseup', endDrag);
        
        // Touch events
        marker.addEventListener('touchstart', startDrag, { passive: false });
        document.addEventListener('touchmove', doDrag, { passive: false });
        document.addEventListener('touchend', endDrag);
    }
    
    // ==================== SCREENS ====================
    
    showScreen(screenName) {
        Object.keys(this.screens).forEach(name => {
            this.screens[name].classList.remove('active');
        });
        this.screens[screenName].classList.add('active');
    }
    
    // ==================== USB CONNECTION ====================
    
    async connect() {
        try {
            await this.connectSerial();
        } catch (e) {
            console.error('Serial connection failed:', e);
            if (e.message.includes('No port selected')) {
                this.showConnectionStatus('No device selected', 'error');
            } else if (e.message.includes('already open') || e.message.includes('in use')) {
                this.showConnectionStatus(
                    'Port in use! Close esphome logs or other serial monitors.',
                    'error'
                );
            } else {
                this.showConnectionStatus('Connection failed: ' + e.message, 'error');
            }
        }
    }
    
    async connectSerial() {
        this.showConnectionStatus('Requesting serial port...', 'connecting');
        
        this.serialPort = await navigator.serial.requestPort({
            filters: ESP32_USB_FILTERS
        });
        
        await this.serialPort.open({ baudRate: 115200 });
        
        this.inputMode = 'serial';
        this.showConnectionStatus('Connected via USB Serial!', 'connected');
        
        // Setup reader/writer with TransformStreams for text encoding/decoding
        const textDecoder = new TextDecoderStream();
        this.readableStreamClosed = this.serialPort.readable.pipeTo(textDecoder.writable);
        this.serialReader = textDecoder.readable.getReader();
        
        const textEncoder = new TextEncoderStream();
        this.writableStreamClosed = textEncoder.readable.pipeTo(this.serialPort.writable);
        this.serialWriter = textEncoder.writable.getWriter();
        
        // Start reading
        this.readSerialLoop();
        
        // Send START command to begin receiving data
        await this.sendSerialCommand('START');
        
        // Start ping keep-alive (every 2 seconds)
        this.startPing();
        
        // Show USB ready screen with live movement preview
        this.showUsbReady('ESP32 (Serial)');
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
        
        // Show global movement bar
        if (this.elements.globalMovementBar) {
            this.elements.globalMovementBar.classList.remove('hidden');
        }
        
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
    
    async readSerialLoop() {
        try {
            while (this.serialPort && this.serialPort.readable) {
                const { value, done } = await this.serialReader.read();
                if (done) break;
                
                // Accumulate data and parse lines
                this.readBuffer += value;
                const lines = this.readBuffer.split('\n');
                this.readBuffer = lines.pop(); // Keep incomplete line
                
                for (const line of lines) {
                    this.parseGameData(line.trim());
                }
            }
        } catch (e) {
            console.error('Serial read error:', e);
            if (this.state !== GameState.IDLE) {
                this.showConnectionStatus('Connection lost. Reconnecting...', 'connecting');
                setTimeout(() => this.connect(), 2000);
            }
        }
    }
    
    parseGameData(line) {
        // Strip ANSI color codes from ESPHome logs
        const cleanLine = line.replace(/\u001b\[[0-9;]*m/g, '');
        
        // Check for system info messages
        // Format: [I][espectre:NNN][espectre]: [sysinfo] key=value or [sysinfo] END
        const sysInfoMatch = cleanLine.match(/\[sysinfo\]\s+(\w+)(?:=(.+))?/);
        if (sysInfoMatch) {
            this.handleSystemInfo(sysInfoMatch[1], sysInfoMatch[2] || '');
            return;
        }
        
        // Look for stream tag with data
        // Format: [I][stream:NNN][espectre]: <movement>,<threshold>
        // Example: [I][stream:075][espectre]: 0.08,1.00
        const match = cleanLine.match(/\[stream:\d+\]\[.*?\]: ([\d.]+),([\d.]+)/);
        if (!match) {
            console.log(line);
            return;
        }
        
        const movement = parseFloat(match[1]);
        const threshold = parseFloat(match[2]);
        
        if (!isNaN(movement) && !isNaN(threshold)) {
            // Only update threshold if user is not dragging the slider
            if (!this.isDraggingThreshold) {
                this.threshold = threshold;
            }
            this.updateMovement(movement);
        }
    }
    
    handleSystemInfo(key, value) {
        this.systemInfo[key] = value;
        
        // Update UI based on key
        switch (key) {
            case 'chip':
                // Update device name in USB ready status (uppercase chip name)
                if (this.elements.usbDeviceName) {
                    this.elements.usbDeviceName.textContent = value.toUpperCase() + ' Connected';
                }
                break;
            case 'threshold':
                if (this.elements.infoThreshold) {
                    this.elements.infoThreshold.textContent = value;
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
    
    async sendSerialCommand(cmd) {
        if (this.serialWriter) {
            await this.serialWriter.write(cmd + '\n');
        }
    }
    
    startPing() {
        // Send PING every 2 seconds to keep connection alive
        this.stopPing();  // Clear any existing interval
        this.pingInterval = setInterval(async () => {
            try {
                await this.sendSerialCommand('PING');
            } catch (e) {
                // Connection lost, stop pinging
                this.stopPing();
            }
        }, 2000);
    }
    
    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    /**
     * Synchronously stop streaming - called on page unload.
     * Uses fire-and-forget pattern since we can't await during beforeunload.
     */
    stopStreamingSync() {
        if (this.serialWriter) {
            // Fire-and-forget: we can't await during unload events
            this.serialWriter.write('STOP\n').catch(() => {});
        }
    }
    
    showConnectionStatus(message, type) {
        this.elements.connectionStatus.textContent = message;
        this.elements.connectionStatus.className = 'connection-status ' + type;
    }
    
    // ==================== INPUT ====================
    
    startMouseMode() {
        this.inputMode = 'mouse';
        
        // Show global movement bar
        if (this.elements.globalMovementBar) {
            this.elements.globalMovementBar.classList.remove('hidden');
        }
        
        this.startGame();
    }
    
    handleMouseMove(e) {
        // Only track in mouse mode
        if (this.inputMode !== 'mouse') return;
        
        const now = performance.now();
        const dt = now - this.lastMouseTime;
        
        if (dt > 0 && this.lastMouseTime > 0) {
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            // Convert to velocity (pixels per ms, scaled)
            // Typical fast mouse movement: 500-1000 pixels in 16ms = 30-60 px/ms
            // We want: threshold (1.0) reachable with moderate mouse speed
            const velocity = (distance / dt) * 0.3;
            
            // Update movement with new velocity (capped at 5.0 for super-fast moves)
            const movement = Math.min(velocity, 5.0);
            this.updateMovement(movement);
            
            // Clear existing decay timer
            if (this.mouseVelocityDecay) {
                clearTimeout(this.mouseVelocityDecay);
            }
            
            // Decay movement to 0 if mouse stops (50ms timeout)
            this.mouseVelocityDecay = setTimeout(() => {
                if (this.inputMode === 'mouse') {
                    this.updateMovement(0);
                }
            }, 50);
        }
        
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.lastMouseTime = now;
    }
    
    updateMovement(value) {
        this.movement = Math.max(0, Math.min(10, value));  // Can exceed 1.0
        
        // Skip visual updates during threshold slider drag to prevent flashing
        if (this.isDraggingThreshold) return;
        
        // Fixed scale: 0% = 0, 100% = 5.0 for movement bar
        const maxDisplay = 5.0;
        const displayPercent = Math.min(100, (this.movement / maxDisplay) * 100);
        
        // Threshold marker: 0% = 0.5, 100% = 5.0
        const minThreshold = 0.5;
        const maxThreshold = 5.0;
        const thresholdPercent = ((this.threshold - minThreshold) / (maxThreshold - minThreshold)) * 100;
        
        this.elements.movementFill.style.width = displayPercent + '%';
        
        // Update threshold marker position
        if (this.elements.movementThresholdMarker) {
            this.elements.movementThresholdMarker.style.left = Math.max(2, Math.min(98, thresholdPercent)) + '%';
        }
        
        // Update numeric values
        if (this.elements.movementValue) {
            this.elements.movementValue.textContent = this.movement.toFixed(2);
        }
        if (this.elements.thresholdValue) {
            this.elements.thresholdValue.textContent = this.threshold.toFixed(2);
        }
        
        // Update bar color based on state
        this.elements.movementFill.classList.remove('warning', 'danger', 'power');
        
        // Power indicator (movement > threshold)
        if (this.movement > this.threshold) {
            this.elements.movementFill.classList.add('power');
        }
        
        if (this.state === GameState.WAITING && this.movement > this.threshold * this.cheatMultiplier) {
            this.elements.movementFill.classList.add('warning');
        }
        if (this.movement > this.threshold * this.cheatMultiplier * 2) {
            this.elements.movementFill.classList.add('danger');
        }
        
        // Check game logic
        this.checkMovement();
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
        // Initialize audio on first user interaction
        this.initAudio();
        
        // Reset stats
        this.score = 0;
        this.streak = 0;
        this.maxStreak = 0;
        this.wave = 1;
        this.spectresDefeated = 0;
        this.bestTime = null;
        this.reactionTimes = [];
        
        // Reset power stats
        this.totalPower = 0;
        this.hitCount = 0;
        this.criticalHits = 0;
        this.strongHits = 0;
        
        this.updateHUD();
        this.showScreen('game');
        
        // Update input mode display
        if (this.inputMode === 'serial') {
            this.elements.inputMode.innerHTML = '<i class="fab fa-usb"></i> USB';
            this.elements.inputMode.classList.add('esp32');
        } else {
            this.elements.inputMode.innerHTML = '<i class="fas fa-mouse"></i> MOUSE';
            this.elements.inputMode.classList.remove('esp32');
        }
        
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
        this.updateMovement(0);
        
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
        
        setTimeout(() => {
            if (this.state === GameState.WAITING) {
                this.trigger();
            }
        }, delay);
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
        setTimeout(() => {
            if (this.state === GameState.TRIGGER) {
                this.handleTooSlow();
            }
        }, this.currentEnemy.maxTime);
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
        
        setTimeout(() => {
            if (this.state === GameState.WAITING) {
                this.trigger();
            }
        }, delay);
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
        setTimeout(() => this.startRound(), 1500);
    }
    
    handleTooSlow() {
        this.state = GameState.LOSE;
        
        // Visual and audio feedback
        this.showMessage('TOO SLOW!', 'lose');
        this.elements.enemy.classList.remove('attacking');
        this.elements.enemy.classList.add('corrupt');
        this.flashScreen('lose');
        this.playSound('lose');
        
        // Game over
        setTimeout(() => this.gameOver(), 1500);
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
        
        // Game over
        setTimeout(() => this.gameOver(), 1500);
    }
    
    async gameOver() {
        this.state = GameState.GAME_OVER;
        
        // Don't send STOP here - keep receiving movement data
        // to show live movement bar on game over screen.
        // STOP will be sent when user goes back home.
        
        // Update results
        document.getElementById('result-spectres').textContent = this.spectresDefeated;
        document.getElementById('result-best').textContent = this.bestTime ? this.bestTime + 'ms' : '---';
        document.getElementById('result-streak').textContent = this.maxStreak;
        document.getElementById('result-score').textContent = this.score;
        
        // Update power stats
        const avgPower = this.hitCount > 0 ? (this.totalPower / this.hitCount) : 0;
        document.getElementById('result-avg-power').textContent = avgPower.toFixed(1) + 'x';
        document.getElementById('result-critical').textContent = this.criticalHits;
        
        // Calculate grade
        const grade = this.calculateGrade();
        const gradeEl = document.getElementById('result-grade');
        gradeEl.textContent = grade;
        gradeEl.className = 'result-grade grade-' + grade.toLowerCase();
        
        this.showScreen('results');
    }
    
    calculateGrade() {
        if (this.spectresDefeated >= 15 && this.bestTime && this.bestTime < 200) return 'S';
        if (this.spectresDefeated >= 10) return 'A';
        if (this.spectresDefeated >= 5) return 'B';
        if (this.spectresDefeated >= 2) return 'C';
        return 'D';
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
        this.elements.streak.textContent = this.streak;
        this.elements.wave.textContent = this.wave;
        this.elements.bestTime.textContent = this.bestTime ? this.bestTime + 'ms' : '---';
        this.elements.score.textContent = this.score;
    }
    
    // ==================== ACTIONS ====================
    
    async restart() {
        // Send START command
        await this.sendSerialCommand('START');
        this.startGame();
    }
    
    share() {
        const avgPower = this.hitCount > 0 ? (this.totalPower / this.hitCount).toFixed(1) : '0.0';
        const text = `ESPectre - The Game\n\n` +
            `Spectres Dissolved: ${this.spectresDefeated}\n` +
            `Best Reaction: ${this.bestTime ? this.bestTime + 'ms' : '---'}\n` +
            `Max Streak: ${this.maxStreak}\n` +
            `Avg Power: ${avgPower}x\n` +
            `Critical Hits: ${this.criticalHits}\n` +
            `Score: ${this.score}\n` +
            `Grade: ${this.calculateGrade()}\n\n` +
            `Play at https://espectre.dev/game`;
        
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
        // Go back to connect screen but keep USB connection active
        this.state = GameState.IDLE;
        this.showScreen('connect');
        
        // If still connected via serial, show the ready state
        if (this.serialPort && this.inputMode === 'serial') {
            // Keep connection-ready visible, movement bar stays visible
            return;
        }
        
        // Mouse mode: reset to pre-connection state
        this.inputMode = 'mouse';
        this.showConnectionStatus('', '');
        
        // Hide global movement bar
        if (this.elements.globalMovementBar) {
            this.elements.globalMovementBar.classList.add('hidden');
        }
        
        // Reset connection box to pre-connection state
        if (this.elements.connectionPre) {
            this.elements.connectionPre.classList.remove('hidden');
        }
        if (this.elements.connectionReady) {
            this.elements.connectionReady.classList.add('hidden');
        }
        
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
        
        // Clear mouse velocity decay timer
        if (this.mouseVelocityDecay) {
            clearTimeout(this.mouseVelocityDecay);
            this.mouseVelocityDecay = null;
        }
    }
    
    async disconnect() {
        // Stop ping keep-alive
        this.stopPing();
        
        // Send STOP command first
        try {
            await this.sendSerialCommand('STOP');
        } catch (e) {}
        
        // Close Serial connection properly
        // Order: cancel reader, close writer, wait for streams, close port
        if (this.serialReader) {
            try {
                await this.serialReader.cancel();
            } catch (e) {}
            this.serialReader = null;
        }
        
        if (this.serialWriter) {
            try {
                await this.serialWriter.close();
            } catch (e) {}
            this.serialWriter = null;
        }
        
        // Wait for streams to finish
        try {
            if (this.readableStreamClosed) await this.readableStreamClosed.catch(() => {});
            if (this.writableStreamClosed) await this.writableStreamClosed.catch(() => {});
        } catch (e) {}
        this.readableStreamClosed = null;
        this.writableStreamClosed = null;
        
        if (this.serialPort) {
            try {
                await this.serialPort.close();
            } catch (e) {}
            this.serialPort = null;
        }
        
        // Clear read buffer
        this.readBuffer = '';
        
        this.state = GameState.IDLE;
        this.inputMode = 'mouse';
        this.showScreen('connect');
        this.showConnectionStatus('Disconnected', '');
        
        // Hide global movement bar
        if (this.elements.globalMovementBar) {
            this.elements.globalMovementBar.classList.add('hidden');
        }
        
        // Reset connection box to pre-connection state
        if (this.elements.connectionPre) {
            this.elements.connectionPre.classList.remove('hidden');
        }
        if (this.elements.connectionReady) {
            this.elements.connectionReady.classList.add('hidden');
        }
        
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
    }
    
    // ==================== AUDIO SYSTEM ====================
    
    initAudio() {
        if (this.audioInitialized) return;
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.audioInitialized = true;
        } catch (e) {
            console.warn('Web Audio not available:', e);
            this.audioEnabled = false;
        }
    }
    
    toggleMute() {
        this.audioEnabled = !this.audioEnabled;
        
        if (this.elements.btnMute) {
            const icon = this.elements.btnMute.querySelector('i');
            if (icon) {
                icon.className = this.audioEnabled ? 'fas fa-volume-up' : 'fas fa-volume-mute';
            }
            this.elements.btnMute.classList.toggle('muted', !this.audioEnabled);
        }
    }
    
    playSound(type, options = {}) {
        if (!this.audioEnabled || !this.audioContext) return;
        
        // Resume audio context if suspended (browser autoplay policy)
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }
        
        const ctx = this.audioContext;
        const now = ctx.currentTime;
        
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
