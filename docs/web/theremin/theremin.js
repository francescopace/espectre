  let isDisconnecting = false;
  let messageCount = 0;

  const ble = new ESPectreBleClient({
      onTelemetry: (telemetry) => handleMessage(telemetry),
      onSysinfoSnapshot: (_snapshot, entries) => {
          const chipEntry = entries.find(([key]) => key === 'chip');
          if (chipEntry) {
              document.getElementById('deviceName').textContent = `${chipEntry[1].toUpperCase()} Connected`;
          }
      },
      onDisconnected: () => {
          if (isDisconnecting) return;
          disconnect(false).then(() => {
              showConnectionStatus('Bluetooth connection lost.', 'error');
          });
      }
  });

  // Optional feature data is not part of the current BLE telemetry payload.
  let currentFeatures = null;

  // Web Audio API
  let audioContext = null;
  let oscillator = null;
  let gainNode = null;
  let isAudioInitialized = false;

  // Theremin State
  let thereminConfig = {
      enabled: true, // Always enabled
      mode: 'continuous', // continuous, quantized, hybrid
      scale: 'chromatic', // pentatonic, major, minor, chromatic
      baseFrequency: 200,
      frequencyRange: 1800,
      // 0 = follow target immediately, 1 = maximum lag.
      smoothingFactor: 0.2,
      volume: 0.5, // 0-1
      logarithmicMapping: true,
      // Fraction of the live movement scale treated as "quiet" in hybrid mode.
      hybridThreshold: 0.08,
      maxMovement: 1.0 // Maximum movement probability that maps to highest frequency
  };

  // Tremolo State
  let tremoloState = {
      packetsProcessed: 0,
      lastPacketsProcessed: 0,
      tremoloPhase: 0,
      tremoloTime: 0
  };

  // Feature Modulation Configuration
  let featureModConfig = {
      waveformEnabled: false,
      vibratoEnabled: false,
      vibratoSpeedMult: 5.0,
      vibratoDepthMult: 20, // cents
      filterEnabled: false,
      filterCutoffBase: 4000, // Hz
      filterResonanceMult: 5,
      stereoPanEnabled: false,
      autoScaleEnabled: false,
      tremoloEnabled: false,
      tremoloSpeed: 5.0, // multiplier
      tremoloDepth: 0.5, // 0-1 (50%)
      effectsEnabled: false
  };

  // Vibrato state
  let vibratoState = {
      phase: 0,
      time: 0
  };

  // Interpolation State
  let lastMovement = 0;
  let targetMovement = 0;
  let smoothedMovement = 0;
  let lastUpdateTime = 0;
  let interpolationActive = false;
  const INTERPOLATION_RATE = 60; // Hz (60 updates per second)
  // Recent peak used so small live scores still span the audible range.
  let movementPeak = 0.5;

  // Visual waveform (amplitude ← movement, oscillation ← pitch)
  const waveVisual = {
      canvas: null,
      ctx: null,
      rafId: 0,
      phase: 0,
      lastFrameMs: 0,
      frequencyHz: 200,
      amplitude: 0,
      reduceMotion: false,
      colors: null
  };

  // Musical Scales
  const PENTATONIC_RATIOS = [1.0, 9/8, 5/4, 3/2, 5/3, 2.0];
  const MAJOR_RATIOS = [1.0, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2.0];
  const MINOR_RATIOS = [1.0, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5, 2.0];
  const CHROMATIC_RATIOS = [];

  // Initialize chromatic scale
  for (let i = 0; i <= 12; i++) {
      CHROMATIC_RATIOS.push(Math.pow(2, i / 12));
  }

  // Get scale ratios
  function getScaleRatios(scale) {
      switch (scale) {
          case 'pentatonic':
              return PENTATONIC_RATIOS;
          case 'major':
              return MAJOR_RATIOS;
          case 'minor':
              return MINOR_RATIOS;
          case 'chromatic':
              return CHROMATIC_RATIOS;
          default:
              return PENTATONIC_RATIOS;
      }
  }

  // Convert frequency to MIDI note number
  function frequencyToMIDI(freq) {
      // A4 = 440 Hz = MIDI note 69
      if (freq <= 0) return null;
      const midiNote = 12 * Math.log2(freq / 440) + 69;
      return midiNote;
  }

  // Convert MIDI note to note name
  function midiToNoteName(midiNote) {
      if (midiNote === null || midiNote < 0 || midiNote > 127) return '-';

      const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
      const note = noteNames[Math.round(midiNote) % 12];
      const octave = Math.floor(Math.round(midiNote) / 12) - 1;
      return note + octave;
  }

  // Get octave from MIDI note
  function midiToOctave(midiNote) {
      if (midiNote === null || midiNote < 0 || midiNote > 127) return '-';
      return Math.floor(Math.round(midiNote) / 12) - 1;
  }

  function initWaveVisual() {
      waveVisual.canvas = document.getElementById('thereminWave');
      if (!waveVisual.canvas) return;
      waveVisual.ctx = waveVisual.canvas.getContext('2d');
      waveVisual.reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      cacheWaveColors();
      resizeWaveVisual();
      window.addEventListener('resize', resizeWaveVisual);
  }

  function cacheWaveColors() {
      const styles = getComputedStyle(document.documentElement);
      waveVisual.colors = {
          accent: styles.getPropertyValue('--accent').trim() || '#7aa2e3',
          accentSecondary: styles.getPropertyValue('--accent-secondary').trim() || '#9b8afb',
          glow: styles.getPropertyValue('--accent-glow').trim() || 'rgba(122, 162, 227, 0.18)'
      };
  }

  function resizeWaveVisual() {
      const canvas = waveVisual.canvas;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.round(rect.width * dpr));
      const height = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
      }
  }

  function movementScale() {
      const configuredMax = Math.max(Number(thereminConfig.maxMovement) || 0, 0.001);
      // Follow recent peaks so low live scores are not crushed into one note,
      // but never exceed the user-configured ceiling.
      return Math.min(configuredMax, Math.max(movementPeak * 1.1, 0.12));
  }

  function updateMovementPeak(movement) {
      const value = Number(movement);
      if (!Number.isFinite(value) || value < 0) return;
      if (value > movementPeak) {
          movementPeak = value;
          return;
      }
      // Decay quickly enough that the next gesture stays expressive.
      movementPeak = Math.max(0.12, movementPeak * 0.985);
  }

  function setWaveVisual(movement, frequencyHz) {
      const scale = movementScale();
      const normalized = Math.min(Math.max(movement / scale, 0), 1);
      // Keep a faint idle ripple so the stage never looks dead.
      waveVisual.amplitude = 0.06 + normalized * 0.94;
      if (Number.isFinite(frequencyHz) && frequencyHz > 0) {
          waveVisual.frequencyHz = frequencyHz;
      }
  }

  function startWaveVisual() {
      if (!waveVisual.canvas || waveVisual.rafId) return;
      // Workspace may have been hidden at init; measure after it becomes visible.
      resizeWaveVisual();
      waveVisual.lastFrameMs = 0;
      setWaveVisual(0, thereminConfig.baseFrequency);
      const tick = (nowMs) => {
          drawWaveVisual(nowMs);
          waveVisual.rafId = requestAnimationFrame(tick);
      };
      waveVisual.rafId = requestAnimationFrame(tick);
  }

  function stopWaveVisual() {
      if (waveVisual.rafId) {
          cancelAnimationFrame(waveVisual.rafId);
          waveVisual.rafId = 0;
      }
      waveVisual.amplitude = 0;
      waveVisual.phase = 0;
      waveVisual.lastFrameMs = 0;
      drawWaveVisual(performance.now(), true);
  }

  function drawWaveVisual(nowMs, forceFlat = false) {
      const canvas = waveVisual.canvas;
      const ctx = waveVisual.ctx;
      if (!canvas || !ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      if (width < 2 || height < 2) return;

      const dt = waveVisual.lastFrameMs ? Math.min(0.05, (nowMs - waveVisual.lastFrameMs) / 1000) : 0;
      waveVisual.lastFrameMs = nowMs;

      const amp = forceFlat ? 0.04 : waveVisual.amplitude;
      const freq = waveVisual.frequencyHz;
      // Map audible pitch into a readable number of cycles across the canvas.
      const cycles = 1.2 + Math.log2(Math.max(freq, 40) / 40) * 1.15;
      const phaseSpeed = (Math.PI * 2) * (0.35 + Math.min(freq, 2000) / 900);
      if (!waveVisual.reduceMotion && !forceFlat) {
          waveVisual.phase = (waveVisual.phase + phaseSpeed * dt) % (Math.PI * 2);
      }

      ctx.clearRect(0, 0, width, height);

      const midY = height * 0.5;
      const maxAmpPx = height * 0.38 * amp;
      if (!waveVisual.colors) cacheWaveColors();
      const accent = waveVisual.colors.accent;
      const accentSecondary = waveVisual.colors.accentSecondary;
      const glow = waveVisual.colors.glow;

      // Soft center glow
      const glowGrad = ctx.createRadialGradient(width * 0.5, midY, 0, width * 0.5, midY, height * 0.7);
      glowGrad.addColorStop(0, glow);
      glowGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = glowGrad;
      ctx.fillRect(0, 0, width, height);

      const buildPath = (amplitudeScale) => {
          ctx.beginPath();
          for (let x = 0; x <= width; x += 2) {
              const t = x / width;
              const angle = (t * cycles * Math.PI * 2) + waveVisual.phase;
              // Slight harmonic for a less clinical sine.
              const y = midY
                  - Math.sin(angle) * maxAmpPx * amplitudeScale
                  - Math.sin(angle * 2) * maxAmpPx * 0.12 * amplitudeScale;
              if (x === 0) ctx.moveTo(x, y);
              else ctx.lineTo(x, y);
          }
      };

      // Filled body under the wave
      buildPath(1);
      ctx.lineTo(width, midY + maxAmpPx * 0.2);
      ctx.lineTo(0, midY + maxAmpPx * 0.2);
      ctx.closePath();
      const fillGrad = ctx.createLinearGradient(0, midY - maxAmpPx, 0, midY + maxAmpPx);
      fillGrad.addColorStop(0, 'rgba(122, 162, 227, 0.22)');
      fillGrad.addColorStop(0.55, 'rgba(155, 138, 251, 0.08)');
      fillGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = fillGrad;
      ctx.fill();

      // Main stroke
      buildPath(1);
      const strokeGrad = ctx.createLinearGradient(0, 0, width, 0);
      strokeGrad.addColorStop(0, accent);
      strokeGrad.addColorStop(0.5, accentSecondary);
      strokeGrad.addColorStop(1, accent);
      ctx.strokeStyle = strokeGrad;
      ctx.lineWidth = Math.max(2, width * 0.0025);
      ctx.lineJoin = 'round';
      ctx.shadowColor = accent;
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Dim mirrored echo
      buildPath(0.35);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
      ctx.lineWidth = Math.max(1, width * 0.0015);
      ctx.stroke();
  }

  // Map movement to frequency (logarithmic or linear)
  function mapMovementToFrequency(movement) {
      updateMovementPeak(movement);
      const scale = movementScale();
      const normalized = Math.min(Math.max(movement / scale, 0), 1);

      const baseFreq = thereminConfig.baseFrequency;
      const freqRange = thereminConfig.frequencyRange;
      const maxFreq = baseFreq + freqRange;

      let freq;
      if (thereminConfig.logarithmicMapping) {
          // Logarithmic mapping (musical scale)
          const octaves = Math.log2(maxFreq / baseFreq);
          const targetOctaves = normalized * octaves;
          freq = baseFreq * Math.pow(2, targetOctaves);
      } else {
          // Linear mapping
          freq = baseFreq + (normalized * freqRange);
      }

      if (!Number.isFinite(freq)) {
          return baseFreq;
      }
      return Math.max(20, Math.min(20000, freq)); // Clamp to audible range
  }

  // Quantize frequency to nearest note in scale
  function quantizeToScale(freq) {
      const ratios = getScaleRatios(thereminConfig.scale);
      const baseFreq = thereminConfig.baseFrequency;

      // Find which octave we're in
      const octave = Math.floor(Math.log2(freq / baseFreq));
      const baseInOctave = baseFreq * Math.pow(2, octave);

      // Calculate ratio within octave
      const ratio = freq / baseInOctave;

      // Find closest note in scale
      let closestIdx = 0;
      let minDiff = Math.abs(ratio - ratios[0]);

      for (let i = 1; i < ratios.length; i++) {
          const diff = Math.abs(ratio - ratios[i]);
          if (diff < minDiff) {
              minDiff = diff;
              closestIdx = i;
          }
      }

      // Return quantized frequency
      return baseInOctave * ratios[closestIdx];
  }

  // Apply theremin mode processing
  function processFrequency(targetFreq, movementDelta) {
      // Get scale (auto or manual)
      const scale = getAutoScale();
      const originalScale = thereminConfig.scale;

      // Temporarily override scale if auto-scale is enabled
      if (featureModConfig.autoScaleEnabled && scale !== originalScale) {
          thereminConfig.scale = scale;
      }

      let result;
      switch (thereminConfig.mode) {
          case 'continuous':
              result = targetFreq;
              break;

          case 'quantized':
              result = quantizeToScale(targetFreq);
              break;

          case 'hybrid': {
              // Quantize only while nearly still; otherwise glide continuously.
              const quietDelta = Math.max(0.015, movementScale() * thereminConfig.hybridThreshold);
              if (Math.abs(movementDelta) < quietDelta) {
                  result = quantizeToScale(targetFreq);
              } else {
                  result = targetFreq;
              }
              break;
          }

          default:
              result = targetFreq;
      }

      // Restore original scale
      if (featureModConfig.autoScaleEnabled && scale !== originalScale) {
          thereminConfig.scale = originalScale;
      }

      return result;
  }

  // Initialize Web Audio API
  function initAudio() {
      if (isAudioInitialized) return;

      try {
          audioContext = new (window.AudioContext || window.webkitAudioContext)();
          gainNode = audioContext.createGain();
          oscillator = audioContext.createOscillator();

          oscillator.type = 'sine';
          oscillator.frequency.value = thereminConfig.baseFrequency;
          oscillator.connect(gainNode);
          gainNode.connect(audioContext.destination);

          gainNode.gain.value = 0; // Start muted
          oscillator.start();

          isAudioInitialized = true;
          console.log('Audio initialized');
      } catch (err) {
          console.error('Failed to initialize audio:', err);
          showNotification('Audio initialization failed. Click to enable.', 'error');
      }
  }

  // Resume audio context (required for autoplay policy)
  async function resumeAudio() {
      if (!audioContext) {
          initAudio();
      }

      if (audioContext.state === 'suspended') {
          await audioContext.resume();
      }

      if (gainNode && gainNode.gain.value === 0) {
          gainNode.gain.value = thereminConfig.volume;
      }
  }

  // Update octave display
  function updateOctaveOrFeature(octave) {
      const featureLabel = document.getElementById('featureLabel');
      const octaveOrFeature = document.getElementById('octaveOrFeature');

      // Always show octave
      featureLabel.textContent = 'Octave';
      octaveOrFeature.textContent = octave;
  }

  // Update oscillator frequency
  function updateFrequency(freq) {
      if (!oscillator || !isAudioInitialized) return;
      if (!Number.isFinite(freq)) return;

      // Clamp frequency to valid range
      freq = Math.max(20, Math.min(20000, freq));

      // Apply vibrato if enabled
      const vibratoRatio = calculateVibrato();
      if (vibratoRatio !== 1.0) {
          freq = freq * vibratoRatio;
      }

      // Update waveform type
      const waveformType = getWaveformType();
      if (oscillator.type !== waveformType) {
          oscillator.type = waveformType;
      }

      // Use exponential ramp for smooth transitions
      const currentFreq = oscillator.frequency.value;
      const now = audioContext.currentTime;

      oscillator.frequency.setTargetAtTime(freq, now, 0.004);

      // Update filter
      updateFilter(freq);

      // Update stereo pan
      updateStereoPan();

      // Calculate and display musical note
      const midiNote = frequencyToMIDI(freq);
      const noteName = midiToNoteName(midiNote);
      const octave = midiToOctave(midiNote);

      // Update UI
      document.getElementById('frequencyValue').textContent = freq.toFixed(1) + ' Hz';
      document.getElementById('currentNote').textContent = noteName;
      updateOctaveOrFeature(octave);
  }


  // Get waveform type based on entropy
  function getWaveformType() {
      if (!featureModConfig.waveformEnabled || !currentFeatures || currentFeatures.entropy === undefined) {
          return 'sine';
      }

      // Normalize entropy (typical range: 0-8)
      const normalized = Math.min(currentFeatures.entropy / 8.0, 1.0);

      if (normalized < 0.33) {
          return 'sine';
      } else if (normalized < 0.66) {
          return 'square';
      } else {
          return 'sawtooth';
      }
  }

  // Calculate vibrato modulation
  function calculateVibrato() {
      if (!featureModConfig.vibratoEnabled || !currentFeatures) {
          return 1.0; // No modulation
      }

      const temporalDelta = currentFeatures.temporal_delta_mean || 0;
      const spatialVar = currentFeatures.spatial_variance || 0;

      // Speed based on temporal_delta_mean (normalize to 0-10 Hz range)
      const speed = Math.min(temporalDelta * featureModConfig.vibratoSpeedMult, 20);

      // Depth based on spatial_variance (normalize to 0-1, then to cents)
      const depthNormalized = Math.min(spatialVar / 10.0, 1.0);
      const depthCents = depthNormalized * featureModConfig.vibratoDepthMult;

      // Update vibrato phase
      vibratoState.time += 1.0 / INTERPOLATION_RATE;
      vibratoState.phase = (vibratoState.phase + (speed / INTERPOLATION_RATE)) % (2 * Math.PI);

      // Calculate vibrato in cents (convert to frequency ratio)
      const vibratoCents = Math.sin(vibratoState.phase) * depthCents;
      const frequencyRatio = Math.pow(2, vibratoCents / 1200);

      return frequencyRatio;
  }

  // Update filter if enabled
  function updateFilter(freq) {
      if (!gainNode || !isAudioInitialized) {
          return;
      }

      // If filter is disabled but exists, remove it
      if (!featureModConfig.filterEnabled && window.biquadFilter) {
          oscillator.disconnect();
          if (window.stereoPanner) {
              oscillator.connect(window.stereoPanner);
          } else {
              oscillator.connect(gainNode);
          }
          window.biquadFilter = null;
          return;
      }

      if (!featureModConfig.filterEnabled) {
          return;
      }

      // Create filter if it doesn't exist
      if (!window.biquadFilter) {
          window.biquadFilter = audioContext.createBiquadFilter();
          window.biquadFilter.type = 'lowpass';
          oscillator.disconnect();
          oscillator.connect(window.biquadFilter);
          window.biquadFilter.connect(gainNode);
      }

      if (!currentFeatures) {
          window.biquadFilter.frequency.value = featureModConfig.filterCutoffBase;
          window.biquadFilter.Q.value = 1;
          return;
      }

      // Cutoff based on kurtosis (normalize: typical range 0-10)
      const kurtosisNorm = Math.min((currentFeatures.kurtosis || 0) / 10.0, 1.0);
      const cutoff = featureModConfig.filterCutoffBase * (0.3 + 0.7 * kurtosisNorm);

      // Resonance based on skewness (normalize: typical range -2 to 2)
      const skewnessNorm = Math.min(Math.max((currentFeatures.skewness || 0) + 2, 0) / 4.0, 1.0);
      const Q = 1 + (skewnessNorm * featureModConfig.filterResonanceMult);

      window.biquadFilter.frequency.setTargetAtTime(cutoff, audioContext.currentTime, 0.01);
      window.biquadFilter.Q.setTargetAtTime(Q, audioContext.currentTime, 0.01);
  }

  // Update stereo pan
  function updateStereoPan() {
      if (!gainNode || !isAudioInitialized) {
          return;
      }

      // If stereo pan is disabled but exists, remove it
      if (!featureModConfig.stereoPanEnabled && window.stereoPanner) {
          oscillator.disconnect();
          if (window.biquadFilter) {
              window.biquadFilter.disconnect();
              oscillator.connect(window.biquadFilter);
              window.biquadFilter.connect(gainNode);
          } else {
              oscillator.connect(gainNode);
          }
          window.stereoPanner = null;
          return;
      }

      if (!featureModConfig.stereoPanEnabled) {
          return;
      }

      // Create panner if it doesn't exist
      if (!window.stereoPanner) {
          window.stereoPanner = audioContext.createStereoPanner();
          // But we need to handle the filter too if it exists
          oscillator.disconnect();
          if (window.biquadFilter) {
              oscillator.connect(window.biquadFilter);
              window.biquadFilter.disconnect();
              window.biquadFilter.connect(window.stereoPanner);
              window.stereoPanner.connect(gainNode);
          } else {
              oscillator.connect(window.stereoPanner);
              window.stereoPanner.connect(gainNode);
          }
      }

      if (!currentFeatures || currentFeatures.spatial_correlation === undefined) {
          window.stereoPanner.pan.value = 0;
          return;
      }

      // Pan based on spatial_correlation (normalize: typical range -1 to 1)
      const pan = Math.max(-1, Math.min(1, currentFeatures.spatial_correlation));
      window.stereoPanner.pan.setTargetAtTime(pan, audioContext.currentTime, 0.01);
  }

  // Auto-select scale based on features
  function getAutoScale() {
      if (!featureModConfig.autoScaleEnabled || !currentFeatures) {
          return thereminConfig.scale; // Use manual selection
      }

      const entropy = currentFeatures.entropy || 0;
      const variance = currentFeatures.variance || 0;

      // Combine entropy and variance (both normalized)
      const complexity = (Math.min(entropy / 8.0, 1.0) + Math.min(variance / 10.0, 1.0)) / 2.0;

      if (complexity < 0.33) {
          return 'pentatonic';
      } else if (complexity < 0.66) {
          return 'major';
      } else {
          return 'chromatic';
      }
  }

  // Tremolo using features
  function updateTremolo() {
      if (!featureModConfig.tremoloEnabled || !gainNode || !isAudioInitialized) {
          // Reset to base volume if tremolo is disabled
          if (gainNode) {
              gainNode.gain.setTargetAtTime(thereminConfig.volume, audioContext.currentTime, 0.01);
          }
          return;
      }

      if (!currentFeatures) {
          return;
      }

      const temporalVar = currentFeatures.temporal_delta_variance || 0;
      const spatialGrad = currentFeatures.spatial_gradient || 0;

      // Speed based on temporal_delta_variance
      const speed = Math.min(temporalVar * featureModConfig.tremoloSpeed, 20);

      // Depth based on spatial_gradient
      const depthNormalized = Math.min(spatialGrad / 5.0, 1.0);
      const depth = depthNormalized * featureModConfig.tremoloDepth;

      // Update tremolo phase
      tremoloState.tremoloTime += 1.0 / INTERPOLATION_RATE;
      tremoloState.tremoloPhase = (tremoloState.tremoloPhase + (speed / INTERPOLATION_RATE)) % (2 * Math.PI);

      const modulation = Math.sin(tremoloState.tremoloPhase) * depth;
      const baseVolume = thereminConfig.volume;
      const modulatedVolume = baseVolume * (1.0 + modulation);
      const clampedVolume = Math.max(0, Math.min(1, modulatedVolume));

      gainNode.gain.setTargetAtTime(clampedVolume, audioContext.currentTime, 0.01);
  }

  // High-frequency interpolation loop
  function startInterpolationLoop() {
      if (interpolationActive) return;
      interpolationActive = true;

      const updateInterval = 1000 / INTERPOLATION_RATE; // ~16.67ms for 60Hz

      function interpolate() {
          if (!interpolationActive) return;

          try {
              if (lastUpdateTime > 0) {
                  // smoothingFactor: 0 = snappy, 1 = maximum lag (matches the UI label).
                  const smooth = Math.min(Math.max(thereminConfig.smoothingFactor, 0), 0.95);
                  const follow = 1 - smooth;
                  smoothedMovement = follow * targetMovement + smooth * smoothedMovement;

                  // Map to frequency
                  const targetFreq = mapMovementToFrequency(smoothedMovement);
                  const movementDelta = Math.abs(targetMovement - lastMovement);
                  const finalFreq = processFrequency(targetFreq, movementDelta);

                  // Update audio
                  if (thereminConfig.enabled) {
                      updateFrequency(finalFreq);
                      // Update tremolo if enabled
                      if (featureModConfig.tremoloEnabled) {
                          updateTremolo();
                      }
                  }

                  // Update UI
                  document.getElementById('movementValue').textContent = smoothedMovement.toFixed(3);
                  setWaveVisual(smoothedMovement, finalFreq);
              } else {
                  // No telemetry data yet, but still update tremolo if enabled.
                  if (featureModConfig.tremoloEnabled) {
                      updateTremolo();
                  }
                  setWaveVisual(0, thereminConfig.baseFrequency);
              }
          } catch (error) {
              console.warn('Theremin interpolation tick failed:', error);
          } finally {
              if (interpolationActive) {
                  setTimeout(interpolate, updateInterval);
              }
          }
      }

      interpolate();
  }

  // Bluetooth Functions
  function showConnectionStatus(message, type = '') {
      const status = document.getElementById('connectionStatus');
      status.textContent = message;
      status.className = `connection-status ${type}`.trim();
  }

  function updateConnectionUi(connected) {
      document.getElementById('connectionPre').hidden = connected;
      document.getElementById('connectionReady').hidden = !connected;
      document.getElementById('thereminWorkspace').hidden = !connected;
      ToolPage.setHeaderConnectionStatus(connected);
      if (connected) {
          requestAnimationFrame(() => resizeWaveVisual());
      }
  }

  function toggleConnection() {
      if (ble.connected) {
          disconnect();
      } else {
          connect();
      }
  }

  async function connect() {
      if (!ESPectreBleClient.supported) {
          showConnectionStatus('Web Bluetooth is not supported. Use Chrome or Edge.', 'error');
          return;
      }

      const connectButton = document.getElementById('connectBtn');
      connectButton.disabled = true;
      showConnectionStatus('Requesting Bluetooth device...', 'connecting');
      trackEvent('tool_connection', {
          tool_name: 'theremin',
          transport: 'bluetooth',
          result: 'attempt'
      });

      try {
          await ble.connect();
          document.getElementById('deviceName').textContent = `${ble.name || 'ESP32'} Connected`;
          showConnectionStatus('', '');
          updateConnectionUi(true);
          await resumeAudio();
          startInterpolationLoop();
          startWaveVisual();
          await ble.writeControl('REQ_SYSINFO');
          trackEvent('tool_connection', {
              tool_name: 'theremin',
              transport: 'bluetooth',
              result: 'success'
          });
      } catch (error) {
          console.error('Bluetooth connection failed:', error);
          const cancelled = error.name === 'NotFoundError';
          showConnectionStatus(
              cancelled ? 'No ESPectre device selected.' : `Connection failed: ${error.message}`,
              'error'
          );
          trackEvent('tool_connection', {
              tool_name: 'theremin',
              transport: 'bluetooth',
              result: 'failure',
              error_type: error.name || 'connection_error'
          });
          await disconnect(false);
      } finally {
          connectButton.disabled = false;
      }
  }

  async function disconnect(clearStatus = true) {
      if (isDisconnecting) return;
      isDisconnecting = true;
      interpolationActive = false;
      stopWaveVisual();

      try {
          await ble.disconnect();
      } finally {
          if (gainNode && audioContext) {
              gainNode.gain.setTargetAtTime(0, audioContext.currentTime, 0.01);
          }
          clearData();
          updateConnectionUi(false);
          if (clearStatus) showConnectionStatus('');
          isDisconnecting = false;
      }
  }


  function handleMessage(data) {
      if (data && data.protocol_version !== undefined) {
          data = {
              ...data,
              movement: data.movement_score,
              state: data.motion_state
          };
      }
      // Check if this is a regular data message
      if (data.movement !== undefined || data.threshold !== undefined || data.state) {
          messageCount++;

          // Show/hide Feature Modulation section based on features presence
          const featureSection = document.getElementById('featureModulationSection');
          if (data.features) {
              // Features are available - show the section
              featureSection.hidden = false;
              currentFeatures = data.features;
          } else {
              // No features (e.g., Micro-ESPectre) - hide the section
              featureSection.hidden = true;
              currentFeatures = null;
          }

          // Update target movement for interpolation
          if (data.movement !== undefined) {
              lastMovement = targetMovement;
              targetMovement = data.movement;
              lastUpdateTime = Date.now();

              // Initialize smoothedMovement on first message
              if (smoothedMovement === 0 && targetMovement > 0) {
                  smoothedMovement = targetMovement;
              }
          }

          // Update tremolo state with packets_processed
          if (data.packets_processed !== undefined) {
              tremoloState.lastPacketsProcessed = tremoloState.packetsProcessed;
              tremoloState.packetsProcessed = data.packets_processed;
          }
      }
  }

  // UI Control Functions
  // SubSection Toggle Function
  function syncSlider(sliderId, valueId) {
      const slider = document.getElementById(sliderId);
      const valueInput = document.getElementById(valueId);

      slider.addEventListener('input', (e) => {
          valueInput.value = e.target.value;
          updateConfig();
      });

      valueInput.addEventListener('input', (e) => {
          slider.value = e.target.value;
          updateConfig();
      });
  }

  function updateConfig() {
      thereminConfig.enabled = true; // Always enabled
      thereminConfig.mode = document.getElementById('thereminMode').value;
      thereminConfig.scale = document.getElementById('thereminScale').value;
      thereminConfig.baseFrequency = parseFloat(document.getElementById('baseFreqValue').value);
      thereminConfig.frequencyRange = parseFloat(document.getElementById('freqRangeValue').value);
      thereminConfig.smoothingFactor = parseFloat(document.getElementById('smoothingValue').value) / 100;
      thereminConfig.volume = parseFloat(document.getElementById('volumeValue').value) / 100;
      thereminConfig.maxMovement = parseFloat(document.getElementById('maxMovementValue').value);
      thereminConfig.logarithmicMapping = document.getElementById('logarithmicMapping').checked;

      // Feature modulation config
      featureModConfig.waveformEnabled = document.getElementById('waveformModEnable').checked;
      featureModConfig.vibratoEnabled = document.getElementById('vibratoModEnable').checked;
      featureModConfig.vibratoSpeedMult = parseFloat(document.getElementById('vibratoSpeedValue').value);
      featureModConfig.vibratoDepthMult = parseFloat(document.getElementById('vibratoDepthValue').value);
      featureModConfig.filterEnabled = document.getElementById('filterModEnable').checked;
      featureModConfig.filterCutoffBase = parseFloat(document.getElementById('filterCutoffValue').value);
      featureModConfig.filterResonanceMult = parseFloat(document.getElementById('filterResonanceValue').value);
      featureModConfig.stereoPanEnabled = document.getElementById('stereoPanEnable').checked;
      featureModConfig.autoScaleEnabled = document.getElementById('autoScaleEnable').checked;
      featureModConfig.tremoloEnabled = document.getElementById('tremoloEnable').checked;
      featureModConfig.tremoloSpeed = parseFloat(document.getElementById('tremoloSpeedValue').value);
      featureModConfig.tremoloDepth = parseFloat(document.getElementById('tremoloDepthValue').value) / 100;
      featureModConfig.effectsEnabled = document.getElementById('effectsModEnable').checked;

      // Update audio volume (tremolo will modulate it if enabled)
      if (gainNode) {
          if (!thereminConfig.enabled) {
              gainNode.gain.value = 0;
          } else if (!featureModConfig.tremoloEnabled) {
              gainNode.gain.value = thereminConfig.volume;
          }
          // If tremolo is enabled, it will be updated in the interpolation loop
      }
  }

  function clearData() {
      messageCount = 0;
      document.getElementById('movementValue').textContent = '-';
      document.getElementById('frequencyValue').textContent = '-';
      document.getElementById('currentNote').textContent = '-';
      document.getElementById('octaveOrFeature').textContent = '-';
      document.getElementById('featureLabel').textContent = 'Octave';

      // Reset theremin state
      lastMovement = 0;
      targetMovement = 0;
      smoothedMovement = 0;
      lastUpdateTime = 0;
      movementPeak = 0.5;
      currentFeatures = null;
      setWaveVisual(0, thereminConfig.baseFrequency);

      // Reset audio if initialized
      if (oscillator && isAudioInitialized) {
          oscillator.frequency.setTargetAtTime(thereminConfig.baseFrequency, audioContext.currentTime, 0.01);
      }
  }

  // Initialize on load
  window.addEventListener('load', () => {
      const connectButton = document.getElementById('connectBtn');
      connectButton.addEventListener('click', toggleConnection);
      document.getElementById('disconnectBtn').addEventListener('click', () => disconnect());
      if (!ESPectreBleClient.supported) {
          connectButton.disabled = true;
          showConnectionStatus('Web Bluetooth is not supported. Use Chrome or Edge.', 'error');
      }

      // Sync sliders
      syncSlider('baseFreq', 'baseFreqValue');
      syncSlider('freqRange', 'freqRangeValue');
      syncSlider('smoothing', 'smoothingValue');
      syncSlider('volume', 'volumeValue');
      syncSlider('maxMovement', 'maxMovementValue');
      syncSlider('tremoloSpeed', 'tremoloSpeedValue');
      syncSlider('tremoloDepth', 'tremoloDepthValue');
      syncSlider('vibratoSpeed', 'vibratoSpeedValue');
      syncSlider('vibratoDepth', 'vibratoDepthValue');
      syncSlider('filterCutoff', 'filterCutoffValue');
      syncSlider('filterResonance', 'filterResonanceValue');

      // Setup event listeners
      document.getElementById('thereminMode').addEventListener('change', updateConfig);
      document.getElementById('thereminScale').addEventListener('change', updateConfig);
      document.getElementById('thereminMode').addEventListener('change', (event) => {
          trackEvent('theremin_configuration', { control: 'mode', setting_value: event.target.value });
      });
      document.getElementById('thereminScale').addEventListener('change', (event) => {
          trackEvent('theremin_configuration', { control: 'scale', setting_value: event.target.value });
      });
      document.getElementById('logarithmicMapping').addEventListener('change', updateConfig);

      // Feature modulation event listeners
      document.getElementById('waveformModEnable').addEventListener('change', updateConfig);
      document.getElementById('vibratoModEnable').addEventListener('change', updateConfig);
      document.getElementById('filterModEnable').addEventListener('change', updateConfig);
      document.getElementById('stereoPanEnable').addEventListener('change', updateConfig);
      document.getElementById('autoScaleEnable').addEventListener('change', updateConfig);
      document.getElementById('tremoloEnable').addEventListener('change', updateConfig);
      document.getElementById('effectsModEnable').addEventListener('change', updateConfig);

      // Initialize audio on user interaction
      document.addEventListener('click', () => {
          if (!isAudioInitialized) {
              initAudio();
          }
      }, { once: true });

      // Initialize config
      updateConfig();
      initWaveVisual();
  });
