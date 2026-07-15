// MQTT Client
  let client = null;
  let messageCount = 0;

  // Current features from MQTT
  let currentFeatures = null;

  // Web Audio API
  let audioContext = null;
  let oscillator = null;
  let gainNode = null;
  let isAudioInitialized = false;

  // Theremin State
  let thereminConfig = {
      enabled: true, // Always enabled
      mode: 'quantized', // continuous, quantized, hybrid
      scale: 'pentatonic', // pentatonic, major, minor, chromatic
      baseFrequency: 200,
      frequencyRange: 1800,
      smoothingFactor: 0.3, // 0-1
      volume: 0.5, // 0-1
      logarithmicMapping: true,
      hybridThreshold: 0.5,
      maxMovement: 4.0 // Maximum movement value that maps to highest frequency
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
  const MQTT_RATE = 1; // Hz (1 message per second)

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

  // Map movement to frequency (logarithmic or linear)
  function mapMovementToFrequency(movement) {
      // Normalize movement (0-1) using configurable max movement value
      const normalized = Math.min(Math.max(movement / thereminConfig.maxMovement, 0), 1);

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

          case 'hybrid':
              if (Math.abs(movementDelta) < thereminConfig.hybridThreshold) {
                  result = quantizeToScale(targetFreq);
              } else {
                  result = targetFreq;
              }
              break;

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

      oscillator.frequency.setTargetAtTime(freq, now, 0.01);

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

          const now = Date.now();
          const timeSinceLastMQTT = (now - lastUpdateTime) / 1000; // seconds

          // If we have a target and it's been less than 2 seconds since last MQTT message
          if (timeSinceLastMQTT < 2.0 && lastUpdateTime > 0) {
              // Linear interpolation between last and target
              const interpolationProgress = Math.min(timeSinceLastMQTT * MQTT_RATE, 1.0);
              const currentMovement = lastMovement + (targetMovement - lastMovement) * interpolationProgress;

              // Apply exponential smoothing
              const alpha = thereminConfig.smoothingFactor;
              smoothedMovement = alpha * currentMovement + (1 - alpha) * smoothedMovement;

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
          } else if (lastUpdateTime > 0) {
              // Use last known value if MQTT is stale, but still apply smoothing
              const alpha = thereminConfig.smoothingFactor;
              smoothedMovement = alpha * targetMovement + (1 - alpha) * smoothedMovement;
              const targetFreq = mapMovementToFrequency(smoothedMovement);
              const finalFreq = processFrequency(targetFreq, 0);
              if (thereminConfig.enabled) {
                  updateFrequency(finalFreq);
              }
              // Always update tremolo if enabled (even if theremin is disabled, to keep it synced)
              if (featureModConfig.tremoloEnabled) {
                  updateTremolo();
              }
              document.getElementById('movementValue').textContent = smoothedMovement.toFixed(3);
          } else {
              // No MQTT data yet, but still update tremolo if enabled
              if (featureModConfig.tremoloEnabled) {
                  updateTremolo();
              }
          }

          setTimeout(interpolate, updateInterval);
      }

      interpolate();
  }

  // MQTT Functions
  function toggleConnection() {
      if (client && client.connected) {
          disconnect();
      } else {
          connect();
      }
  }

  function connect() {
      const topic = document.getElementById('topic').value;
      client = ToolPage.connectMqtt({
          clientPrefix: 'theremin',
          subscription: topic,
          onStatus: updateStatus,
          onMessage: (_topic, message) => {
              try {
                  handleMessage(JSON.parse(message.toString()));
              } catch (error) {
                  console.error('Error parsing message:', error);
              }
          },
          onSubscribed: () => {
              resumeAudio();
              startInterpolationLoop();
          }
      });
  }


  function disconnect() {
      client = ToolPage.disconnectMqtt(client, updateStatus);
  }


  function updateStatus(connected) {
      ToolPage.setConnectionStatus(connected);
      document.getElementById('clearBtn').hidden = !connected;
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
      currentFeatures = null;

      // Reset audio if initialized
      if (oscillator && isAudioInitialized) {
          oscillator.frequency.setTargetAtTime(thereminConfig.baseFrequency, audioContext.currentTime, 0.01);
      }
  }

  // Initialize on load
  window.addEventListener('load', () => {
      updateTransportWarning();
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
  });
