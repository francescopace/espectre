let client = null;
  let chart = null;
  let maxDataPoints = 60;
  let isPaused = false;
  let pendingData = [];
  let currentDeviceId = null;
  let lastOtaState = null;
  let lastTrackedDeviceProfile = null;
  let capabilityState = {
      supportsInfo: false,
      supportsRuntimeThreshold: false,
      supportsRuntimeDetector: false,
      supportsOta: false,
      supportsStats: false
  };
  const thresholdUiMin = 0.0;
  const thresholdUiMax = 10.0;
  const thresholdDragTolerancePx = 14;
  let isThresholdDragging = false;
  let draggedThresholdValue = null;
  let currentThresholdValue = 1.0;

  const chartData = {
      labels: [],
      movement: [],
      threshold: []
  };

  function showConnectionStatus(message, type = '') {
      const status = document.getElementById('connectionStatus');
      if (!status) return;
      status.textContent = message;
      status.className = `connection-status ${type}`.trim();
  }

  function setChartHint(message) {
      const hint = document.getElementById('chartHint');
      if (hint) hint.textContent = message;
  }

  function setPauseButtonState(paused) {
      const pauseBtn = document.getElementById('pauseBtn');
      if (!pauseBtn) return;
      pauseBtn.innerHTML = paused
          ? '<i class="fas fa-play" aria-hidden="true"></i> Resume'
          : '<i class="fas fa-pause" aria-hidden="true"></i> Pause';
      pauseBtn.classList.toggle('btn-secondary', !paused);
      pauseBtn.classList.toggle('btn-success', paused);
  }

  function setChartTitleState(paused, unavailable = false) {
      const chartTitle = document.getElementById('chartTitle');
      if (!chartTitle) return;
      if (unavailable) {
          chartTitle.innerHTML = '<i class="fas fa-triangle-exclamation" aria-hidden="true"></i> Chart unavailable';
          return;
      }
      chartTitle.innerHTML = paused
          ? '<i class="fas fa-chart-line" aria-hidden="true"></i> Real-Time Chart <span class="chart-paused-badge">Paused</span>'
          : '<i class="fas fa-chart-line" aria-hidden="true"></i> Real-Time Chart';
  }

  function openConnectModal() {
      const modal = document.getElementById('mqttConnectModal');
      if (!modal) return;
      modal.hidden = false;
      document.body.classList.add('monitor-modal-open');
      updateTransportWarning();
      const broker = document.getElementById('broker');
      if (broker && !broker.disabled) {
          window.setTimeout(() => broker.focus(), 50);
      }
  }

  function closeConnectModal() {
      const modal = document.getElementById('mqttConnectModal');
      if (!modal) return;
      modal.hidden = true;
      document.body.classList.remove('monitor-modal-open');
  }

  function updateConnectionUi(connected) {
      const connectionPre = document.getElementById('connectionPre');
      const connectionReady = document.getElementById('connectionReady');
      const workspace = document.getElementById('monitorWorkspace');
      if (connectionPre) connectionPre.hidden = connected;
      if (connectionReady) connectionReady.hidden = !connected;
      if (workspace) workspace.hidden = !connected;
      document.body.classList.toggle('monitor-connected', connected);
      ToolPage.setHeaderConnectionStatus(connected);

      if (connected) {
          const broker = document.getElementById('broker');
          const label = document.getElementById('connectedBrokerLabel');
          const brokerLabel = broker && broker.value ? broker.value : 'MQTT broker';
          if (label) label.textContent = `${brokerLabel} Connected`;
          closeConnectModal();
          setChartHint('Streaming live movement. Drag the threshold handle when supported.');
          if (chart) {
              window.requestAnimationFrame(() => {
                  chart.resize();
                  updateThresholdDragHintPosition();
              });
          }
      } else {
          showConnectionStatus('', '');
          setChartHint('Waiting for live telemetry…');
      }
  }

  function setConnectBusy(busy) {
      const submit = document.getElementById('mqttConnectSubmitBtn');
      const connectBtn = document.getElementById('connectBtn');
      if (submit) {
          submit.disabled = busy;
          submit.textContent = busy ? 'Connecting…' : 'Connect';
      }
      if (connectBtn) connectBtn.disabled = busy;
  }

  // Initialize Chart
  function initChart() {
      if (typeof Chart === 'undefined') {
          setChartTitleState(false, true);
          setChartHint('Chart.js failed to load. Check the browser network policy and reload the page.');
          showNotification('Chart.js failed to load. Check the browser network policy and reload the page.', 'error');
          return;
      }
      const ctx = document.getElementById('motionChart').getContext('2d');
      const palette = getComputedStyle(document.documentElement);
      const accentColor = palette.getPropertyValue('--accent').trim();
      const accentGlow = palette.getPropertyValue('--accent-glow').trim();
      const thresholdColor = palette.getPropertyValue('--text-dim').trim() || palette.getPropertyValue('--text-secondary').trim();
      const gridColor = palette.getPropertyValue('--border').trim();
      const tickColor = palette.getPropertyValue('--text-dim').trim() || '#b8b8c7';
      chart = new Chart(ctx, {
          type: 'line',
          data: {
              labels: chartData.labels,
              datasets: [
                  {
                      label: 'Movement',
                      data: chartData.movement,
                      borderColor: accentColor,
                      backgroundColor: accentGlow,
                      borderWidth: 2.5,
                      pointRadius: 0,
                      pointHoverRadius: 4,
                      tension: 0.35,
                      fill: true,
                      yAxisID: 'y'
                  },
                  {
                      label: 'Threshold',
                      data: chartData.threshold,
                      borderColor: thresholdColor,
                      backgroundColor: 'transparent',
                      borderWidth: 2,
                      borderDash: [6, 5],
                      pointRadius: 0,
                      tension: 0,
                      fill: false,
                      yAxisID: 'y'
                  }
              ]
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              interaction: {
                  mode: 'index',
                  intersect: false
              },
              animation: {
                  duration: 220
              },
              scales: {
                  y: {
                      type: 'linear',
                      display: true,
                      position: 'left',
                      beginAtZero: true,
                      border: {
                          display: false
                      },
                      grid: {
                          color: gridColor
                      },
                      ticks: {
                          color: tickColor,
                          maxTicksLimit: 6
                      },
                      title: {
                          display: true,
                          text: 'Movement / Threshold',
                          color: tickColor
                      }
                  },
                  x: {
                      border: {
                          display: false
                      },
                      grid: {
                          display: false
                      },
                      ticks: {
                          color: tickColor,
                          maxTicksLimit: 8,
                          maxRotation: 0
                      },
                      title: {
                          display: false
                      }
                  }
              },
              plugins: {
                  legend: {
                      display: true,
                      position: 'top',
                      align: 'end',
                      labels: {
                          color: tickColor,
                          boxWidth: 12,
                          usePointStyle: true,
                          pointStyle: 'line',
                          padding: 16
                      }
                  },
                  tooltip: {
                      mode: 'index',
                      intersect: false,
                      backgroundColor: palette.getPropertyValue('--bg-card').trim() || '#1a1a24',
                      titleColor: tickColor,
                      bodyColor: palette.getPropertyValue('--text-primary').trim() || '#ffffff',
                      borderColor: gridColor,
                      borderWidth: 1,
                      padding: 10
                  }
              }
          }
      });
      setupThresholdDrag();
  }

  function togglePause() {
      isPaused = !isPaused;
      setPauseButtonState(isPaused);
      setChartTitleState(isPaused);
      setChartHint(isPaused
          ? 'Chart paused. Incoming samples are queued until you resume.'
          : 'Drag the threshold handle on the right when the device supports runtime threshold updates.');

      if (!isPaused && pendingData.length > 0) {
          pendingData.forEach(data => addDataToChart(data));
          pendingData = [];
      }
  }

  function toggleConnection() {
      if (client && client.connected) {
          disconnect();
      } else {
          openConnectModal();
      }
  }

  function connect() {
      const baseTopic = document.getElementById('baseTopic').value;
      const protocolTopics = resolveProtocolTopics(baseTopic);
      if (!protocolTopics) {
          alert('Set a device-specific base topic like espectre/v1/devices/0x00007c2c6742bbac. Wildcards and suffixes are not supported.');
          return;
      }

      setConnectBusy(true);
      showConnectionStatus('Connecting to MQTT broker…', 'connecting');
      setChartHint('Connecting to the MQTT broker…');
      client = ToolPage.connectMqtt({
          clientPrefix: 'monitor',
          subscription: protocolTopics.all,
          onStatus: updateStatus,
          onMessage: (topic, message) => handleMessage(message.toString(), topic),
          onSubscribed: () => {
              setCurrentDeviceId(protocolTopics.deviceId);
              setChartHint('Subscribed. Waiting for live telemetry…');
              window.setTimeout(requestInfo, 500);
          }
      });
      if (!client) {
          setConnectBusy(false);
          showConnectionStatus('Unable to start MQTT connection.', 'error');
          setChartHint('Waiting for live telemetry…');
      }
  }


  function disconnect() {
      closeConnectModal();
      client = ToolPage.disconnectMqtt(client, updateStatus);
  }


  function setMqttConfigLocked(locked) {
      const fieldIds = ['protocol', 'broker', 'port', 'wsPath', 'baseTopic', 'username', 'password'];
      fieldIds.forEach((id) => {
          const field = document.getElementById(id);
          if (!field) return;
          field.readOnly = locked;
          field.disabled = locked;
      });
  }

  function updateStatus(connected) {
      setMqttConfigLocked(connected);
      setConnectBusy(false);
      updateConnectionUi(connected);
      if (!connected) {
          capabilityState = {
              supportsInfo: false,
              supportsRuntimeThreshold: false,
              supportsRuntimeDetector: false,
              supportsOta: false,
              supportsStats: false
          };
          applyCapabilityState();
      }
  }



  function addDataToChart(data) {
      const now = new Date();
      const timeLabel = now.toLocaleTimeString('en-US');
      const thresholdValue = isThresholdDragging && draggedThresholdValue !== null
          ? draggedThresholdValue
          : (data.threshold || 0);

      chartData.labels.push(timeLabel);
      chartData.movement.push(data.movement || 0);
      chartData.threshold.push(thresholdValue);

      if (chartData.labels.length > maxDataPoints) {
          chartData.labels.shift();
          chartData.movement.shift();
          chartData.threshold.shift();
      }

      chart.update('none');
      updateThresholdDragHintPosition();
  }

  function updateMetrics(data) {
      const stateCard = document.getElementById('stateCard');
      const stateValue = document.getElementById('stateValue');
      stateValue.textContent = data.state === 'motion' ? 'MOTION' : 'IDLE';

      if (data.state === 'motion') {
          stateCard.classList.remove('idle');
          stateCard.classList.add('motion');
      } else {
          stateCard.classList.remove('motion');
          stateCard.classList.add('idle');
      }

      const movement = Number(data.movement);
      const threshold = Number(data.threshold);
      const hasMovement = Number.isFinite(movement);
      const hasThreshold = Number.isFinite(threshold) && threshold > 0;
      document.getElementById('movementValue').textContent =
          hasMovement && hasThreshold ? `${((movement / threshold) * 100).toFixed(1)}%` : '-';
      document.getElementById('thresholdValue').textContent =
          Number.isFinite(threshold) ? threshold.toFixed(6) : '-';
      if (data.detector) {
          setInfoField('detectorAlgorithm', String(data.detector).toUpperCase());
      }
      if (data.threshold !== undefined) {
          setThresholdControls(data.threshold);
      }
  }

  function setInfoField(id, value) {
      const target = document.getElementById(id);
      if (!target) return;
      const row = target.closest('.info-field');
      const hasValue = value !== undefined && value !== null && String(value).trim() !== '' && value !== '-';
      if (row) {
          row.hidden = !hasValue;
      }
      target.textContent = hasValue ? value : '-';
  }

  function setThresholdControls(threshold, force = false) {
      const parsed = Number(threshold);
      if (!Number.isFinite(parsed)) return;
      const editingThreshold = isThresholdDragging;
      if (editingThreshold && !force) return;
      currentThresholdValue = parsed;
      document.getElementById('thresholdValue').textContent = parsed.toFixed(6);
  }

  function clampThresholdValue(value) {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) {
          return null;
      }
      return Math.min(thresholdUiMax, Math.max(thresholdUiMin, parsed));
  }

  function getCurrentThresholdValue() {
      const clamped = clampThresholdValue(currentThresholdValue);
      if (clamped !== null) {
          return clamped;
      }
      const latestThreshold = chartData.threshold.length > 0 ? chartData.threshold[chartData.threshold.length - 1] : null;
      const latestClamped = clampThresholdValue(latestThreshold);
      return latestClamped !== null ? latestClamped : 1.0;
  }

  function setChartThresholdLine(threshold) {
      const clamped = clampThresholdValue(threshold);
      if (clamped === null) {
          return;
      }
      const points = Math.max(chartData.labels.length, 1);
      chartData.threshold = Array(points).fill(clamped);
      if (chart && chart.data && chart.data.datasets && chart.data.datasets[1]) {
          chart.data.datasets[1].data = chartData.threshold;
          chart.update('none');
          updateThresholdDragHintPosition(clamped);
      }
  }

  function thresholdFromPointerEvent(event) {
      if (!chart || !chart.scales || !chart.scales.y) {
          return null;
      }
      const yScale = chart.scales.y;
      const canvasRect = chart.canvas.getBoundingClientRect();
      const y = event.clientY - canvasRect.top;
      return clampThresholdValue(yScale.getValueForPixel(y));
  }

  function canStartThresholdDrag() {
      return !!chart && !!chart.scales && !!chart.scales.y && capabilityState.supportsRuntimeThreshold === true;
  }

  function commitThresholdValue(threshold) {
      const clamped = clampThresholdValue(threshold);
      if (clamped === null || !capabilityState.supportsRuntimeThreshold) {
          return;
      }
      setThresholdControls(clamped, true);
      setChartThresholdLine(clamped);
      sendCommand('set_threshold', { threshold: clamped });
  }

  function updateThresholdDragHintPosition(threshold = null) {
      const hint = document.getElementById('thresholdDragHint');
      if (!hint || !chart || !chart.scales || !chart.scales.y || !chart.canvas) {
          return;
      }
      const resolvedThreshold = clampThresholdValue(
          threshold !== null ? threshold : (draggedThresholdValue !== null ? draggedThresholdValue : getCurrentThresholdValue())
      );
      if (resolvedThreshold === null) {
          return;
      }
      const canvasTop = chart.canvas.offsetTop || 0;
      const canvasHeight = chart.canvas.clientHeight || 0;
      const rawTop = canvasTop + chart.scales.y.getPixelForValue(resolvedThreshold);
      const clampedTop = Math.min(
          Math.max(rawTop, canvasTop + 20),
          canvasTop + Math.max(20, canvasHeight - 20)
      );
      hint.style.top = `${clampedTop}px`;
  }

  function setThresholdDragHintState(enabled, active = false) {
      const hint = document.getElementById('thresholdDragHint');
      if (!hint) {
          return;
      }
      hint.hidden = !enabled;
      hint.classList.toggle('active', enabled && active);
      if (enabled) {
          updateThresholdDragHintPosition();
      }
  }

  function setupThresholdDrag() {
      if (!chart || !chart.canvas) {
          return;
      }
      const hint = document.getElementById('thresholdDragHint');
      if (!hint) {
          return;
      }

      hint.addEventListener('pointerdown', (event) => {
          if (!canStartThresholdDrag()) {
              return;
          }
          const threshold = thresholdFromPointerEvent(event);
          if (threshold === null) {
              return;
          }
          isThresholdDragging = true;
          draggedThresholdValue = threshold;
          setThresholdControls(threshold, true);
          setChartThresholdLine(threshold);
          setThresholdDragHintState(true, true);
          hint.setPointerCapture(event.pointerId);
          event.preventDefault();
      });

      hint.addEventListener('pointermove', (event) => {
          if (!isThresholdDragging) {
              return;
          }
          const threshold = thresholdFromPointerEvent(event);
          if (threshold === null) {
              return;
          }
          draggedThresholdValue = threshold;
          setThresholdControls(threshold, true);
          setChartThresholdLine(threshold);
          event.preventDefault();
      });

      const finishDrag = (event) => {
          if (!isThresholdDragging) {
              return;
          }
          const threshold = draggedThresholdValue;
          isThresholdDragging = false;
          draggedThresholdValue = null;
          if (event) {
              hint.releasePointerCapture(event.pointerId);
          }
          setThresholdDragHintState(capabilityState.supportsRuntimeThreshold === true, false);
          if (threshold !== null) {
              commitThresholdValue(threshold);
          }
      };

      hint.addEventListener('pointerup', finishDrag);
      hint.addEventListener('pointercancel', finishDrag);
  }

  function setOtaControlsVisibility(supported) {
      const enabled = supported === true;
      const card = document.getElementById('firmwareUpgradeCard');
      if (card) {
          card.hidden = !enabled;
      }
      document.querySelectorAll('.ota-action').forEach((button) => {
          button.hidden = !enabled;
      });
  }

  function setThresholdControlAvailability(supported) {
      const enabled = supported === true;
      setThresholdDragHintState(enabled, false);
  }

  function setStatsControlsVisibility(supported) {
      const enabled = supported === true;
      const statsBtn = document.getElementById('statsBtn');
      if (statsBtn) {
          statsBtn.hidden = !enabled;
      }
  }

  function applyCapabilityState() {
      const configurationCard = document.getElementById('deviceConfigurationCard');
      if (configurationCard) {
          configurationCard.hidden = capabilityState.supportsInfo !== true;
      }
      setThresholdControlAvailability(capabilityState.supportsRuntimeThreshold);
      setOtaControlsVisibility(capabilityState.supportsOta);
      setStatsControlsVisibility(capabilityState.supportsStats);
  }

  function trimTopic(topic) {
      return (topic || '').trim().replace(/\/+$/, '');
  }

  function extractDeviceIdFromBaseTopic(baseTopic) {
      const parts = trimTopic(baseTopic).split('/').filter(Boolean);
      const devicesIndex = parts.indexOf('devices');
      if (devicesIndex >= 0 && parts.length === devicesIndex + 2) {
          const candidate = parts[devicesIndex + 1];
          if (candidate && !candidate.includes('+') && !candidate.includes('#') && !candidate.includes('<') && !candidate.includes('>')) {
              return candidate;
          }
      }
      return null;
  }

  function resolveProtocolTopics(inputBaseTopic) {
      const base = trimTopic(inputBaseTopic);
      const deviceId = extractDeviceIdFromBaseTopic(base);
      if (!base || !deviceId) {
          return null;
      }

      return {
          base,
          deviceId,
          all: `${base}/#`,
          info: `${base}/info`,
          status: `${base}/status`,
          otaState: `${base}/ota/state`,
          commandsRequest: `${base}/commands/request`,
          commandsAccepted: `${base}/commands/accepted`,
          commandsRejected: `${base}/commands/rejected`
      };
  }

  function setCurrentDeviceId(deviceId) {
      if (!deviceId) return;
      currentDeviceId = deviceId;
      setInfoField('protocolDeviceId', deviceId);
  }

  function commandTopicFor(inputBaseTopic) {
      const topics = resolveProtocolTopics(inputBaseTopic);
      if (!topics) {
          return null;
      }
      return topics.commandsRequest;
  }

  // MQTT Command Functions
  function sendCommand(cmd, params = {}) {
      if (!client || !client.connected) {
          showNotification('Not connected to MQTT broker', 'error');
          return;
      }

      const baseTopic = document.getElementById('baseTopic').value;
      const cmdTopic = commandTopicFor(baseTopic);
      if (!cmdTopic) {
          showNotification('Set a device-specific base topic before sending commands', 'error');
          return;
      }
      const message = JSON.stringify({
          protocol_version: '1.0',
          command_id: 'web-' + Date.now(),
          command: cmd,
          ...params
      });

      client.publish(cmdTopic, message, (err) => {
          if (err) {
              showNotification(`Failed to send command: ${err.message}`, 'error');
          }
          trackEvent('monitor_command', {
              command: cmd,
              result: err ? 'failure' : 'success'
          });
          // Removed "Command sent" notification for cleaner real-time updates
          // Only device responses will show notifications now
      });
  }


  // System Info Functions
  function requestInfo() {
      sendCommand('info');
  }

  function requestStats() {
      if (!capabilityState.supportsStats) {
          showNotification('Statistics are not supported by this frontend', 'error');
          return;
      }
      sendCommand('stats');
  }

  function requestOtaStatus() {
      sendCommand('ota_status');
  }

  function requestOtaCheck() {
      sendCommand('ota_check');
  }

  function requestOtaStart() {
      sendCommand('ota_start');
  }

  // Section Toggle Function
  // SubSection Toggle Function
  // Notification Function
  // Enhanced Message Handler for Info/Stats and regular data
  function normalizeEspectreProtocolMessage(data, topic) {
      if (!data || data.protocol_version === undefined) {
          return data;
      }
      if (data.device_id) {
          setCurrentDeviceId(data.device_id);
      } else if (topic && topic.startsWith('espectre/v1/devices/')) {
          setCurrentDeviceId(topic.split('/')[3]);
      }
      if (data.accepted !== undefined && data.message) {
          return {
              response: data.message,
              responseType: data.accepted ? 'success' : 'error'
          };
      }
      const isOtaTopic = typeof topic === 'string' && topic.endsWith('/ota/state');
      if (isOtaTopic || data.current_version !== undefined || data.update_available !== undefined) {
          return {
              ...data,
              ota_state: true
          };
      }
      if (data.motion_state || data.movement_score !== undefined) {
          return {
              ...data,
              state: data.motion_state,
              movement: data.movement_score,
              threshold: data.threshold
          };
      }
      if (data.firmware_version || data.chip || data.frontend) {
          populateUIFromConfig(data, { notify: false });
          return { ...data, info_only: true };
      }
      return data;
  }

  // Enhanced Message Handler for Info/Stats and regular data
  function handleMessage(message, topic = '') {
      try {
          let data = JSON.parse(message);
          data = normalizeEspectreProtocolMessage(data, topic);

          // Log all messages to console for debugging
          console.log('Received message:', data);

          // Check if this is an info response
          if (data.info_only || data.network || data.mqtt || data.device || data.detection) {
              console.log('Detected INFO response');
              if (!data.info_only) {
                  populateUIFromConfig(data);
              }
              return;
          }

          if (data.ota_state) {
              console.log('Detected OTA state');
              updateOtaStatus(data);
              return;
          }

          // Check if this is a stats response
          const isStatsTopic = typeof topic === 'string' && topic.endsWith('/stats');
          if (isStatsTopic || data.uptime !== undefined ||
              data.free_memory_kb !== undefined || data.loop_time_ms !== undefined) {
              console.log('Detected STATS response');
              displayStats(normalizeStatsPayload(data));
              return;
          }

          // Check if this is a simple response message (has "response" field)
          if (data.response && typeof data.response === 'string') {
              console.log('Detected command response');
              showNotification(data.response, data.responseType || 'info');
              return;
          }

          // Regular data message (has movement, threshold, state)
          if (data.movement !== undefined || data.threshold !== undefined || data.state) {
              const now = new Date();
              document.getElementById('lastUpdate').textContent = now.toLocaleTimeString(undefined, { hour12: false });

              updateMetrics(data);

              if (isPaused) {
                  pendingData.push(data);
              } else {
                  addDataToChart(data);
              }
          }

      } catch (err) {
          console.error('Error parsing message:', err);
      }
  }

  function normalizeStatsPayload(stats) {
      if (!stats || typeof stats !== 'object') {
          return {};
      }

      return {
          ...stats,
          uptime: stats.uptime,
          free_memory_kb: stats.free_memory_kb,
          loop_time_ms: stats.loop_time_ms
      };
  }

  function formatMetric(value, digits = 3, suffix = '') {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) {
          return 'N/A';
      }
      return `${parsed.toFixed(digits)}${suffix}`;
  }

  // Format uptime to human readable format
  function formatUptime(uptime) {
      if (uptime === undefined || uptime === null || uptime === '') return 'N/A';

      // If uptime is already a formatted string, return it as is
      if (typeof uptime === 'string') {
          return uptime;
      }

      // If uptime is a number (seconds), format it
      if (typeof uptime === 'number') {
          const hours = Math.floor(uptime / 3600);
          const minutes = Math.floor((uptime % 3600) / 60);
          const secs = uptime % 60;
          return `${hours}h ${minutes}m ${secs}s`;
      }

      return 'N/A';
  }

  // Refresh stats in modal
  function refreshStats() {
      const reloadBtn = document.getElementById('statsReloadBtn');
      if (reloadBtn) {
          reloadBtn.classList.add('spinning');
          reloadBtn.disabled = true;
      }

      // Send stats command
      sendCommand('stats');

      // Re-enable button after a delay
      setTimeout(() => {
          if (reloadBtn) {
              reloadBtn.classList.remove('spinning');
              reloadBtn.disabled = false;
          }
      }, 1000);
  }

  // Display Stats in modal
  function displayStats(stats) {
      stats = normalizeStatsPayload(stats);

      // Check if modal already exists
      let overlay = document.querySelector('.modal-overlay');
      let isUpdate = false;

      if (overlay) {
          // Modal exists, just update the content
          isUpdate = true;
      } else {
          // Create new modal overlay
          overlay = document.createElement('div');
          overlay.className = 'modal-overlay';
          overlay.onclick = (e) => {
              if (e.target === overlay) closeModal();
          };
      }

      // Create or get modal content
      let modal = overlay.querySelector('.modal-content');
      if (!modal) {
          modal = document.createElement('div');
          modal.className = 'modal-content';
      } else {
          modal.innerHTML = ''; // Clear existing content
      }

      // Modal header with reload button
      const header = document.createElement('div');
      header.className = 'modal-header';
      header.innerHTML = `
          <h3><i class="fas fa-chart-simple" aria-hidden="true"></i> Statistics</h3>
          <div class="modal-header-actions">
              <button class="modal-reload" id="statsReloadBtn" onclick="refreshStats()" title="Refresh statistics" type="button" aria-label="Refresh statistics">
                  <i class="fas fa-rotate-right" aria-hidden="true"></i>
              </button>
          </div>
      `;

      // Modal body with organized sections
      const body = document.createElement('div');
      body.className = 'modal-body';

      let html = '';

      html += `<div class="stat-list">`;
      html += `<div class="stat-line">`;
      html += `<span class="stat-label"><i class="fas fa-clock" aria-hidden="true"></i> Uptime</span>`;
      html += `<span class="stat-value">${formatUptime(stats.uptime)}</span>`;
      html += `</div>`;
      if (stats.free_memory_kb !== undefined) {
          html += `<div class="stat-line">`;
          html += `<span class="stat-label"><i class="fas fa-memory" aria-hidden="true"></i> Free Memory</span>`;
          html += `<span class="stat-value highlight">${formatMetric(stats.free_memory_kb, 1, ' KB')}</span>`;
          html += `</div>`;
      }
      if (stats.loop_time_ms !== undefined) {
          html += `<div class="stat-line">`;
          html += `<span class="stat-label"><i class="fas fa-bolt" aria-hidden="true"></i> Loop Time</span>`;
          html += `<span class="stat-value highlight">${formatMetric(stats.loop_time_ms, 2, ' ms')}</span>`;
          html += `</div>`;
      }
      html += `</div>`;

      body.innerHTML = html;

      modal.appendChild(header);
      modal.appendChild(body);

      if (!isUpdate) {
          overlay.appendChild(modal);
          document.body.appendChild(overlay);
      }
  }

  // Close modal
  function closeModal() {
      const overlay = document.querySelector('.modal-overlay');
      if (overlay) {
          overlay.style.animation = 'fadeIn 0.3s ease-out reverse';
          setTimeout(() => overlay.remove(), 300);
      }
  }

  // Populate UI controls from config data
  function populateUIFromConfig(config, options = {}) {
      console.log('Populating UI from config:', config);

      if (config.chip && config.frontend) {
          const frontend = String(config.frontend).toLowerCase();
          const chip = String(config.chip).toLowerCase();
          const profile = `${frontend}:${chip}`;
          if (profile !== lastTrackedDeviceProfile) {
              trackEvent('device_profile', {
                  tool_name: 'monitor',
                  frontend,
                  chip
              });
              lastTrackedDeviceProfile = profile;
          }
      }

      [
          'deviceType',
          'deviceIP',
          'deviceMAC',
          'wifiChannel',
          'detectorAlgorithm',
          'protocolDeviceId',
          'protocolDeviceName',
          'protocolDeviceLabel',
          'firmwareVersion'
      ].forEach((id) => setInfoField(id, null));

      // Update Device & Network Information
      setInfoField('deviceType', config.chip || (config.device && config.device.type));
      setInfoField('detectorAlgorithm', config.frontend);
      setInfoField('firmwareVersion', config.firmware_version);
      setCurrentDeviceId(config.device_id);
      setInfoField('protocolDeviceName', config.device_name);
      setInfoField('protocolDeviceLabel', config.device_label);
      if (config.network) {
          setInfoField('deviceIP', config.network.ip_address);
          setInfoField('deviceMAC', config.network.mac_address);
          if (config.network.channel) {
              const primary = config.network.channel.primary || '-';
              const secondary = config.network.channel.secondary || 0;
              const channelText = secondary > 0 ? `${primary} + ${secondary}` : `${primary}`;
              setInfoField('wifiChannel', channelText);
          }
      }

      // Detection algorithm info
      if (config.detection) {
          if (config.detection.algorithm) {
              setInfoField('detectorAlgorithm', config.detection.algorithm.toUpperCase());
          }
      }

      capabilityState = {
          supportsInfo: config.supports_info === true,
          supportsRuntimeThreshold: config.supports_runtime_threshold === true,
          supportsRuntimeDetector: config.supports_runtime_detector === true,
          supportsOta: config.supports_ota === true,
          supportsStats: config.supports_stats === true
      };
      applyCapabilityState();

      if (config.firmware_version) {
          setInfoField('firmwareVersion', config.firmware_version);
      }

      if (options.notify !== false) {
          showNotification('Configuration loaded from device', 'success');
      }
  }

  function setOtaValue(id, value) {
      const target = document.getElementById(id);
      if (!target) return;
      const hasValue = value !== undefined && value !== null && String(value).trim() !== '' && value !== '-';
      const nextValue = hasValue ? value : '-';
      if ('value' in target) {
          target.value = nextValue;
      } else {
          target.textContent = nextValue;
          const row = target.closest('.info-field');
          if (row) {
              row.hidden = !hasValue;
          }
      }
      updateOtaSummaryVisibility();
  }

  function updateOtaSummaryVisibility() {
      const summaryBox = document.getElementById('otaSummaryBox');
      if (!summaryBox) {
          return;
      }
      const visibleRows = summaryBox.querySelectorAll('.info-field:not([hidden])');
      summaryBox.hidden = visibleRows.length === 0;
  }

  function updateOtaStatus(data) {
      const state = data.state || '-';
      setOtaValue('otaStateValue', state);
      if (data.current_version) {
          setInfoField('firmwareVersion', data.current_version);
      }
      setOtaValue('otaTargetVersionValue', data.target_version);
      setOtaValue(
          'otaUpdateAvailableValue',
          data.update_available === undefined ? '-' : (data.update_available ? 'Yes' : 'No')
      );
      setOtaValue('otaMessageValue', data.message);

      if (state && state !== '-' && state !== lastOtaState) {
          lastOtaState = state;
          const type = state === 'error' ? 'error' : (state === 'update_available' ? 'info' : 'success');
          const message = data.message ? `OTA: ${state} - ${data.message}` : `OTA: ${state}`;
          showNotification(message, type);
      }
  }

  // Debounce helper function
  function debounce(func, wait) {
      let timeout;
      return function executedFunction(...args) {
          const later = () => {
              clearTimeout(timeout);
              func(...args);
          };
          clearTimeout(timeout);
          timeout = setTimeout(later, wait);
      };
  }

  // Initialize on load
  window.addEventListener('load', () => {
      const connectButton = document.getElementById('connectBtn');
      const disconnectButton = document.getElementById('disconnectBtn');
      const submitButton = document.getElementById('mqttConnectSubmitBtn');
      const modal = document.getElementById('mqttConnectModal');

      if (connectButton) connectButton.addEventListener('click', toggleConnection);
      if (disconnectButton) disconnectButton.addEventListener('click', disconnect);
      if (submitButton) submitButton.addEventListener('click', connect);

      if (modal) {
          modal.querySelectorAll('[data-close-mqtt-modal]').forEach((el) => {
              el.addEventListener('click', closeConnectModal);
          });
      }

      document.addEventListener('keydown', (event) => {
          if (event.key !== 'Escape') return;
          const openModal = document.getElementById('mqttConnectModal');
          if (openModal && !openModal.hidden) closeConnectModal();
      });

      initChart();
      updateTransportWarning();
      applyCapabilityState();
      updateOtaSummaryVisibility();
      updateConnectionUi(false);
  });

  window.addEventListener('resize', () => {
      updateThresholdDragHintPosition();
  });
