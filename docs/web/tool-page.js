/* Shared behavior for ESPectre browser tools. */

(function () {
    'use strict';

    const localCommands = {
        monitor: './espectre ui mqtt',
        theremin: './espectre ui theremin'
    };

    function byId(id) {
        return document.getElementById(id);
    }

    function toggleSection(sectionId) {
        const content = byId(`${sectionId}Content`);
        const arrow = byId(`${sectionId}Arrow`);
        if (!content) return;

        const collapsed = content.classList.toggle('collapsed');
        if (arrow) arrow.classList.toggle('rotate', collapsed);

        const trigger = arrow && arrow.closest('.collapsible-header');
        if (trigger) trigger.setAttribute('aria-expanded', String(!collapsed));
    }

    function updateTransportWarning() {
        const protocol = byId('protocol');
        const warning = byId('insecureWsWarning');
        if (!protocol || !warning) return;
        warning.classList.toggle(
            'visible',
            window.location.protocol === 'https:' && protocol.value === 'ws'
        );
    }

    function websocketConnectionError(error) {
        const protocol = byId('protocol');
        if (protocol && window.location.protocol === 'https:' && protocol.value === 'ws') {
            const command = localCommands[document.body.dataset.tool] || './espectre ui';
            return `The browser may have blocked ws:// as mixed content. Try ${command} locally, or configure a wss:// listener.`;
        }
        return error && error.message
            ? error.message
            : 'Unable to reach the MQTT WebSocket endpoint.';
    }

    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.setAttribute('role', type === 'error' ? 'alert' : 'status');
        notification.textContent = message;
        document.body.appendChild(notification);

        window.setTimeout(() => {
            notification.classList.add('leaving');
            window.setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    function setConnectionStatus(connected) {
        const indicator = byId('statusIndicator');
        const statusText = byId('statusText');
        const connectButton = byId('connectBtn');
        if (!indicator || !statusText || !connectButton) return;

        indicator.classList.toggle('connected', connected);
        indicator.classList.toggle('disconnected', !connected);
        statusText.textContent = connected ? 'Connected' : 'Disconnected';
        connectButton.textContent = connected ? 'Disconnect' : 'Connect';
        connectButton.classList.toggle('btn-primary', !connected);
        connectButton.classList.toggle('btn-danger', connected);
        setHeaderConnectionStatus(connected);
    }

    function setHeaderConnectionStatus(connected) {
        const headerButton = byId('header-connect-action');
        if (!headerButton) return;

        const transport = document.body.dataset.tool === 'monitor' ? 'MQTT' : 'Bluetooth';
        const action = connected ? 'Disconnect from' : 'Connect to';
        const label = `${action} ${transport}`;
        headerButton.classList.toggle('connected', connected);
        headerButton.title = label;
        headerButton.setAttribute('aria-label', label);
    }

    function mqttConnectionOptions(clientPrefix) {
        const protocol = byId('protocol');
        const broker = byId('broker');
        const port = byId('port');
        const path = byId('wsPath');
        const username = byId('username');
        const password = byId('password');

        if (!protocol || !broker || !port || !path || !broker.value || !port.value) {
            window.alert('Please fill in all required fields.');
            return null;
        }

        const normalizedPath = (path.value.trim() || '/mqtt').replace(/^\/?/, '/');
        const options = {
            clean: true,
            connectTimeout: 4000,
            clientId: `espectre_${clientPrefix}_${Math.random().toString(16).slice(2, 10)}`
        };
        if (username && username.value) options.username = username.value;
        if (password && password.value) options.password = password.value;

        return {
            options,
            url: `${protocol.value}://${broker.value}:${port.value}${normalizedPath}`
        };
    }

    function connectMqtt({ clientPrefix, subscription, onMessage, onStatus, onSubscribed }) {
        if (typeof window.mqtt === 'undefined') {
            window.alert('MQTT.js failed to load. Check the browser network policy and reload the page.');
            return null;
        }
        if (!subscription) {
            window.alert('Please fill in all required fields.');
            return null;
        }

        const connection = mqttConnectionOptions(clientPrefix);
        if (!connection) return null;

        try {
            const client = window.mqtt.connect(connection.url, connection.options);
            client.on('connect', () => {
                onStatus(true);
                const config = byId('configContent');
                const arrow = byId('configArrow');
                if (config) config.classList.add('collapsed');
                if (arrow) arrow.classList.add('rotate');
                const trigger = arrow && arrow.closest('.collapsible-header');
                if (trigger) trigger.setAttribute('aria-expanded', 'false');

                client.subscribe(subscription, (error) => {
                    if (error) {
                        console.error('Subscribe error:', error);
                        window.alert('Error subscribing to topic.');
                        return;
                    }
                    if (onSubscribed) onSubscribed(client);
                });
            });
            client.on('message', onMessage);
            client.on('error', (error) => {
                console.error('Connection error:', error);
                window.alert(`Connection error: ${websocketConnectionError(error)}`);
                onStatus(false);
            });
            client.on('close', () => onStatus(false));
            return client;
        } catch (error) {
            console.error('Connection error:', error);
            window.alert(`Connection error: ${websocketConnectionError(error)}`);
            return null;
        }
    }

    function disconnectMqtt(client, onStatus) {
        if (client) client.end();
        onStatus(false);
        return null;
    }

    function initCollapsibles() {
        document.querySelectorAll('.collapsible-header').forEach((trigger) => {
            if (trigger.tagName !== 'BUTTON') {
                trigger.setAttribute('role', 'button');
                trigger.setAttribute('tabindex', '0');
            }

            const content = trigger.nextElementSibling;
            const expanded = !content || !content.classList.contains('collapsed');
            trigger.setAttribute('aria-expanded', String(expanded));

            trigger.addEventListener('keydown', (event) => {
                if (event.key !== 'Enter' && event.key !== ' ') return;
                event.preventDefault();
                trigger.click();
            });
        });
    }

    function initShell(page) {
        document.body.dataset.tool = page;
        loadHeader({ page });
        loadFooter();
        initCollapsibles();
    }

    window.ToolPage = {
        connectMqtt,
        disconnectMqtt,
        initShell,
        setConnectionStatus,
        setHeaderConnectionStatus,
        showNotification,
        toggleSection,
        updateTransportWarning,
        websocketConnectionError
    };

    // Compatibility for existing page handlers while their domain logic stays local.
    window.toggleConfig = () => toggleSection('config');
    window.toggleSection = toggleSection;
    window.toggleSubSection = toggleSection;
    window.showNotification = showNotification;
    window.updateTransportWarning = updateTransportWarning;
    window.websocketConnectionError = websocketConnectionError;

    document.addEventListener('DOMContentLoaded', () => {
        const page = document.body.dataset.tool;
        if (page) initShell(page);
    });
}());
