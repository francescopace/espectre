/*
 * ESPectre - Cosmic Background (WebGL Shader)
 * 
 * Animated WiFi wave effect used as background across the site.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

(function() {
    const canvas = document.getElementById('cosmic-bg');
    if (!canvas) return;
    
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) return;

    const palette = getComputedStyle(document.documentElement);

    function readPaletteColor(property) {
        const value = palette.getPropertyValue(property).trim();
        const match = /^#([0-9a-f]{6})$/i.exec(value);
        if (!match) {
            console.warn(`cosmic-bg invalid palette color: ${property}`);
            return null;
        }

        const hex = match[1];
        return [0, 2, 4].map((offset) => parseInt(hex.slice(offset, offset + 2), 16) / 255);
    }

    const backgroundPrimary = readPaletteColor('--bg-primary');
    const backgroundSecondary = readPaletteColor('--bg-secondary');
    const accentPrimary = readPaletteColor('--accent');
    const accentSecondary = readPaletteColor('--accent-secondary');
    if (!backgroundPrimary || !backgroundSecondary || !accentPrimary || !accentSecondary) return;
    
    // Vertex shader
    const vsSource = `
        attribute vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    `;
    
    // Fragment shader - ESPectre cosmic waves
    const fsSource = `
        precision mediump float;
        uniform vec2 iResolution;
        uniform float iTime;
        uniform vec2 iMouse;
        uniform vec3 paletteBgPrimary;
        uniform vec3 paletteBgSecondary;
        uniform vec3 paletteAccentPrimary;
        uniform vec3 paletteAccentSecondary;
        
        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
        
        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));
            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }
        
        float fbm(vec2 p) {
            float value = 0.0;
            float amplitude = 0.5;
            for(int i = 0; i < 4; i++) {
                value += amplitude * noise(p);
                p *= 2.0;
                amplitude *= 0.5;
            }
            return value;
        }
        
        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;
            
            float time = iTime * 0.3;
            
            // Concentric waves emanating from center (like WiFi)
            float dist = length(p);
            float wave1 = sin(dist * 8.0 - time * 2.0) * 0.5;
            float wave2 = sin(dist * 12.0 - time * 2.5 + 1.0) * 0.3;
            float wave3 = sin(dist * 6.0 - time * 1.5 - 0.5) * 0.4;
            
            // Mouse interference - creates disturbance like motion detection
            vec2 mousePos = (iMouse / iResolution) * 2.0 - 1.0;
            mousePos.x *= iResolution.x / iResolution.y;
            float mouseDist = length(p - mousePos);
            float mouseWave = sin(mouseDist * 15.0 - time * 4.0) * exp(-mouseDist * 2.0) * 0.5;
            
            float waves = (wave1 + wave2 + wave3 + mouseWave) * 0.3;
            
            // Organic texture
            vec2 noisePos = p * 2.0 + vec2(time * 0.1, time * 0.05);
            float noiseValue = fbm(noisePos) * 0.3;
            
            float pattern = waves + noiseValue;
            
            // Derive the animated background from the shared CSS palette.
            vec3 color1 = mix(paletteBgPrimary, paletteBgSecondary, 0.35);
            vec3 color2 = paletteAccentPrimary;
            vec3 color3 = paletteAccentSecondary;
            vec3 color4 = mix(paletteBgSecondary, paletteAccentPrimary, 0.12);
            
            // Slowly sweep between the two brand colors while the wave field
            // controls their intensity independently.
            float auroraMix = 0.5 + 0.5 * sin(
                p.x * 1.4 + p.y * 0.8 - time * 0.45 + noiseValue * 4.0
            );
            vec3 auroraColor = mix(color2, color3, auroraMix);
            float waveIntensity = 0.10 + smoothstep(-0.35, 0.55, pattern) * 0.38;

            vec3 finalColor = mix(color1, color4, noiseValue * 0.8);
            finalColor += auroraColor * waveIntensity;
            
            // Subtle glow in center
            float glow = exp(-dist * 1.5) * 0.15;
            vec3 glowColor = mix(
                color2,
                color3,
                0.5 + 0.5 * sin(time * 0.35)
            );
            finalColor += glow * glowColor;
            
            // Vignette
            float vignette = 1.0 - length(uv - 0.5) * 1.0;
            vignette = smoothstep(0.0, 1.0, vignette);
            finalColor *= vignette;
            
            // Keep it subtle - this is a background
            finalColor *= 0.6;
            
            gl_FragColor = vec4(finalColor, 1.0);
        }
    `;
    
    function createShader(type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            console.warn('cosmic-bg shader compile failed:', gl.getShaderInfoLog(shader));
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }

    const vs = createShader(gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl.FRAGMENT_SHADER, fsSource);
    if (!vs || !fs) return;

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.warn('cosmic-bg program link failed:', gl.getProgramInfoLog(program));
        return;
    }
    gl.useProgram(program);
    
    // Fullscreen quad
    const vertices = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    
    const position = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    
    const iResolution = gl.getUniformLocation(program, 'iResolution');
    const iTime = gl.getUniformLocation(program, 'iTime');
    const iMouse = gl.getUniformLocation(program, 'iMouse');
    const paletteBgPrimary = gl.getUniformLocation(program, 'paletteBgPrimary');
    const paletteBgSecondary = gl.getUniformLocation(program, 'paletteBgSecondary');
    const paletteAccentPrimary = gl.getUniformLocation(program, 'paletteAccentPrimary');
    const paletteAccentSecondary = gl.getUniformLocation(program, 'paletteAccentSecondary');

    gl.uniform3fv(paletteBgPrimary, backgroundPrimary);
    gl.uniform3fv(paletteBgSecondary, backgroundSecondary);
    gl.uniform3fv(paletteAccentPrimary, accentPrimary);
    gl.uniform3fv(paletteAccentSecondary, accentSecondary);
    
    let mouseX = 0, mouseY = 0;
    
    // Only track mouse on non-touch devices (avoids interference on mobile)
    const isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    if (!isMobile) {
        document.addEventListener('mousemove', (e) => {
            mouseX = e.clientX;
            mouseY = canvas.height - e.clientY;
        });
    }
    
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
    
    function drawFrame(time) {
        gl.uniform2f(iResolution, canvas.width, canvas.height);
        gl.uniform1f(iTime, time * 0.001);
        gl.uniform2f(iMouse, mouseX, mouseY);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    // Halve the frame rate on touch devices to save battery
    const frameInterval = isMobile ? 1000 / 30 : 0;
    let lastFrameTime = 0;

    function render(time) {
        if (time - lastFrameTime >= frameInterval) {
            lastFrameTime = time;
            drawFrame(time);
        }
        requestAnimationFrame(render);
    }

    window.addEventListener('resize', resize);
    resize();

    // Respect reduced-motion preference: draw a single static frame
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        drawFrame(0);
        return;
    }

    requestAnimationFrame(render);
})();
