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
            
            // ESPectre color palette - teal/cyan theme
            vec3 color1 = vec3(0.0, 0.08, 0.06);   // Deep dark teal (background)
            vec3 color2 = vec3(0.0, 0.83, 0.67);   // Bright teal (accent #00d4aa)
            vec3 color3 = vec3(0.0, 0.5, 0.45);    // Medium teal
            vec3 color4 = vec3(0.1, 0.2, 0.25);    // Dark blue-gray
            
            // Color based on pattern
            float t = fract(pattern + time * 0.1);
            vec3 finalColor;
            if(t < 0.33) {
                finalColor = mix(color1, color3, t * 3.0);
            } else if(t < 0.66) {
                finalColor = mix(color3, color2, (t - 0.33) * 3.0);
            } else {
                finalColor = mix(color2, color1, (t - 0.66) * 3.0);
            }
            
            // Intensity based on waves
            finalColor *= (0.3 + pattern * 0.5);
            
            // Subtle glow in center
            float glow = exp(-dist * 1.5) * 0.15;
            finalColor += glow * vec3(0.0, 0.83, 0.67);
            
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
        return shader;
    }
    
    const vs = createShader(gl.VERTEX_SHADER, vsSource);
    const fs = createShader(gl.FRAGMENT_SHADER, fsSource);
    
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
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
    
    let mouseX = 0, mouseY = 0;
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = canvas.height - e.clientY;
    });
    
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
    
    function render(time) {
        gl.uniform2f(iResolution, canvas.width, canvas.height);
        gl.uniform1f(iTime, time * 0.001);
        gl.uniform2f(iMouse, mouseX, mouseY);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        requestAnimationFrame(render);
    }
    
    window.addEventListener('resize', resize);
    resize();
    requestAnimationFrame(render);
})();

