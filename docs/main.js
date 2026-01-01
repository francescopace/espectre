/*
 * ESPectre - Landing Page Scripts
 * 
 * Interactive effects for the ESPectre project landing page.
 * Scroll effects and markdown documentation loader.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

// Dynamic header - shrink and glow on scroll
let lastScroll = 0;

window.addEventListener('scroll', function() {
    const header = document.getElementById('site-header');
    if (!header) return;
    
    const currentScroll = window.scrollY;
    
    if (currentScroll > 50) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
    
    lastScroll = currentScroll;
});

// Documentation loader
(function() {
    const DOCS = {
        'README': 'README.md',
        'SETUP': 'SETUP.md',
        'TUNING': 'TUNING.md',
        'PERFORMANCE': 'PERFORMANCE.md',
        'MICRO_ESPECTRE': 'micro-espectre/README.md',
        'TESTING': 'test/README.md',
        'CHANGELOG': 'CHANGELOG.md',
        'CONTRIBUTING': 'CONTRIBUTING.md',
        'ROADMAP': 'ROADMAP.md'
    };

    const GITHUB_RAW = 'https://raw.githubusercontent.com/francescopace/espectre/main/';

    // Helper function to generate slug from heading text (same as GitHub)
    function slugify(text) {
        return text
            .toLowerCase()
            .trim()
            .replace(/[^\w\s-]/g, '')  // Remove special chars except spaces and hyphens
            .replace(/\s+/g, '-')       // Replace spaces with hyphens
            .replace(/-+/g, '-');       // Replace multiple hyphens with single
    }

    // Configure marked to add IDs to headings (compatible with marked v5+)
    const renderer = {
        heading(token) {
            const text = token.text;
            const depth = token.depth;
            const slug = slugify(text);
            return `<h${depth} id="${slug}">${text}</h${depth}>\n`;
        }
    };
    marked.use({ renderer: renderer });

    const params = new URLSearchParams(window.location.search);
    const docName = params.get('doc');

    if (docName && DOCS[docName]) {
        const docSection = document.getElementById('doc-content');
        const landingSection = document.getElementById('landing-content');

        if (docSection && landingSection) {
            // Show doc section, hide landing
            docSection.classList.remove('hidden');
            landingSection.classList.add('hidden');

            // Show loading state
            docSection.innerHTML = `
                <div class="doc-section-inner">
                    <div class="doc-loading">
                        <i class="fas fa-spinner"></i>
                        <p>Loading ${docName}...</p>
                    </div>
                </div>
            `;

            // Fetch and render markdown
            fetch(GITHUB_RAW + DOCS[docName])
                .then(response => {
                    if (!response.ok) throw new Error('Document not found');
                    return response.text();
                })
                .then(markdown => {
                    // Process README: remove header section (badges, title, Medium link)
                    // and add a clean title
                    if (docName === 'README') {
                        const tocIndex = markdown.indexOf('## Table of Contents');
                        if (tocIndex !== -1) {
                            markdown = '# ESPectre\n\n**Motion detection system based on Wi-Fi spectrum analysis (CSI), with native Home Assistant integration via ESPHome.**\n\n' + markdown.substring(tocIndex);
                        }
                    }
                    
                    // Process MICRO_ESPECTRE: replace MicroPython title with cleaner one
                    if (docName === 'MICRO_ESPECTRE') {
                        markdown = markdown.replace(/^# .+$/m, '# Micro-ESPectre');
                    }
                    
                    // Fix image paths to point to GitHub raw
                    // Handle relative paths like "images/foo.png" or "../images/foo.png"
                    const docPath = DOCS[docName];
                    const docDir = docPath.includes('/') ? docPath.substring(0, docPath.lastIndexOf('/') + 1) : '';
                    
                    markdown = markdown.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(match, alt, src) {
                        // Skip if already absolute URL
                        if (src.startsWith('http://') || src.startsWith('https://')) {
                            return match;
                        }
                        // Resolve relative path
                        let fullPath = docDir + src;
                        // Handle "../" in path
                        while (fullPath.includes('../')) {
                            fullPath = fullPath.replace(/[^/]+\/\.\.\//g, '');
                        }
                        return `![${alt}](${GITHUB_RAW}${fullPath})`;
                    });
                    
                    // Map of markdown files to doc keys
                    const MD_TO_DOC = {
                        'README.md': 'README',
                        'SETUP.md': 'SETUP',
                        'TUNING.md': 'TUNING',
                        'PERFORMANCE.md': 'PERFORMANCE',
                        'CHANGELOG.md': 'CHANGELOG',
                        'CONTRIBUTING.md': 'CONTRIBUTING',
                        'ROADMAP.md': 'ROADMAP',
                        'micro-espectre/README.md': 'MICRO_ESPECTRE',
                        'micro-espectre/ALGORITHMS.md': 'ALGORITHMS',
                        'test/README.md': 'TESTING'
                    };
                    
                    // Fix markdown links to use ?doc= parameter
                    markdown = markdown.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function(match, text, href) {
                        // Skip if already absolute URL or anchor
                        if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('#')) {
                            return match;
                        }
                        // Skip images (already processed)
                        if (match.startsWith('!')) {
                            return match;
                        }
                        // Resolve relative path
                        let fullPath = docDir + href;
                        // Handle "../" in path
                        while (fullPath.includes('../')) {
                            fullPath = fullPath.replace(/[^/]+\/\.\.\//g, '');
                        }
                        // Remove anchor from path for lookup
                        const pathWithoutAnchor = fullPath.split('#')[0];
                        const anchor = fullPath.includes('#') ? '#' + fullPath.split('#')[1] : '';
                        
                        // Check if it's a known markdown file
                        if (MD_TO_DOC[pathWithoutAnchor]) {
                            return `[${text}](index.html?doc=${MD_TO_DOC[pathWithoutAnchor]}${anchor})`;
                        }
                        // For other files, link to GitHub
                        if (href.endsWith('.md') || href.endsWith('.yaml') || href.endsWith('.py') || href.endsWith('.cpp') || href.endsWith('.h')) {
                            return `[${text}](https://github.com/francescopace/espectre/blob/main/${fullPath})`;
                        }
                        return match;
                    });
                    
                    // Parse markdown to HTML (renderer adds IDs to headings)
                    let html = marked.parse(markdown);
                    
                    docSection.innerHTML = `
                        <div class="doc-section-inner">
                            ${html}
                        </div>
                    `;
                    document.title = `ESPectre - ${docName}`;
                    // Scroll to top
                    window.scrollTo(0, 0);
                })
                .catch(error => {
                    docSection.innerHTML = `
                        <div class="doc-section-inner">
                            <div class="doc-error">
                                <i class="fas fa-exclamation-triangle"></i>
                                <p>Error loading document: ${error.message}</p>
                            </div>
                        </div>
                    `;
                });
        }
    }
})();
