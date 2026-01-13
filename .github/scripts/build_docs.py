#!/usr/bin/env python3
"""
ESPectre - Documentation Builder

Converts Markdown documentation files to static HTML pages
with the same look and feel as the main website.

Usage:
    python build_docs.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import markdown
from pathlib import Path
import re
import sys
from datetime import datetime
import os

# Documentation files to convert
# Format: 'source.md': ('output_path', 'Page Title', 'Description')
DOCS = {
    'README.md': (
        'docs/documentation/index.html',
        'Documentation',
        'Complete documentation for ESPectre Wi-Fi motion detection system'
    ),
    'SETUP.md': (
        'docs/documentation/setup/index.html',
        'Setup Guide',
        'Installation and configuration guide for ESPectre'
    ),
    'TUNING.md': (
        'docs/documentation/tuning/index.html',
        'Tuning Guide',
        'Parameter tuning guide for optimal motion detection'
    ),
    'PERFORMANCE.md': (
        'docs/documentation/performance/index.html',
        'Performance',
        'Performance metrics and benchmarks'
    ),
    'CHANGELOG.md': (
        'docs/documentation/changelog/index.html',
        'Changelog',
        'Version history and release notes'
    ),
    'ROADMAP.md': (
        'docs/documentation/roadmap/index.html',
        'Roadmap',
        'Project roadmap and future plans'
    ),
    'CONTRIBUTING.md': (
        'docs/documentation/contributing/index.html',
        'Contributing',
        'How to contribute to ESPectre'
    ),
    'micro-espectre/ALGORITHMS.md': (
        'docs/documentation/algorithms/index.html',
        'Algorithms',
        'Scientific documentation of motion detection algorithms'
    ),
    'micro-espectre/README.md': (
        'docs/documentation/micro-espectre/index.html',
        'Micro-ESPectre',
        'R&D platform for WiFi CSI motion detection'
    ),
    'micro-espectre/ML_DATA_COLLECTION.md': (
        'docs/documentation/ml-data-collection/index.html',
        'ML Data Collection',
        'Guide for collecting labeled data for machine learning'
    ),
}

# Map original .md paths to new doc paths (for link rewriting)
LINK_MAP = {
    'README.md': '/documentation/',
    'SETUP.md': '/documentation/setup/',
    'TUNING.md': '/documentation/tuning/',
    'PERFORMANCE.md': '/documentation/performance/',
    'CHANGELOG.md': '/documentation/changelog/',
    'ROADMAP.md': '/documentation/roadmap/',
    'CONTRIBUTING.md': '/documentation/contributing/',
    'CODE_OF_CONDUCT.md': 'https://github.com/francescopace/espectre/blob/main/CODE_OF_CONDUCT.md',
    'SECURITY.md': 'https://github.com/francescopace/espectre/blob/main/SECURITY.md',
    'LICENSE': 'https://github.com/francescopace/espectre/blob/main/LICENSE',
    'micro-espectre/ALGORITHMS.md': '/documentation/algorithms/',
    'micro-espectre/README.md': '/documentation/micro-espectre/',
    'micro-espectre/ML_DATA_COLLECTION.md': '/documentation/ml-data-collection/',
    'micro-espectre/tools/README.md': 'https://github.com/francescopace/espectre/blob/main/micro-espectre/tools/README.md',
    'test/README.md': 'https://github.com/francescopace/espectre/blob/main/test/README.md',
    'docs/game/README.md': 'https://github.com/francescopace/espectre/blob/main/docs/game/README.md',
}

# HTML template for documentation pages
TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | ESPectre</title>
    <meta name="description" content="{description}">
    <meta name="keywords" content="ESPectre, WiFi sensing, CSI, motion detection, Home Assistant, ESPHome, {title}">
    <meta name="author" content="Francesco Pace">

    <!-- Canonical URL -->
    <link rel="canonical" href="https://espectre.dev{canonical_path}">

    <!-- Open Graph / Facebook -->
    <meta property="og:url" content="https://espectre.dev{canonical_path}">
    <meta property="og:type" content="article">
    <meta property="og:title" content="{title} | ESPectre">
    <meta property="og:description" content="{description}">
    <meta property="og:image" content="https://espectre.dev/espectre.png">

    <!-- Twitter -->
    <meta name="twitter:card" content="summary_large_image">
    <meta property="twitter:domain" content="espectre.dev">
    <meta property="twitter:url" content="https://espectre.dev{canonical_path}">
    <meta name="twitter:title" content="{title} | ESPectre">
    <meta name="twitter:description" content="{description}">
    <meta name="twitter:image" content="https://espectre.dev/espectre.png">

    <!-- Favicon -->
    <link rel="icon"
        href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>👻</text></svg>">

    <!-- Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-S0NQNG0V11"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', 'G-S0NQNG0V11');
    </script>

    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">

    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <!-- Highlight.js for syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

    <!-- Styles -->
    <link rel="stylesheet" href="/styles.css">
    <link rel="stylesheet" href="/_pagefind/pagefind-ui.css">
</head>
<body class="doc-page">
    <canvas id="cosmic-bg"></canvas>

    <header id="site-header">
        <noscript>
            <a href="/" class="logo">ESPectre</a>
            <nav>
                <a href="/">Home</a>
                <a href="/documentation/">Docs</a>
                <a href="https://github.com/francescopace/espectre">GitHub</a>
            </nav>
        </noscript>
    </header>

    <main>
        <section class="doc-section">
            <div class="doc-search">
                <div id="search"></div>
            </div>
            <div class="doc-section-inner">
                <article class="doc-content">
                    {content}
                </article>
            </div>
        </section>
    </main>

    <footer id="site-footer"></footer>

    <script src="/components.js"></script>
    <script>loadHeader(); loadFooter();</script>
    <script src="/cosmic-bg.js"></script>
    <script src="/analytics.js"></script>
    <script>hljs.highlightAll();</script>
    <script src="/_pagefind/pagefind-ui.js"></script>
    <script>
        window.addEventListener('DOMContentLoaded', () => {{
            new PagefindUI({{
                element: "#search",
                showSubResults: true,
                showImages: false
            }});
        }});
    </script>
    
    <!-- Buy Me a Coffee Widget -->
    <script data-name="BMC-Widget" data-cfasync="false" src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" data-id="espectre" data-description="Support me on Buy me a coffee!" data-color="#40DCA5" data-position="Right" data-x_margin="18" data-y_margin="18"></script>
</body>
</html>
'''



def rewrite_links(content: str, source_path: str) -> str:
    """Rewrite internal .md links to their new paths."""
    source_dir = str(Path(source_path).parent)
    if source_dir == '.':
        source_dir = ''
    
    def replace_link(match):
        full_match = match.group(0)
        text = match.group(1)
        href = match.group(2)
        
        # Skip external links and anchors
        if href.startswith(('http://', 'https://', '#', 'mailto:')):
            return full_match
        
        # Handle anchor in link
        anchor = ''
        if '#' in href:
            href, anchor = href.split('#', 1)
            anchor = '#' + anchor
        
        # Resolve relative path
        if source_dir and not href.startswith('/'):
            # Handle ../ paths
            if href.startswith('../'):
                # Go up from source directory
                parts = source_dir.split('/')
                href_parts = href.split('/')
                up_count = 0
                for p in href_parts:
                    if p == '..':
                        up_count += 1
                    else:
                        break
                remaining_path = '/'.join(href_parts[up_count:])
                base_path = '/'.join(parts[:-up_count]) if up_count <= len(parts) else ''
                href = (base_path + '/' + remaining_path).lstrip('/')
            else:
                href = source_dir + '/' + href
        
        # Check if we have a mapping for this file
        if href in LINK_MAP:
            new_href = LINK_MAP[href] + anchor
            return f'[{text}]({new_href})'
        
        # For images, point to GitHub raw
        if href.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
            github_url = f'https://raw.githubusercontent.com/francescopace/espectre/main/{href}'
            return f'[{text}]({github_url})'
        
        # For other files (yaml, py, etc.), link to GitHub blob
        if href.endswith(('.yaml', '.yml', '.py', '.cpp', '.h', '.json')):
            github_url = f'https://github.com/francescopace/espectre/blob/main/{href}'
            return f'[{text}]({github_url})'
        
        return full_match
    
    # Match markdown links: [text](url)
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replace_link, content)
    
    return content


def rewrite_images(content: str, source_path: str) -> str:
    """Rewrite image paths to point to GitHub raw."""
    source_dir = str(Path(source_path).parent)
    if source_dir == '.':
        source_dir = ''
    
    def replace_image(match):
        alt = match.group(1)
        src = match.group(2)
        
        # Skip external images
        if src.startswith(('http://', 'https://')):
            return match.group(0)
        
        # Resolve relative path
        if source_dir and not src.startswith('/'):
            if src.startswith('../'):
                parts = source_dir.split('/')
                src_parts = src.split('/')
                up_count = sum(1 for p in src_parts if p == '..')
                remaining = '/'.join(src_parts[up_count:])
                base = '/'.join(parts[:-up_count]) if up_count <= len(parts) else ''
                src = (base + '/' + remaining).lstrip('/')
            else:
                src = source_dir + '/' + src
        
        # Point to GitHub raw
        github_url = f'https://raw.githubusercontent.com/francescopace/espectre/main/{src}'
        return f'![{alt}]({github_url})'
    
    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, content)
    return content


def clean_content(content: str, source_path: str) -> str:
    """Clean up markdown content for better rendering.
    
    Applies the same transformations as main.js does at runtime.
    """
    # README.md: Remove header section (badges, title, Medium link) 
    # and add a clean title - same as main.js
    if source_path == 'README.md':
        toc_marker = '## Table of Contents'
        toc_index = content.find(toc_marker)
        if toc_index != -1:
            clean_header = (
                '# ESPectre\n\n'
                '**Motion detection system based on Wi-Fi spectrum analysis (CSI), '
                'with native Home Assistant integration via ESPHome.**\n\n'
            )
            content = clean_header + content[toc_index:]
    
    # micro-espectre/README.md: Replace MicroPython title with cleaner one
    # Same as main.js: markdown.replace(/^# .+$/m, '# Micro-ESPectre')
    if source_path == 'micro-espectre/README.md':
        content = re.sub(r'^# .+$', '# Micro-ESPectre', content, count=1, flags=re.MULTILINE)
    
    return content


def build_docs():
    """Build all documentation pages."""
    print("Building ESPectre documentation...\n")
    
    # Configure markdown with extensions
    md = markdown.Markdown(
        extensions=[
            'toc',
            'tables', 
            'fenced_code',
            'codehilite',
            'attr_list',
            'md_in_html',
        ],
        extension_configs={
            'toc': {
                'permalink': False,  # We'll make the heading itself a link
            },
            'codehilite': {
                'css_class': 'highlight',
                'guess_lang': False,
            },
        }
    )
    
    built_count = 0
    errors = []
    
    for source_path, (dest_path, title, description) in DOCS.items():
        try:
            # Read source file
            source_file = Path(source_path)
            if not source_file.exists():
                errors.append(f"Source not found: {source_path}")
                continue
            
            content = source_file.read_text(encoding='utf-8')
            
            # Clean and process content
            content = clean_content(content, source_path)
            content = rewrite_links(content, source_path)
            content = rewrite_images(content, source_path)
            
            # Convert to HTML
            md.reset()
            html_content = md.convert(content)
            
            # Make headings themselves links (like GitHub)
            # Transform: <h2 id="foo">Title</h2> → <h2 id="foo"><a href="#foo">Title</a></h2>
            html_content = re.sub(
                r'<(h[1-6]) id="([^"]+)">(.+?)</\1>',
                r'<\1 id="\2"><a href="#\2" class="heading-link">\3</a></\1>',
                html_content
            )
            
            # Calculate canonical path
            canonical_path = '/' + dest_path.replace('docs/', '').replace('index.html', '')
            
            # Generate final HTML
            html = TEMPLATE.format(
                title=title,
                description=description,
                canonical_path=canonical_path,
                content=html_content,
            )
            
            # Write output file
            output_file = Path(dest_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(html, encoding='utf-8')
            
            print(f"  ✓ {source_path} → {dest_path}")
            built_count += 1
            
        except Exception as e:
            errors.append(f"{source_path}: {str(e)}")
    
    
    print(f"\nBuilt {built_count} documentation pages")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors:
            print(f"  ✗ {error}")
        return 1
    
    # Generate sitemap
    generate_sitemap()
    
    return 0


def generate_sitemap():
    """Generate sitemap.xml with dynamic lastmod dates."""
    
    # Static pages with their source files for lastmod
    static_pages = [
        ('https://espectre.dev/', 'docs/index.html', 'daily', '1.0'),
        ('https://espectre.dev/game/', 'docs/game/index.html', 'daily', '0.8'),
    ]
    
    # Documentation pages with priorities
    doc_priorities = {
        'README.md': ('https://espectre.dev/documentation/', 'daily', '0.9'),
        'SETUP.md': ('https://espectre.dev/documentation/setup/', 'daily', '0.8'),
        'TUNING.md': ('https://espectre.dev/documentation/tuning/', 'daily', '0.8'),
        'PERFORMANCE.md': ('https://espectre.dev/documentation/performance/', 'daily', '0.7'),
        'micro-espectre/ALGORITHMS.md': ('https://espectre.dev/documentation/algorithms/', 'daily', '0.8'),
        'micro-espectre/README.md': ('https://espectre.dev/documentation/micro-espectre/', 'daily', '0.7'),
        'CHANGELOG.md': ('https://espectre.dev/documentation/changelog/', 'daily', '0.6'),
        'ROADMAP.md': ('https://espectre.dev/documentation/roadmap/', 'daily', '0.6'),
        'CONTRIBUTING.md': ('https://espectre.dev/documentation/contributing/', 'daily', '0.5'),
        'micro-espectre/ML_DATA_COLLECTION.md': ('https://espectre.dev/documentation/ml-data-collection/', 'daily', '0.5'),
    }
    
    def get_lastmod(filepath):
        """Get file modification date in YYYY-MM-DD format."""
        try:
            mtime = os.path.getmtime(filepath)
            return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')
        except:
            return datetime.now().strftime('%Y-%m-%d')
    
    urls = []
    
    # Add static pages
    for loc, source, changefreq, priority in static_pages:
        lastmod = get_lastmod(source)
        urls.append(f'''    <url>
        <loc>{loc}</loc>
        <lastmod>{lastmod}</lastmod>
        <changefreq>{changefreq}</changefreq>
        <priority>{priority}</priority>
    </url>''')
    
    # Add documentation pages
    for source, (loc, changefreq, priority) in doc_priorities.items():
        lastmod = get_lastmod(source)
        urls.append(f'''    <url>
        <loc>{loc}</loc>
        <lastmod>{lastmod}</lastmod>
        <changefreq>{changefreq}</changefreq>
        <priority>{priority}</priority>
    </url>''')
    
    sitemap = f'''<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{chr(10).join(urls)}
</urlset>
'''
    
    sitemap_path = Path('docs/sitemap.xml')
    sitemap_path.write_text(sitemap, encoding='utf-8')
    print(f"  ✓ Generated sitemap.xml")


if __name__ == '__main__':
    sys.exit(build_docs())
