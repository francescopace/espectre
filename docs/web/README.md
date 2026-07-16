# Website

The static website in this directory provides curated product guides, browser
tools, and interactive demos published at `espectre.dev`.

## Theme And Palette

[`styles.css`](styles.css) is the single source of truth for the website color
palette. Its `:root` block defines:

- background surfaces and their transparent variants
- primary and secondary brand colors
- text, borders, shadows, and code colors
- semantic success, warning, and danger states
- game-specific visual states
- guide-specific surfaces

Page-specific stylesheets must consume these custom properties with `var(...)`
instead of introducing new color literals. Add a semantic custom property to
`styles.css` when a page needs a color or transparency that is not already
available.

The animated background in [`cosmic-bg.js`](cosmic-bg.js) reads
`--bg-primary`, `--bg-secondary`, `--accent`, and `--accent-secondary` from the
computed CSS palette and passes them to the WebGL shader. These four properties
must remain six-digit hexadecimal colors so the shader can parse them.

Intentional exceptions are:

- the raster favicon, which is maintained as a separate image asset
- the fixed black and white QR code colors, which preserve scanner contrast
- colors inside third-party or generated assets, such as `qrcode.js`

## Updating The Palette

To update the website theme:

1. Change the custom properties in `styles.css`.
2. Check the homepage, guides, browser tools, and game states for
   contrast and readability.

For a local visual check, run:

```bash
python -m http.server 8080 --directory docs/web
```

Then open `http://localhost:8080` and test the relevant pages at desktop and
mobile widths.
