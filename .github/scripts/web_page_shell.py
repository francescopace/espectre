#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Render shared website chrome for generated HTML pages."""

from __future__ import annotations

from html import escape


def route_label(route: dict[str, str]) -> str:
    if route["name"] == "home":
        return "Home"
    return route.get("pathLabel") or route["title"].split(" | ")[0]


def render_header(manifest: dict, *, logo_version: str, active: str) -> str:
    links = []
    for name in manifest["navigation"]["main"]:
        route = manifest["by_name"][name]
        selected = name == active
        class_name = "nav-link active" if selected else "nav-link"
        current = ' aria-current="page"' if selected else ""
        links.append(
            f'      <a href="{escape(route["staticPath"], quote=True)}" class="{class_name}"{current}>'
            f'{escape(route_label(route))}</a>'
        )
    links.append(
        '      <a href="https://github.com/francescopace/espectre" target="_blank" '
        'rel="noopener" class="nav-link">GitHub ↗</a>'
    )
    navigation = "\n".join(links)
    return f"""<header class="site-header">
  <div class="site-header-inner">
    <a href="/" class="brand">
      <img src="/assets/images/brand/espectre-logo.svg?v={logo_version}" alt="" width="30" height="30" aria-hidden="true">
      ESPectre
    </a>
    <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="main-navigation">
      <span aria-hidden="true">☰</span><span class="sr-only">Open navigation</span>
    </button>
    <nav class="main-nav" id="main-navigation" aria-label="Main">
{navigation}
    </nav>
  </div>
</header>"""


def render_footer(manifest: dict, *, logo_version: str) -> str:
    links = []
    for name in manifest["navigation"]["footer"]:
        route = manifest["by_name"][name]
        links.append(
            f'      <a href="{escape(route["staticPath"], quote=True)}">'
            f'{escape(route_label(route))}</a>'
        )
        if name == "privacy":
            links.append(
                '      <a href="/privacy/#cookie-settings" '
                'class="js-cookie-settings">Cookie settings</a>'
            )
    footer_links = "\n".join(links)
    return f"""<footer class="site-footer">
  <div class="site-footer-inner">
    <div class="footer-brand">
      <img src="/assets/images/brand/espectre-logo.svg?v={logo_version}" alt="" width="23" height="23" aria-hidden="true">
      ESPectre © 2026 · Open source Wi-Fi sensing platform
    </div>
    <div class="footer-links">
{footer_links}
    </div>
  </div>
</footer>"""


def render_consent_banner() -> str:
    return """<aside class="consent-banner js-consent-banner" role="dialog" aria-labelledby="consent-title" aria-describedby="consent-description" hidden>
  <div class="consent-summary">
    <strong id="consent-title">Your privacy choices</strong>
    <p id="consent-description">Choose whether to allow Google Analytics usage statistics and Google Ads campaign measurement. Both are optional. We do not use remarketing or personalized advertising.</p>
    <a href="/privacy/">Privacy details</a>
  </div>
  <div class="consent-actions">
    <button type="button" class="btn-secondary btn-sm js-consent-reject">Reject all</button>
    <button type="button" class="btn-secondary btn-sm js-consent-customize" aria-expanded="false" aria-controls="consent-preferences">Customize</button>
    <button type="button" class="btn-secondary btn-sm js-consent-accept">Accept all</button>
  </div>
  <div id="consent-preferences" class="consent-preferences js-consent-preferences" hidden>
    <fieldset>
      <legend>Optional measurement</legend>
      <label for="consent-analytics"><input type="checkbox" id="consent-analytics" class="js-consent-analytics" aria-describedby="consent-analytics-help"> Website statistics</label>
      <p id="consent-analytics-help">Allow Google Analytics to measure pages visited and product interactions, so we can improve the site and tools.</p>
      <label for="consent-campaigns"><input type="checkbox" id="consent-campaigns" class="js-consent-campaigns" aria-describedby="consent-campaigns-help"> Campaign measurement</label>
      <p id="consent-campaigns-help">Allow advertising cookies and share ad-click identifiers and contact or SDK link selections with Google Ads to measure campaign results.</p>
    </fieldset>
    <p>You can use every page and tool with both options off. Change your choice anytime through Cookie settings.</p>
    <button type="button" class="btn-secondary btn-sm js-consent-save">Save preferences</button>
  </div>
</aside>"""
