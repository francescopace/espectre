# Security Policy

For responsible deployment guidance or to report suspected abuse of ESPectre or related Wi-Fi sensing, see [Security and responsible use](https://espectre.dev/security/). The process below is for product and project vulnerabilities.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### Preferred: GitHub Security Advisories

Use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/francescopace/espectre/security)
2. Click "Report a vulnerability"
3. Fill in the details

This allows private discussion and coordinated disclosure. When a CVE is appropriate, the maintainers can request one through the advisory or provide an existing identifier.

### Information to Include
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### Alternative Contact

If you cannot use GitHub Security Advisories, email <security@espectre.dev>.

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity, typically 30-90 days

### Scope

Security issues relevant to ESPectre include:
- Wi-Fi/CSI data exposure
- MQTT authentication bypass
- ESPHome/Home Assistant integration vulnerabilities
- Firmware vulnerabilities on ESP32
- Vulnerable dependency versions or configurations distributed by ESPectre

### Out of Scope

- Vulnerabilities that exist only in an upstream dependency and do not affect any version or configuration distributed by ESPectre; report those to the upstream project
- Issues requiring physical access to the device
- Social engineering attacks

## Responsible Disclosure

We kindly ask that you:
- Give us reasonable time to fix the issue before public disclosure
- Avoid accessing or modifying other users' data
- Act in good faith to avoid privacy violations

Thank you for helping keep ESPectre secure!
