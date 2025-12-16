# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

Only the latest major version receives security updates. We recommend always using the latest release.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

### Preferred: GitHub Security Advisories

Use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/francescopace/espectre/security)
2. Click "Report a vulnerability"
3. Fill in the details

This allows private discussion, coordinated disclosure, and automatic CVE assignment.

### Information to Include
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### Alternative Contact

If you cannot use GitHub Security Advisories, email security@espectre.dev.

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity, typically 30-90 days

### Scope

Security issues relevant to ESPectre include:
- WiFi/CSI data exposure
- MQTT authentication bypass
- ESPHome/Home Assistant integration vulnerabilities
- Firmware vulnerabilities on ESP32

### Out of Scope

- Vulnerabilities in dependencies (report to upstream projects)
- Issues requiring physical access to the device
- Social engineering attacks

## Responsible Disclosure

We kindly ask that you:
- Give us reasonable time to fix the issue before public disclosure
- Avoid accessing or modifying other users' data
- Act in good faith to avoid privacy violations

Thank you for helping keep ESPectre secure!
