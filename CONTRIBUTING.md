# Contributing

Thank you for your interest in contributing to ESPectre! This document provides guidelines and information for contributors.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [First-Time Contributors](#first-time-contributors)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Contributions](#code-contributions)
- [Data Contributions](#data-contributions)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)
- [Community](#community)

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to contact@espectre.dev.

---

## First-Time Contributors

New to open source? Welcome! Here's how to get started:

1. Check [past contributions](https://github.com/francescopace/espectre/issues?q=is%3Aissue+is%3Aclosed+label%3A%22good+first+issue%22) for inspiration
2. Read through this guide before submitting your first PR
3. Don't hesitate to ask questions in the issue comments

We appreciate all contributions, no matter how small!

---

## Ways to Contribute

| Type | Description | Skill Level |
|------|-------------|-------------|
| **Bug Reports** | Report issues with clear reproduction steps | Beginner |
| **Documentation** | Improve guides, fix typos, add examples | Beginner |
| **Data Collection** | Contribute labeled CSI datasets | Beginner |
| **Code Review** | Review Pull Requests | Intermediate |
| **Bug Fixes** | Fix reported issues | Intermediate |
| **New Features** | Implement roadmap items | Advanced |
| **Algorithm R&D** | Develop new detection algorithms | Advanced |

---

## Development Setup

### Prerequisites

- Python 3.12 (recommended)
- ESP32 device (S3/C6 recommended)
- Home Assistant (optional, for testing ESPHome integration)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/francescopace/espectre.git
cd espectre

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r micro-espectre/requirements.txt
```

### Running Tests

```bash
# C++ tests (ESPHome component)
cd test && pio test

# Python tests (Micro-ESPectre)
cd micro-espectre && pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Code Contributions

### Workflow

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```
4. **Make changes** with tests and documentation
5. **Run tests** to ensure nothing is broken
6. **Commit** with clear messages (see [Commit Guidelines](#commit-guidelines))
7. **Push** to your fork
8. **Open a Pull Request** to the `main` branch

### Commit Guidelines

Use clear, descriptive commit messages:

```
<type>: <short description>

<optional body with more details>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code refactoring (no functional change)
- `perf`: Performance improvement
- `chore`: Maintenance tasks

**Examples:**
```
feat: add low-pass filter for noise reduction
fix: correct calibration for edge cases
docs: update TUNING.md with filter examples
test: add unit tests for Hampel filter
```

### Code Style

#### C++ (ESPHome Component)

- Follow ESPHome component conventions
- Use ESP-IDF framework (not Arduino)
- Use `ESP_LOGD`, `ESP_LOGI`, `ESP_LOGW`, `ESP_LOGE` for logging
- All code and comments in English

**File Header:**
```cpp
/*
 * ESPectre - [Component Name]
 * 
 * [Brief description]
 * 
 * Author: [your name] <[your email]>
 * License: GPLv3
 */
```

#### Python (Micro-ESPectre)

- MicroPython compatible (no asyncio, limited stdlib)
- Memory-efficient (ESP32 constraints)
- Use `config.py` for constants
- All code and comments in English

**File Header:**
```python
"""
Micro-ESPectre - [Module Name]

[Brief description]

Author: [your name] <[your email]>
License: GPLv3
"""
```

### Pull Request Guidelines

- **Target branch**: Always `main`
- **Title**: Clear, descriptive title
- **Description**: Explain what and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update relevant docs
- **Single focus**: One feature/fix per PR

### Quality Standards

| Requirement | Target |
|-------------|--------|
| Test coverage | >80% for core modules |
| CI passing | All checks must pass |
| Documentation | Features require docs |
| Code review | At least one approval |

---

## Data Contributions

Help build a diverse CSI dataset for ML training! Your contributions will improve gesture recognition and HAR models for everyone.

### How to Contribute Data

1. **Collect data** following [ML_DATA_COLLECTION.md](micro-espectre/ML_DATA_COLLECTION.md)
2. **Ensure quality**:
   - At least 10 samples per label
   - 30+ seconds per sample
   - Quiet room for baseline recordings
3. **Document your setup**:
   - ESP32 model (S3, C6, etc.)
   - Distance from router
   - Room type (living room, office, etc.)
   - Any notable characteristics
4. **Submit via Pull Request**:
   - Add your data to `micro-espectre/data/<label>/`
   - Include a brief description in the PR

### Priority Gestures

We're particularly looking for these gestures useful for smart home automation:

| Priority | Gesture | Description | Use Case |
|----------|---------|-------------|----------|
| 🔴 High | `swipe_left` / `swipe_right` | Hand swipe in air | Change scene, adjust brightness |
| 🔴 High | `push` / `pull` | Push away / pull toward | Turn on/off, open/close |
| 🔴 High | `circle_cw` / `circle_ccw` | Circular hand motion | Dimmer, thermostat |
| 🟡 Medium | `clap` | Hand clap | Toggle lights |
| 🟡 Medium | `sit_down` / `stand_up` | Sitting/standing | TV mode, energy saving |
| 🟡 Medium | `fall` | Person falling | Elderly safety alert |
| 🟢 Low | `idle` | No movement | Baseline (always needed) |

### Data Privacy

- CSI data is **anonymous** - contains only radio channel characteristics
- No personal information, images, or audio
- You retain ownership of your contributions
- All contributions will be credited

---

## Documentation

Good documentation is essential! Here's how you can help:

### Types of Documentation

| Type | Location | Description |
|------|----------|-------------|
| **README** | `README.md` | Project overview, quick start |
| **Setup Guide** | `SETUP.md` | Installation and configuration |
| **Tuning Guide** | `TUNING.md` | Parameter optimization |
| **Algorithms** | `micro-espectre/ALGORITHMS.md` | Scientific documentation |
| **API Docs** | Code comments | Function/class documentation |

### Documentation Guidelines

- Write in clear, simple English
- Include code examples where helpful
- Keep formatting consistent with existing docs
- Test any commands or code snippets you include

---

## Reporting Issues

### Before Reporting

1. **Search existing issues** to avoid duplicates
2. **Check the FAQ** in README.md
3. **Try the latest version** from `develop` branch

### Bug Reports

Include:
- **Description**: Clear description of the bug
- **Steps to reproduce**: Numbered steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**:
  - ESP32 model (S3, C6, etc.)
  - ESPectre version
  - Home Assistant version (if applicable)
  - Relevant configuration

### Feature Requests

Include:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How might it work?
- **Alternatives**: Other approaches considered

---

## Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and design discussions

### Stay Updated

- **Watch** the repository for updates
- **Star** if you find it useful
- **Share** with others who might benefit

---

## Recognition

All contributors are recognized in:
- Pull Request acknowledgments
- Release notes for significant contributions
- Data contributors credited in dataset documentation

---

## License

By contributing to ESPectre, you agree that your contributions will be licensed under the **GPLv3** license. This means:

- Your code will remain free and open source
- You retain copyright of your contributions
- No CLA (Contributor License Agreement) is required

See [LICENSE](LICENSE) for details.

