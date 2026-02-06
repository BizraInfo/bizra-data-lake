# Contributing to BIZRA

Thank you for your interest in contributing to BIZRA. Every contribution matters.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `pytest` (Python) and `cargo test --workspace` (Rust)
5. Submit a pull request

## Development Setup

### Python
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

### Rust
```bash
cd bizra-omega
cargo build --workspace
cargo test --workspace
```

## Code Standards

### Quality Thresholds
- **Ihsan (Excellence)**: All code must pass with quality score >= 0.95
- **SNR**: Signal-to-noise ratio >= 0.85 for all operations
- **ADL (Justice)**: Gini coefficient <= 0.40 for resource distribution

### Style
- **Python**: Black formatter, isort for imports, mypy strict mode
- **Rust**: `cargo fmt` and `cargo clippy`
- **Commits**: Conventional commits (`feat:`, `fix:`, `docs:`, etc.)

### Security
- Never hardcode secrets or API keys
- Use environment variables for all credentials
- Validate all external input
- No `eval()` or equivalent

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add test coverage for new features
4. One approval required for merge

## Reporting Issues

Use GitHub Issues with clear reproduction steps and expected vs actual behavior.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
