# Contributing to Face Gallery

Thank you for your interest in contributing to Face Gallery! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/face-gallery/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU, etc.)
   - Relevant logs or screenshots

### Suggesting Features

1. Check [Discussions](https://github.com/yourusername/face-gallery/discussions) for similar ideas
2. Create a new discussion or issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative approaches considered
   - Any potential drawbacks

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/face-gallery.git
   cd face-gallery
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   ./scripts/install.sh
   source .venv/bin/activate
   ```

4. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

5. **Test your changes**
   ```bash
   # Backend tests
   cd backend
   pytest

   # Manual testing
   cd ..
   ./scripts/run.sh
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add feature description"
   ```

   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes
   - `refactor:` Code refactoring
   - `test:` Test additions/changes
   - `chore:` Maintenance tasks

7. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

   Then open a PR on GitHub with:
   - Clear description of changes
   - Reference to related issues
   - Screenshots/examples if applicable

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8
- **Line length**: 100 characters max
- **Imports**: Group stdlib, third-party, local
- **Type hints**: Use where appropriate

### Testing

- Write tests for new features
- Maintain or improve test coverage
- Run full test suite before submitting PR

### Documentation

- Update README.md if adding features
- Add docstrings to new functions/classes
- Update config documentation if adding settings
- Include examples for new functionality

### Project Structure

When adding new code:

**Backend changes**:
- Core library: `backend/src/face_search/`
- Examples: `backend/examples/`
- Tests: `backend/tests/`

**Frontend changes**:
- Application: `frontend/app.py`
- Static files: `frontend/static/`

**Docker changes**:
- Dockerfiles: `docker/`
- Compose files: `docker-compose.yml`, `docker-compose.cpu.yml`

**Documentation**:
- Main docs: `README.md`, `QUICKSTART.md`, etc.
- Detailed docs: `docs/`
- Config docs: `config/README.md`

## Development Setup

### Backend Development

```bash
cd backend
pip install -e .
pip install pytest pytest-cov

# Run tests
pytest

# With coverage
pytest --cov=face_search --cov-report=html
```

### Frontend Development

```bash
cd frontend
export FACE_VIEWER_DEBUG=1
python app.py

# Test with different configs
CONFIG_PATH=/path/to/test/config.json python app.py
```

### Docker Development

```bash
# Build and test GPU image
docker-compose build
docker-compose up

# Build and test CPU image
docker-compose -f docker-compose.cpu.yml build
docker-compose -f docker-compose.cpu.yml up

# Clean rebuild
docker-compose down
docker system prune -a
docker-compose build --no-cache
```

## Areas for Contribution

### High Priority

- [ ] Performance optimizations
- [ ] Better error messages
- [ ] Cross-platform testing (Windows, macOS, Linux)
- [ ] Documentation improvements
- [ ] Example datasets/tutorials

### Medium Priority

- [ ] Additional model support
- [ ] Video processing
- [ ] Mobile responsive UI
- [ ] Advanced search filters
- [ ] Export/import features

### Low Priority

- [ ] Multi-user support
- [ ] API authentication
- [ ] Plugin system
- [ ] Alternative frontends

## Questions?

- Open a [Discussion](https://github.com/yourusername/face-gallery/discussions)
- Ask in existing Issues or PRs
- Read the documentation in `docs/`

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

Thank you for contributing! 🎉
