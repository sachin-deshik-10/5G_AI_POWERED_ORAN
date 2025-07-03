# Contributing to AI-Powered 5G Open RAN Optimizer

We welcome contributions to the AI-Powered 5G Open RAN Optimizer! This document provides guidelines for contributing to this project.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Azure CLI (for cloud deployment)
- Git
- Docker (for containerized deployment)

### Development Setup

1. **Fork the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/5G_AI_POWERED_ORAN.git
   cd 5G_AI_POWERED_ORAN
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Run tests**

   ```bash
   pytest tests/
   ```

## 📝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear and descriptive title
- A detailed description of the suggested enhancement
- Explain why this enhancement would be useful
- Include examples if applicable

### Pull Requests

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards outlined below
   - Add tests for your changes
   - Update documentation as needed

3. **Test your changes**

   ```bash
   pytest tests/
   flake8 src/
   black src/
   ```

4. **Commit your changes**

   ```bash
   git commit -m "feat: add amazing feature"
   ```

5. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

## 📋 Coding Standards

### Python Code Style

We follow PEP 8 with these specific guidelines:

- **Line length**: 88 characters (Black default)
- **Imports**: Use absolute imports, group imports as per PEP 8
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Use type hints for all public functions

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

### Code Review Process

1. All submissions require review
2. We use GitHub pull requests for this purpose
3. Reviewers will check for:
   - Code quality and style
   - Test coverage
   - Documentation updates
   - Performance implications
   - Security considerations

## 🧪 Testing Guidelines

### Unit Tests

- Write unit tests for all new functions
- Aim for >90% code coverage
- Use pytest fixtures for common test data
- Mock external dependencies

### Integration Tests

- Test API endpoints thoroughly
- Test Azure deployment scenarios
- Test data pipeline end-to-end

### Performance Tests

- Benchmark critical algorithms
- Test scalability with large datasets
- Monitor memory usage and latency

## 📚 Documentation

- Update README.md for user-facing changes
- Update docstrings for all modified functions
- Add examples for new features
- Update API documentation as needed

## 🔒 Security

- Never commit sensitive information (keys, passwords, etc.)
- Use environment variables for configuration
- Follow OWASP guidelines for web security
- Report security vulnerabilities privately

## 📊 Research Contributions

For research-related contributions:

- Follow academic standards for citations
- Include mathematical proofs where applicable
- Provide experimental validation
- Update relevant research documentation in `docs/research/`

## 🏷️ Release Process

1. Version bumping follows [Semantic Versioning](https://semver.org/)
2. Releases are created from the `main` branch
3. Each release includes:
   - CHANGELOG entry
   - Git tag
   - Release notes
   - Docker images
   - Azure deployment templates

## 💬 Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [contact@5g-oran-optimizer.com](mailto:contact@5g-oran-optimizer.com)

## 🎯 Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Research publications (for significant research contributions)

## 📜 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to the AI-Powered 5G Open RAN Optimizer! 🚀
