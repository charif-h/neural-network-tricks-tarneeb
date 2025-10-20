# Contributing to Neural Network Tricks Tarneeb

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/neural-network-tricks-tarneeb.git
   cd neural-network-tricks-tarneeb
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guide
- Use meaningful variable and function names
- Add docstrings to all classes and functions
- Keep functions small and focused (Single Responsibility Principle)
- Use type hints where appropriate

### Documentation

- Update README.md if you change functionality
- Add docstrings following Google or NumPy style
- Comment complex algorithms or non-obvious code
- Update CODE_EVALUATION.md if you address issues listed there

### Testing

While we don't currently have a test suite:
- Manually test your changes thoroughly
- Consider adding unit tests for new features
- Run the example scripts to ensure nothing breaks

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable
- Example: "Fix card comparison bug in largerThan method (#123)"

## Priority Areas for Contribution

Based on CODE_EVALUATION.md, here are high-priority areas:

### High Priority
1. **Fix naming inconsistencies** (e.g., `distripute` → `distribute`)
2. **Add error handling** (division by zero, edge cases)
3. **Add unit tests** using pytest
4. **Separate concerns** (game logic vs. AI training)
5. **Add type hints** throughout the codebase

### Medium Priority
6. **Clean up commented code** in GTarneeb.py
7. **Implement proper logging** instead of print statements
8. **Optimize performance** (use numpy operations more efficiently)
9. **Add model checkpointing** and versioning
10. **Refactor TarneebPlayer** to reduce complexity

### Low Priority
11. **Add CI/CD pipeline** (GitHub Actions)
12. **Add visualization tools** for game analysis
13. **Implement Tricks game variant**
14. **Add hyperparameter tuning**
15. **Create Jupyter notebooks** for experimentation

## Project Structure

```
neural-network-tricks-tarneeb/
├── Cards/              # Card and deck implementations
├── Tarneeb/            # Tarneeb game logic
├── Player.py           # Base player class
├── GenModel.py         # Neural network generation
├── constants.py        # Game constants
├── example.py          # Usage examples
└── README.md           # Project documentation
```

## Submitting Changes

1. **Test your changes** thoroughly
2. **Update documentation** as needed
3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub:
   - Describe what you changed and why
   - Reference any related issues
   - Include screenshots for UI changes

## Questions?

- Open an issue on GitHub
- Check CODE_EVALUATION.md for known issues
- Review existing code and documentation

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to Neural Network Tricks Tarneeb! 🎴🤖
