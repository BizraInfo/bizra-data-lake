---
paths:
  - "**/*.py"
  - "core/**"
  - "tests/**"
---

# Python Code Standards

## Style
- Follow PEP 8 and PEP 484 (type hints)
- Use `black` formatting conventions
- Docstrings for public APIs (Google style)
- f-strings over .format()

## Patterns
- Dataclasses for data containers
- async/await for I/O operations
- Context managers for resources
- Explicit is better than implicit

## Imports
```python
# Standard library
import os
from typing import Optional, List

# Third-party
import numpy as np

# Local
from .module import function
```

## Error Handling
- Specific exceptions over bare `except`
- Custom exceptions for domain errors
- Logging over print statements
