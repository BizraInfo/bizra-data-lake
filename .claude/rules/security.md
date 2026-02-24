---
paths:
  - "**/*"
---

# Security Rules (Global)

## Never Do
- Hardcode secrets, API keys, passwords
- Log sensitive data
- Trust user input without validation
- Use `eval()` or equivalent
- Disable security features

## Always Do
- Validate all external input
- Use parameterized queries
- Implement rate limiting for APIs
- Use HTTPS for external calls
- Follow principle of least privilege

## Secrets Management
- Use environment variables
- Never commit .env files
- Rotate credentials regularly
- Use secret managers in production
