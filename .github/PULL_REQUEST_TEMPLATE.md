## 📝 Description

<!-- Provide a clear and concise description of what this PR does -->


## 🔗 Related Issues

<!-- Link to related issues: Fixes #123, Relates to #456 -->


## 📋 Type of Change

<!-- Check all that apply -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📚 Documentation update
- [ ] 🔧 Configuration change
- [ ] ♻️ Refactoring (no functional changes)
- [ ] 🧪 Test addition/modification
- [ ] 🏛️ Governance/Constitution change (requires SAT Council approval)

---

## 🏛️ BIZRA Compliance Checklist

### Fail-Closed Verification
- [ ] All error paths fail visibly (no silent `pass` or `...`)
- [ ] Exceptions are logged AND receipted
- [ ] Fallback behaviors are explicit and documented

### Receipt-Native
- [ ] All decisions/mutations emit structured receipts
- [ ] Receipts include: action, rationale, confidence, trace_id
- [ ] No UPDATE/DELETE on receipt tables

### SNR Quality Gates
- [ ] SNR calculations use approved formulas (if applicable)
- [ ] Low-confidence decisions are gated (not executed)
- [ ] Probe failures trigger FAIL_CLOSED

### Sovereignty First
- [ ] No hardcoded external URLs without SOVEREIGNTY_OVERRIDE
- [ ] Default embedders use NullEmbedder
- [ ] API keys use environment variables only

### Security
- [ ] Input validation on all user-facing endpoints
- [ ] No SQL/prompt injection vulnerabilities
- [ ] No secrets in code or logs

---

## 📊 Ihsān Scorecard (Self-Assessment)

Rate your contribution (1-5):

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Technical Excellence** | /5 | Clean, efficient, tested |
| **Human Impact** | /5 | Serves user flourishing |
| **Justice (Fairness)** | /5 | No bias, accessible |
| **Trustworthiness** | /5 | Auditable, reproducible |

**Weighted Average**: ___ / 5.0 (minimum 3.5 required)

---

## 🧪 Testing

<!-- Describe how this was tested -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed
- [ ] Edge cases covered
- [ ] Failure scenarios tested

**Test Coverage**: ___% (target: ≥95% for critical paths)

---

## 📸 Screenshots / Logs

<!-- If applicable, add screenshots or relevant log outputs -->


---

## 🔄 Deployment Notes

<!-- Any special deployment considerations -->

- [ ] Database migration required
- [ ] Environment variable changes
- [ ] Configuration file changes
- [ ] Dependency updates
- [ ] No special deployment steps

---

## 🐰 CodeRabbit Review Notes

<!-- Optional: Specific areas you'd like CodeRabbit to focus on -->

**Focus Areas**:
- 

**Known Issues / Tech Debt**:
- 

---

## ✅ Final Checklist

- [ ] I have read the [BIZRA Review Standards](.github/CODERABBIT_INSTRUCTIONS.md)
- [ ] My code follows the project style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] All new and existing tests pass locally
- [ ] Any dependent changes have been merged and published

---

*"Excellence with benevolence, justice with transparency, trustworthiness by design."*
