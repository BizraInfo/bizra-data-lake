---
paths:
  - "**/*.ts"
  - "**/*.tsx"
  - "apps/**"
  - "packages/**"
---

# TypeScript Code Standards

## Types
- `strict: true` always
- No `any` unless absolutely necessary
- Prefer `interface` over `type` for objects
- Use `unknown` over `any` for truly unknown types

## React (if applicable)
- Functional components only
- Custom hooks for shared logic
- Props interfaces suffixed with `Props`
- Event handlers prefixed with `handle`

## Async
- Prefer `async/await` over `.then()`
- Error boundaries for async components
- Loading/error states explicit

## Testing
- Vitest + React Testing Library
- Test behavior, not implementation
- Integration tests for critical paths
