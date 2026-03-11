# Release Policy — bizra-node0

## Branch Protection

- `main` is protected: no direct push, PR required
- All CI checks must pass before merge
- At least 1 approval from a maintainer (MoMo or designated SAT-5 reviewer)

## Release Process

1. **Tag**: `git tag -s v$VERSION -m "Release v$VERSION"`
2. **Verify**: `bash scripts/node0_genesis_ceremony.sh --full` must pass
3. **Build**: `pip wheel .` produces `bizra_node0-$VERSION-py3-none-any.whl`
4. **Sign**: `gpg --detach-sign --armor bizra_node0-$VERSION-py3-none-any.whl`
5. **Publish**: GitHub Release with signed wheel + ceremony JSON receipt

## Versioning

- SemVer 2.0.0: `MAJOR.MINOR.PATCH`
- MAJOR: breaking changes to operator surface or lifecycle contract
- MINOR: new gates, commands, or features (backward compatible)
- PATCH: bug fixes, security patches, doc corrections

## Signing

- Release artifacts must be GPG-signed by a recognized key
- Ceremony JSON receipt included with every release
- `UPSTREAM_IMPORT_MANIFEST.yaml` pinned to upstream commit hash

## Hotfix Policy

- Hotfixes branch from `main`, merge to `main` via PR
- Emergency override requires SAT-5 consensus audit trail
- No bypass of MVSA gate or ceremony verification
