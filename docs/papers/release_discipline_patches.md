# BIZRA Release Discipline Patches
# Audit findings: mutable refs in CI, container immutability gaps

# These are the exact patches for the three files the audit flagged.
# Apply with: git apply release_discipline.patch
# Or apply manually using the instructions below.

## Patch 1: Pin setup-uv version (lock-deps.yml:28)
#
# BEFORE:
#   uses: astral-sh/setup-uv@latest
#
# AFTER:
#   uses: astral-sh/setup-uv@v4.3.0  # Pinned — no mutable refs
#
# File: .github/workflows/lock-deps.yml
# Line: 28

## Patch 2: Pin nomic-embed-text model version (docker-compose.flywheel.yml:85)
#
# BEFORE:
#   command: pull nomic-embed-text:latest
#
# AFTER:
#   command: pull nomic-embed-text:v1.5  # Pinned — reproducible preload
#
# File: docker-compose.flywheel.yml
# Line: 85
#
# NOTE: Verify the exact version tag from `ollama list` on NODE0.
#       Use the specific digest if available:
#       command: pull nomic-embed-text@sha256:<digest>

## Patch 3: Enable readOnlyRootFilesystem (rollouts.yaml:69)
#
# BEFORE:
#   readOnlyRootFilesystem: false
#
# AFTER:
#   readOnlyRootFilesystem: true
#
# File: deploy/argocd/rollouts.yaml
# Line: 69
#
# IMPORTANT: This may require adding tmpfs volume mounts for:
#   - /tmp (application temp files)
#   - /var/run (PID files)
#   - /home/app/.cache (model cache)
#
# Add to the container spec:
#   volumeMounts:
#     - name: tmp
#       mountPath: /tmp
#     - name: run
#       mountPath: /var/run
#     - name: cache
#       mountPath: /home/app/.cache
#
# And to the pod spec:
#   volumes:
#     - name: tmp
#       emptyDir:
#         sizeLimit: 100Mi
#     - name: run
#       emptyDir:
#         sizeLimit: 10Mi
#     - name: cache
#       emptyDir:
#         sizeLimit: 500Mi

## Verification after applying:
#
# 1. grep -r "latest" .github/workflows/ deploy/ docker-compose*.yml
#    Expected: zero results (all refs pinned)
#
# 2. grep "readOnlyRootFilesystem: false" deploy/
#    Expected: zero results
#
# 3. Run CI to verify lock-deps still works with pinned version
#
# 4. Test container startup with readOnlyRootFilesystem: true
#    docker-compose -f docker-compose.flywheel.yml up --build
