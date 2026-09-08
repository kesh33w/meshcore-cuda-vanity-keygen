#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_NAME="meshcore-cuda-vanity-keygen"
DESCRIPTION="GPU-accelerated MeshCore Ed25519 vanity key generator with a simple GUI and secure private-key output."
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_DIR"

if git ls-files | grep -Eq '(^|/)results/|(^|/)meshcore_cuda_vanity$|(^|/)__pycache__/|\.pyc$'; then
    echo "Refusing to publish: generated keys, binaries, or caches are tracked by Git." >&2
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "Installing GitHub CLI..."
    sudo apt-get update
    sudo apt-get install -y gh
fi

if ! gh auth status --hostname github.com >/dev/null 2>&1; then
    echo "GitHub authentication is required. Follow the browser prompt."
    gh auth login --hostname github.com --git-protocol https --web
fi

OWNER="$(gh api user --jq .login)"
FULL_REPOSITORY="$OWNER/$REPOSITORY_NAME"

git branch -M main

if gh repo view "$FULL_REPOSITORY" >/dev/null 2>&1; then
    echo "Repository already exists: https://github.com/$FULL_REPOSITORY"
    if git remote get-url origin >/dev/null 2>&1; then
        git remote set-url origin "https://github.com/$FULL_REPOSITORY.git"
    else
        git remote add origin "https://github.com/$FULL_REPOSITORY.git"
    fi
    git push -u origin main
else
    gh repo create "$FULL_REPOSITORY" \
        --public \
        --description "$DESCRIPTION" \
        --source . \
        --remote origin \
        --push
fi

echo
echo "Published: https://github.com/$FULL_REPOSITORY"
