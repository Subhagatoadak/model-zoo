#!/usr/bin/env bash
set -euo pipefail

# Build wheel and source distribution, then upload to Artifactory using twine.
# Usage:
#   ARTIFACTORY_URL=https://artifactory.example.com/api/pypi/pypi-local \
#   TWINE_USERNAME=token TWINE_PASSWORD=xxxxx \
#   scripts/publish.sh

python -m pip install --upgrade build twine
rm -rf dist/
python -m build

if [[ -z "${ARTIFACTORY_URL:-}" ]]; then
  echo "ARTIFACTORY_URL is not set. Example: https://artifactory.example.com/api/pypi/pypi-local"
  exit 1
fi

twine upload --repository-url "$ARTIFACTORY_URL" dist/*

echo "Publish complete."

