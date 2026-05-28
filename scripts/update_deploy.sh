#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ops_common.sh"

cd "$(ombre_repo_root)"

COMPOSE_FILE="$(ombre_compose_file)"
HEALTH_URL="${HEALTH_URL:-$(ombre_default_health_url "${COMPOSE_FILE}")}"

echo "Repo: $(pwd)"
echo "Compose: ${COMPOSE_FILE}"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Pull latest code..."
  git pull --ff-only
fi

echo "Update containers..."
if grep -Eq '^[[:space:]]*build:' "${COMPOSE_FILE}"; then
  ombre_compose -f "${COMPOSE_FILE}" up -d --build --remove-orphans
else
  ombre_compose -f "${COMPOSE_FILE}" pull
  ombre_compose -f "${COMPOSE_FILE}" up -d --remove-orphans
fi

ombre_compose -f "${COMPOSE_FILE}" ps
ombre_wait_for_health "${HEALTH_URL}" "${HEALTH_TRIES:-30}" "${HEALTH_DELAY:-2}"

echo "Update done."
