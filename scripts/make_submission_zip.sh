#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 TEAMNAME_AGENT /absolute/path/to/checkpoint-N"
  exit 1
fi

TEAM_DIR="$1"
CKPT_FILE="$2"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="${REPO_DIR}/ray_ppo_agent"

if [[ ! -f "${CKPT_FILE}" ]]; then
  echo "Checkpoint file not found: ${CKPT_FILE}"
  exit 1
fi

rm -rf "${REPO_DIR:?}/${TEAM_DIR}"
cp -r "${TEMPLATE_DIR}" "${REPO_DIR}/${TEAM_DIR}"

CKPT_BASENAME="$(basename "${CKPT_FILE}")"
cp "${CKPT_FILE}" "${REPO_DIR}/${TEAM_DIR}/${CKPT_BASENAME}"

if [[ -f "${CKPT_FILE}.tune_metadata" ]]; then
  cp "${CKPT_FILE}.tune_metadata" "${REPO_DIR}/${TEAM_DIR}/${CKPT_BASENAME}.tune_metadata"
fi

echo "${CKPT_BASENAME}" > "${REPO_DIR}/${TEAM_DIR}/checkpoint_path.txt"

cd "${REPO_DIR}"
rm -f "${TEAM_DIR}.zip"
zip -r "${TEAM_DIR}.zip" "${TEAM_DIR}" >/dev/null

echo "Created: ${REPO_DIR}/${TEAM_DIR}.zip"
echo "Checkpoint inside zip: ${CKPT_BASENAME}"
