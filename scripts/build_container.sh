#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${MI50_IMAGE:-mi50-rocm10-builder:10.0.0-mi50.5}"
BASE_IMAGE="${MI50_BASE_IMAGE:-ubuntu:24.04@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517}"
METADATA="${MI50_CONTAINER_METADATA:-${ROOT_DIR}/out/container-image-provenance.json}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build the reproducible container" >&2
  exit 2
fi

mkdir -p "$(dirname "${METADATA}")"
IID_FILE="$(mktemp)"
trap 'rm -f "${IID_FILE}"' EXIT
docker build --pull \
  --file "${ROOT_DIR}/containers/ubuntu-24.04/Dockerfile" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --iidfile "${IID_FILE}" \
  --tag "${IMAGE}" \
  "${ROOT_DIR}"
python3 "${ROOT_DIR}/scripts/write_container_provenance.py" \
  --output "${METADATA}" \
  --image "${IMAGE}" \
  --base-image "${BASE_IMAGE}" \
  --iid-file "${IID_FILE}"
echo "built ${IMAGE}; GPU execution remains pending-hardware"
