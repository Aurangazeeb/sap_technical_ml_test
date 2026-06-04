#!/usr/bin/env bash
# Containerized startup script — builds and runs via Docker.
# Usage: ./start_container.sh [--build-only]
#
# Flags:
#   --build-only   Build the image but don't run it
#
# Override settings at runtime with -e flags:
#   ./start_container.sh  (uses defaults from Dockerfile ENV)
#   SAP_API_PORT=9000 ./start_container.sh  (change host-mapped port)

set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="sap-cxii-similarity"
CONTAINER_NAME="sap-cxii-similarity-api"
HOST_PORT="${SAP_API_PORT:-8000}"

echo "▶ Building Docker image: ${IMAGE_NAME}…"
docker build -t "${IMAGE_NAME}" .

if [[ "${1:-}" == "--build-only" ]]; then
    echo "✓ Image built successfully. Skipping run."
    exit 0
fi

# Stop any existing container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "  Stopping existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
fi

echo "▶ Running container: ${CONTAINER_NAME}"
echo "  API:  http://localhost:${HOST_PORT}/docs"
echo ""

exec docker run --rm \
    --name "${CONTAINER_NAME}" \
    -p "${HOST_PORT}:8000" \
    -e SAP_API_HOST=0.0.0.0 \
    -e SAP_API_PORT=8000 \
    "${IMAGE_NAME}"
