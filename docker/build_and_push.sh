#!/usr/bin/env bash
# Build the image and push to a registry (GHCR by default).
#   REGISTRY=ghcr.io/lyttonfeng ./build_and_push.sh
#   REGISTRY=docker.io/youruser  TAG=v1 ./build_and_push.sh
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/lyttonfeng}"
IMAGE="${IMAGE:-naive-meeting-analysis}"
TAG="${TAG:-cu130-vllm0.22-$(date +%Y%m%d)}"
REF="${REGISTRY}/${IMAGE}:${TAG}"

cd "$(dirname "$0")"

echo "[build] ${REF}"
# Requires buildkit; build on a CUDA host or with --platform linux/amd64.
docker build --platform linux/amd64 -t "${REF}" -t "${REGISTRY}/${IMAGE}:latest" .

echo "[push] ${REF}"
# GHCR login first:  echo \$GHCR_PAT | docker login ghcr.io -u <user> --password-stdin
docker push "${REF}"
docker push "${REGISTRY}/${IMAGE}:latest"

echo "[done] pull with:  docker pull ${REF}"
