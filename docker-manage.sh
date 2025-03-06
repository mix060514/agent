#!/bin/bash
# Script to manage Docker operations

case "$1" in
  build)
    echo "Building Docker image..."
    docker build -t agent-app -f docker/Dockerfile .
    ;;
  run)
    echo "Running Docker container..."
    docker run --name agent-container agent-app
    ;;
  dev)
    echo "Starting development container..."
    docker run -it --name agent-dev-container agent-app bash
    ;;
  clean)
    echo "Removing containers and images..."
    docker rm -f agent-container agent-dev-container 2>/dev/null || true
    docker rmi agent-app 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {build|run|dev|clean}"
    exit 1
    ;;
esac

echo "Operation complete!"
