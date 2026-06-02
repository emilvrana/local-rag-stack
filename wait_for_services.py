#!/usr/bin/env python3
"""
Wait for RAG stack services to become ready.

Docker Compose starts containers quickly, but the services inside them
(PostgreSQL, TEI embeddings, Ollama) need time to initialize. This script
polls each service until it responds, with configurable timeouts.

Useful in:
  - CI pipelines: wait before running tests
  - Docker entrypoints: wait for dependencies before starting the app
  - Development: confirm the stack is ready before querying

Usage:
    # Wait for all services (default 120s timeout per service)
    python wait_for_services.py

    # Custom timeout
    python wait_for_services.py --timeout 60

    # Wait for a specific service only
    python wait_for_services.py --service postgres

    # Use in docker-compose (healthcheck or entrypoint)
    python wait_for_services.py --timeout 300 --interval 5
"""

import os
import sys
import time
import argparse
import signal
import socket
import urllib.request
import urllib.error

from dotenv import load_dotenv

load_dotenv()

# Service configuration
SERVICES = {
    "postgres": {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5433")),
        "check": "tcp",
        "description": "PostgreSQL with pgvector",
    },
    "embeddings": {
        "url": os.getenv("EMBEDDING_URL", "http://localhost:8081/embed"),
        "check": "http_post",
        "description": "TEI embeddings service",
    },
    "llm": {
        "url": os.getenv("LLM_URL", "http://localhost:8080/v1"),
        "check": "http_get",
        "path": "/models",
        "description": "Ollama LLM service",
    },
}

# Defaults
DEFAULT_TIMEOUT = 120  # seconds
DEFAULT_INTERVAL = 2  # seconds between retries


def check_tcp(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
        return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def check_http_get(url: str, timeout: int = 5) -> bool:
    """Check if an HTTP GET endpoint responds."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
        return False


def check_http_post(url: str, timeout: int = 5) -> bool:
    """Check if an HTTP POST endpoint responds (lightweight probe)."""
    try:
        data = b'{"inputs": "health"}'
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
        return False


def wait_for_service(name: str, config: dict, timeout: int, interval: int) -> bool:
    """Poll a service until it's ready or timeout expires."""
    desc = config["description"]
    check_type = config["check"]

    # Resolve check parameters
    if check_type == "tcp":
        host, port = config["host"], config["port"]
        check = lambda: check_tcp(host, port)
        label = f"{host}:{port}"
    elif check_type == "http_get":
        url = config["url"] + config.get("path", "")
        check = lambda: check_http_get(url)
        label = url
    elif check_type == "http_post":
        url = config["url"]
        check = lambda: check_http_post(url)
        label = url
    else:
        print(f"  [SKIP] {name}: unknown check type '{check_type}'")
        return True

    print(f"  Waiting for {desc} ({label})...", end="", flush=True)
    start = time.monotonic()
    dots = 0

    while time.monotonic() - start < timeout:
        if check():
            elapsed = time.monotonic() - start
            print(f" OK ({elapsed:.1f}s)")
            return True
        dots += 1
        if dots % 10 == 0:
            print(".", end="", flush=True)
        time.sleep(interval)

    print(f" FAILED (timeout after {timeout}s)")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Wait for local-rag-stack services to become ready"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Max seconds to wait per service (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Seconds between retries (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--service", choices=list(SERVICES.keys()),
        help="Wait for a specific service only (default: all)"
    )
    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        print("\nInterrupted. Services may not be fully ready.")
        sys.exit(1)
    signal.signal(signal.SIGINT, handle_sigint)

    targets = {args.service: SERVICES[args.service]} if args.service else SERVICES

    print(f"Waiting for {len(targets)} service(s) (timeout: {args.timeout}s each)...\n")

    failed = []
    for name, config in targets.items():
        if not wait_for_service(name, config, args.timeout, args.interval):
            failed.append(name)

    print()
    if failed:
        print(f"❌ Services not ready: {', '.join(failed)}")
        print("   Try: docker-compose ps   docker-compose logs <service>")
        sys.exit(1)
    else:
        print("✓ All services ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()