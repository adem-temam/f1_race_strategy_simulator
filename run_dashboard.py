#!/usr/bin/env python3
"""Phase 6: Standalone launcher for the F1 Race Strategy Simulator Interactive Dashboard."""

import argparse
import socket
import sys
import threading
import time
import webbrowser

import uvicorn


def find_free_port(start_port: int = 8000) -> int:
    """Find an available TCP port starting from start_port."""
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def open_browser_delayed(url: str, delay: float = 1.0) -> None:
    """Open web browser after a short delay allowing server initialization."""
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    parser = argparse.ArgumentParser(
        description="F1 Race Strategy Simulator - Interactive Dashboard Launcher"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser",
    )
    args = parser.parse_args()

    port = args.port
    # Check if requested port is available, else find free
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((args.host, port)) == 0:
            port = find_free_port(port + 1)
            print(f"[NOTE] Port {args.port} in use. Re-binding to free port {port}.")

    url = f"http://{args.host}:{port}"

    print("=" * 75)
    print("  F1 RACE STRATEGY SIMULATOR - INTERACTIVE WEB DASHBOARD (PHASE 6)")
    print("=" * 75)
    print(f"  Server URL:    {url}")
    print(f"  Documentation: {url}/docs")
    print("=" * 75)
    print("  Press CTRL+C to stop the dashboard server.\n")

    if not args.no_browser:
        open_browser_delayed(url, delay=1.2)

    try:
        uvicorn.run(
            "src.dashboard.app:app",
            host=args.host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nDashboard server terminated cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
