from __future__ import annotations

import sys

try:
    import ApplicationServices
    import Quartz
except ImportError as exc:
    print(f"Missing macOS frameworks: {exc}")
    raise SystemExit(1)


def main() -> None:
    print(f"executable: {sys.executable}")
    print(f"Accessibility trusted: {bool(ApplicationServices.AXIsProcessTrusted())}")
    print(f"Input Monitoring granted: {bool(Quartz.CGPreflightListenEventAccess())}")
    print(f"Post Event Access granted: {bool(Quartz.CGPreflightPostEventAccess())}")


if __name__ == "__main__":
    main()
