"""System trust-store integration.

Corporate networks often terminate TLS with a MITM proxy whose root CA lives in
the OS trust store but NOT in Python's bundled ``certifi`` roots — so boto3/httpx
fail with certificate errors. ``truststore`` makes Python's ``ssl`` use the OS
trust store, which includes that corporate CA. This is the programmatic
equivalent of uv's ``--system-certs``.

Called once from ``AwsSessionFactory`` so every AWS client benefits; safe and
idempotent, and a no-op if ``truststore`` isn't installed.
"""

from __future__ import annotations

_injected = False


def enable_system_trust() -> bool:
    """Route Python TLS verification through the OS trust store. Idempotent."""
    global _injected
    if _injected:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
        _injected = True
        return True
    except Exception:
        return False
