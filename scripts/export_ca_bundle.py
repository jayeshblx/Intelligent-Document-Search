"""Export a CA bundle that trusts BOTH public roots and the corporate proxy CA.

Behind a TLS-intercepting corporate proxy, Python's bundled ``certifi`` roots
don't include the proxy's root CA, so boto3 fails to verify. This writes a
combined PEM (certifi + the Windows trust stores) and prints the path to set as
``AWS__CA_BUNDLE`` — a version-independent alternative to ``truststore`` (which
recurses on Python 3.14).

    uv run python scripts/export_ca_bundle.py
    # then set in .env:  AWS__CA_BUNDLE=./certs/ca-bundle.pem
"""

from __future__ import annotations

import ssl
from pathlib import Path

import certifi

OUT = Path("certs/ca-bundle.pem")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sections: list[str] = [certifi.contents()]
    seen: set[str] = set()
    added = 0

    for store in ("ROOT", "CA"):
        try:
            certs = ssl.enum_certificates(store)  # Windows-only
        except (AttributeError, OSError):
            continue
        for cert_bytes, encoding, _trust in certs:
            if encoding != "x509_asn":
                continue
            pem = ssl.DER_cert_to_PEM_cert(cert_bytes)
            if pem not in seen:
                seen.add(pem)
                sections.append(pem)
                added += 1

    OUT.write_text("\n".join(sections), encoding="ascii")
    print(f"Wrote {OUT} (certifi + {added} OS certs).")
    print(f"Set in .env:  AWS__CA_BUNDLE=./{OUT.as_posix()}")


if __name__ == "__main__":
    main()
