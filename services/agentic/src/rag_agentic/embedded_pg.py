"""A private PostgreSQL server inside the agent container, fed from S3.

The index lives in S3 only, as a ``pg_dump``. A dump is not queryable -- reading
Postgres data needs a Postgres process -- so the agent runs its own: one server per
container, started on first use under the writable cache dir, with one database per
tenant restored from that tenant's dump.

It is a disposable read replica of S3, which shapes every setting below:

* ``fsync``/``full_page_writes``/``synchronous_commit`` are off. A crash loses
  nothing that matters; the database is rebuilt from S3 on the next load.
* No TCP listener at all (``listen_addresses=''``). Clients connect over a Unix
  socket inside the container, so nothing outside it can reach the server.
* ``maintenance_work_mem`` is raised because rebuilding the HNSW vector index is
  most of a restore's cost.
* ``dynamic_shared_memory_type = mmap``: a parallel index build otherwise sizes a
  segment in ``/dev/shm``, which containers cap (Docker: 64 MB) far below the
  ~250 MB a build asks for -- restores failed with "No space left on device".
  mmap backs it with files in the data directory instead.

Linux only (Unix sockets, and PostgreSQL refuses to run as root). The container
image installs PostgreSQL 16 + pgvector and runs as a non-root user. Local Windows
dev uses ``AGENTIC_POSTGRES_SOURCE=dsn`` against the docker-compose database instead.
"""

from __future__ import annotations

import asyncio
import glob
import os
import shutil
import subprocess
from pathlib import Path

from rag_core.logging import get_logger

_log = get_logger("rag_agentic.embedded_pg")

SUPERUSER = "agent"
_START_TIMEOUT_S = 120

# Appended to postgresql.conf after initdb. See the module docstring for why each is safe.
_SERVER_SETTINGS = """
listen_addresses = ''
unix_socket_directories = '{socket_dir}'
fsync = off
full_page_writes = off
synchronous_commit = off
max_connections = 60
shared_buffers = 128MB
maintenance_work_mem = 256MB
max_parallel_maintenance_workers = 2
dynamic_shared_memory_type = mmap
log_min_messages = warning
"""


class EmbeddedPostgresError(RuntimeError):
    """The embedded server could not be found, started, or restored into."""


def find_bin_dir(configured: str = "") -> Path:
    """Directory holding ``initdb``/``pg_ctl``/``pg_restore``.

    Debian installs these under ``/usr/lib/postgresql/<major>/bin`` and puts only
    wrappers on PATH, so that tree is searched too (newest major first).
    """
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    if found := shutil.which("pg_ctl"):
        candidates.append(Path(found).parent)
    majors = sorted(
        glob.glob("/usr/lib/postgresql/*/bin"),
        key=lambda p: int(Path(p).parent.name) if Path(p).parent.name.isdigit() else 0,
        reverse=True,
    )
    candidates.extend(Path(p) for p in majors)
    for directory in candidates:
        if (directory / "initdb").exists() and (directory / "pg_ctl").exists():
            return directory
    raise EmbeddedPostgresError(
        "PostgreSQL binaries (initdb, pg_ctl, pg_restore) not found; install PostgreSQL 16 "
        "+ pgvector in the image or set AGENTIC_POSTGRES_BIN_DIR. For local dev without "
        "them, set AGENTIC_POSTGRES_SOURCE=dsn and AGENTIC_POSTGRES_DSN."
    )


class EmbeddedPostgres:
    """One private PostgreSQL server; a database per tenant, restored from a dump."""

    def __init__(self, root: Path, *, bin_dir: str = "", restore_jobs: int = 2) -> None:
        self._root = root
        self._data = root / "data"
        # Kept short on purpose: a Unix socket path is limited to ~107 bytes.
        self._socket_dir = root / "s"
        self._log_file = root / "server.log"
        self._configured_bin = bin_dir
        self._bin: Path | None = None
        self._restore_jobs = restore_jobs
        self._lock = asyncio.Lock()
        self._started = False

    def dsn(self, database: str) -> str:
        """libpq URI connecting over the private Unix socket."""
        return f"postgresql://{SUPERUSER}@/{database}?host={self._socket_dir}"

    # ---- lifecycle ----------------------------------------------------------

    async def ensure_started(self) -> None:
        if self._started:
            return
        async with self._lock:
            if self._started:
                return
            await asyncio.to_thread(self._start_sync)
            self._started = True

    def _start_sync(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            raise EmbeddedPostgresError(
                "PostgreSQL refuses to run as root; run the agent container as a non-root user"
            )
        self._bin = find_bin_dir(self._configured_bin)
        self._root.mkdir(parents=True, exist_ok=True)
        self._socket_dir.mkdir(parents=True, exist_ok=True)

        if not (self._data / "PG_VERSION").exists():
            # A half-initialised directory from a crashed start is unusable; start over.
            shutil.rmtree(self._data, ignore_errors=True)
            self._run(
                "initdb",
                "-D", str(self._data),
                f"--username={SUPERUSER}",
                "--auth=trust",  # socket-only, inside the container: no network exposure
                "--encoding=UTF8",
                "--locale=C",
                "--no-sync",
            )
            with (self._data / "postgresql.conf").open("a", encoding="utf-8") as conf:
                conf.write(_SERVER_SETTINGS.format(socket_dir=self._socket_dir))

        self._run(
            "pg_ctl",
            "-D", str(self._data),
            "-l", str(self._log_file),
            "-w", "-t", str(_START_TIMEOUT_S),
            "start",
        )
        _log.info("embedded_postgres_started", root=str(self._root), bin=str(self._bin))

    async def aclose(self) -> None:
        async with self._lock:
            if not self._started:
                return
            self._started = False
            await asyncio.to_thread(
                self._run, "pg_ctl", "-D", str(self._data), "-m", "fast", "-w", "stop", check=False
            )
            _log.info("embedded_postgres_stopped")

    # ---- per-tenant databases -----------------------------------------------

    async def restore(self, database: str, dump_path: Path) -> None:
        """(Re)create ``database`` and restore ``dump_path`` into it."""
        await self.ensure_started()
        await self._admin(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
        await self._admin(f'CREATE DATABASE "{database}"')
        try:
            await asyncio.to_thread(
                self._run,
                "pg_restore",
                "-h", str(self._socket_dir),
                "-U", SUPERUSER,
                "-d", database,
                "--no-owner",
                "--no-privileges",
                # A restore into a fresh database emits no errors, so any error is real:
                # serving a partially restored index would silently miss documents.
                "--exit-on-error",
                "-j", str(self._restore_jobs),
                str(dump_path),
            )
        except EmbeddedPostgresError:
            await self._admin(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')
            raise

    async def drop(self, database: str) -> None:
        if self._started:
            await self._admin(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')

    async def _admin(self, sql: str) -> None:
        """Run a statement that cannot be inside a transaction (CREATE/DROP DATABASE)."""
        import psycopg

        def _exec() -> None:
            with psycopg.connect(self.dsn("postgres"), autocommit=True) as conn:
                # Database names are generated hashes, never caller input.
                conn.execute(sql)

        await asyncio.to_thread(_exec)

    # ---- plumbing -----------------------------------------------------------

    def _run(self, tool: str, *args: str, check: bool = True) -> None:
        binary = (self._bin or find_bin_dir(self._configured_bin)) / tool
        proc = subprocess.run(
            [str(binary), *args], capture_output=True, text=True, check=False
        )
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()[-800:]
            if tool == "pg_ctl" and self._log_file.exists():
                detail += "\n--- server.log ---\n" + self._log_file.read_text(errors="replace")[-800:]
            raise EmbeddedPostgresError(f"{tool} failed (exit {proc.returncode}): {detail}")
