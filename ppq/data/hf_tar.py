"""Seekable HTTP range reader + tar member index (stdlib tarfile only).

Lets the pipeline read individual members out of large remote tar shards
(e.g. the FLAC/JSON of one session) without downloading the whole shard -
in particular skipping the MP4 video members, which are ~75% of the bytes
in the video dataset.

The HTTP layer is a single ``fetch(offset, nbytes) -> bytes`` callable, so
tests can back it with a local file or a small range-capable HTTP server.
"""
from __future__ import annotations

import collections
import io
import json
import logging
import os
import tarfile
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB read-ahead chunks for sequential scans


def resolve_hf_url(repo_id: str, path: str, token: str | None = None,
                   repo_type: str = "dataset") -> tuple[str, int]:
    """Return (final CDN url, content length) for a file in a HF repo."""
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(repo_id, path, repo_type=repo_type)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.head(url, headers=headers, allow_redirects=True, timeout=60)
    if r.status_code == 401:
        raise PermissionError(
            f"access denied (401) for {repo_id}/{path}. Accept the dataset "
            "terms on its Hugging Face page with this account, then retry.")
    r.raise_for_status()
    size = int(r.headers["Content-Length"])
    return r.url, size


def make_fetch(url: str, session: requests.Session | None = None,
               retries: int = 4):
    """Range-GET fetcher with retries. Returns fetch(offset, nbytes) -> bytes."""
    sess = session or requests.Session()

    def fetch(off: int, n: int) -> bytes:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                r = sess.get(url, headers={"Range": f"bytes={off}-{off + n - 1}"},
                             timeout=300)
                if r.status_code in (200, 206):
                    return r.content
                last = RuntimeError(f"range GET {url} -> HTTP {r.status_code}")
            except requests.RequestException as e:  # noqa: PERF203
                last = e
            time.sleep(1.0 + attempt)
        raise last  # type: ignore[misc]

    return fetch


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file backed by a range fetch callable.

    Reads are served from a small LRU of 1 MiB chunks so tar header scans
    (many small reads, mostly forward) don't re-download.
    """

    def __init__(self, size: int, fetch, n_cache: int = 16):
        self.size = int(size)
        self._fetch = fetch
        self._pos = 0
        self._n_cache = n_cache
        self._cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        else:
            raise ValueError(f"bad whence: {whence}")
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def _chunk(self, idx: int) -> bytes:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        off = idx * CHUNK
        data = self._fetch(off, min(CHUNK, self.size - off))
        self._cache[idx] = data
        while len(self._cache) > self._n_cache:
            self._cache.popitem(last=False)
        return data

    def readinto(self, b) -> int:
        n = len(b)
        if self._pos >= self.size:
            return 0
        n = min(n, self.size - self._pos)
        mv = memoryview(b)
        done = 0
        while done < n:
            idx = (self._pos + done) // CHUNK
            chunk = self._chunk(idx)
            inner = (self._pos + done) - idx * CHUNK
            take = min(len(chunk) - inner, n - done)
            if take <= 0:
                break
            mv[done:done + take] = chunk[inner:inner + take]
            done += take
        self._pos += done
        return done


def tar_index(size: int, fetch, cache_path: str | Path | None = None
              ) -> dict[str, tuple[int, int]]:
    """member name -> (data offset, size). Scans only tar header blocks
    (member data is seeked over, not downloaded). Cached as JSON."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                return {k: (int(v[0]), int(v[1])) for k, v in raw.items()}
            except (json.JSONDecodeError, OSError, ValueError):
                logger.warning("ignoring unreadable tar index cache %s", cache_path)
    idx: dict[str, tuple[int, int]] = {}
    fileobj = io.BufferedReader(HTTPRangeFile(size, fetch), buffer_size=CHUNK)
    with tarfile.open(fileobj=fileobj, mode="r:") as tf:
        for m in tf:
            if m.isfile():
                idx[m.name] = (m.offset_data, m.size)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        tmp.write_text(json.dumps(idx), encoding="utf-8")
        os.replace(tmp, cache_path)
    return idx


def read_member(fetch, index: dict[str, tuple[int, int]], name: str) -> bytes:
    """Fetch one member's bytes with a single range GET."""
    off, n = index[name]
    return fetch(off, n)
