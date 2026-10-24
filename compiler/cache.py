from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading

CacheValue = tuple[str, str, str]
_FIELDS = ('ptx', 'unoptimized', 'optimized')
_VERSION = 1
_MAX_RECORD_BYTES = 64 * 1024 * 1024
_KEY = re.compile(r'[0-9a-f]{64}')
_PTX_HEADERS = tuple(re.compile(pattern, re.MULTILINE) for pattern in (
    r'^\s*\.version[ \t]+\d+\.\d+[ \t]*$',
    r'^\s*\.target[ \t]+sm_\d+[^\n]*$',
    r'^\s*\.address_size[ \t]+64[ \t]*$',
    r'^\s*(?:\.visible[ \t]+)?\.entry[ \t]+',
))


def _valid(value: CacheValue) -> bool:
    return (len(value) == 3 and all(type(text) is str and text for text in value)
            and all(pattern.search(value[0]) for pattern in _PTX_HEADERS)
            and all('target triple = "nvptx64-nvidia-cuda"' in text and 'define ptx_kernel void ' in text for text in value[1:]))


def _encode(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _record(key: str, value: CacheValue) -> dict:
    payload = {'version': _VERSION, 'key': key, **dict(zip(_FIELDS, value, strict=True))}
    return {**payload, 'sha256': hashlib.sha256(_encode(payload)).hexdigest()}


def _decode(data: object, key: str) -> CacheValue | None:
    if type(data) is not dict or set(data) != {'version', 'key', 'sha256', *_FIELDS}:
        return None
    if type(data['version']) is not int or data['version'] != _VERSION or data['key'] != key:
        return None
    value = tuple(data[field] for field in _FIELDS)
    if not _valid(value) or type(data['sha256']) is not str or not _KEY.fullmatch(data['sha256']):
        return None
    payload = {name: item for name, item in data.items() if name != 'sha256'}
    if hashlib.sha256(_encode(payload)).hexdigest() != data['sha256']:
        return None
    return value


class CompilationCache:
    def __init__(self, max_entries: int = 128, max_bytes: int = 64 * 1024 * 1024) -> None:
        if type(max_entries) is not int or type(max_bytes) is not int or max_entries < 1 or max_bytes < 1:
            raise ValueError('cache limits must be positive integers')
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._values: OrderedDict[str, tuple[CacheValue, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._bytes = 0

    def _remember(self, key: str, value: CacheValue) -> None:
        size = sum(sys.getsizeof(text) for text in value)
        with self._lock:
            previous = self._values.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            if size > self.max_bytes:
                return
            self._values[key] = value, size
            self._bytes += size
            while len(self._values) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, removed) = self._values.popitem(last=False)
                self._bytes -= removed

    def get(self, key: str, root: str | None = None) -> CacheValue | None:
        if not _KEY.fullmatch(key):
            return None
        with self._lock:
            hit = self._values.get(key)
            if hit is not None:
                self._values.move_to_end(key)
                return hit[0]
        if not root:
            return None
        try:
            with (Path(root) / (key + '.json')).open('rb') as stream:
                payload = stream.read(_MAX_RECORD_BYTES + 1)
            if len(payload) > _MAX_RECORD_BYTES:
                return None
            value = _decode(json.loads(payload), key)
        except (OSError, ValueError, TypeError, RecursionError):
            return None
        if value is not None:
            self._remember(key, value)
        return value

    def put(self, key: str, value: CacheValue, root: str | None = None) -> None:
        if not _KEY.fullmatch(key) or not _valid(value):
            return
        self._remember(key, value)
        if not root or sum(len(text) for text in value) > _MAX_RECORD_BYTES:
            return
        temporary: str | None = None
        try:
            payload = _encode(_record(key, value))
            if len(payload) > _MAX_RECORD_BYTES:
                return
            directory = Path(root)
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='wb', prefix='.' + key + '.', suffix='.tmp', dir=directory, delete=False) as stream:
                temporary = stream.name
                stream.write(payload)
            os.replace(temporary, directory / (key + '.json'))
        except (OSError, ValueError):
            pass
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
