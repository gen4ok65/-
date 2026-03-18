from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class StoredAPIKey:
    label: str
    key_prefix: str
    key_hash: str
    created_at: str


class APIKeyStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[StoredAPIKey]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [StoredAPIKey(**item) for item in payload]

    def _write(self, items: list[StoredAPIKey]) -> None:
        self.path.write_text(
            json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _hash(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def create_key(self, label: str) -> str:
        safe_label = (label or "default").strip() or "default"
        raw_key = f"mla_{secrets.token_urlsafe(32)}"
        items = self._read()
        items.append(
            StoredAPIKey(
                label=safe_label,
                key_prefix=raw_key[:12],
                key_hash=self._hash(raw_key),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        self._write(items)
        return raw_key

    def validate(self, raw_key: str | None) -> bool:
        if not raw_key:
            return False
        expected = self._hash(raw_key)
        return any(item.key_hash == expected for item in self._read())

    def list_keys(self) -> list[dict[str, str]]:
        return [
            {
                "label": item.label,
                "key_prefix": item.key_prefix,
                "created_at": item.created_at,
            }
            for item in self._read()
        ]
