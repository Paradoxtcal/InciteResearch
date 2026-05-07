"""
Persistence / checkpoint selection.
Uses MemorySaver by default, or SqliteSaver / PostgresSaver when configured.
"""
import os
import json
import base64
from pathlib import Path
from threading import RLock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import WRITES_IDX_MAP


class FileCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(self, root_dir: str | Path, *, serde=None) -> None:
        super().__init__(serde=serde)
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._cache: dict[str, dict] = {}

    def _path_for_thread(self, thread_id: str) -> Path:
        safe = (thread_id or "unknown").replace("/", "_").replace("\\", "_")
        return self._root / f"{safe}.json"

    def _encode_typed(self, typed: tuple[str, bytes]) -> dict:
        t, b = typed
        return {"t": t, "b": base64.b64encode(b).decode("ascii")}

    def _decode_typed(self, data: dict) -> tuple[str, bytes]:
        t = str(data.get("t", "empty"))
        b64 = str(data.get("b", ""))
        try:
            raw = base64.b64decode(b64.encode("ascii")) if b64 else b""
        except Exception:
            raw = b""
        return (t, raw)

    def _load_thread(self, thread_id: str) -> dict:
        with self._lock:
            if thread_id in self._cache:
                return self._cache[thread_id]
            path = self._path_for_thread(thread_id)
            if not path.exists():
                data = {"latest_id": None, "storage": {}, "writes": {}, "blobs": {}}
                self._cache[thread_id] = data
                return data
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("invalid checkpoint file")
            except Exception:
                raw = {"latest_id": None, "storage": {}, "writes": {}, "blobs": {}}
            raw.setdefault("latest_id", None)
            raw.setdefault("storage", {})
            raw.setdefault("writes", {})
            raw.setdefault("blobs", {})
            self._cache[thread_id] = raw
            return raw

    def _save_thread(self, thread_id: str) -> None:
        with self._lock:
            data = self._cache.get(thread_id)
            if not isinstance(data, dict):
                return
            path = self._path_for_thread(thread_id)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_blobs(self, thread_id: str, checkpoint_ns: str, versions: dict) -> dict:
        data = self._load_thread(thread_id)
        blobs = data.get("blobs") or {}
        out: dict = {}
        for ch, ver in (versions or {}).items():
            key = f"{checkpoint_ns}|{ch}|{ver}"
            if key not in blobs:
                continue
            typed = self._decode_typed(blobs[key])
            if typed[0] != "empty":
                out[ch] = self.serde.loads_typed(typed)
        return out

    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        data = self._load_thread(thread_id)
        storage = data.get("storage") or {}
        ns_map = storage.get(checkpoint_ns) or {}

        checkpoint_id = get_checkpoint_id(config) or data.get("latest_id")
        if not checkpoint_id:
            return None
        saved = ns_map.get(checkpoint_id)
        if not saved:
            return None

        checkpoint_enc, metadata_enc, parent_checkpoint_id = saved
        try:
            checkpoint_ = self.serde.loads_typed(self._decode_typed(checkpoint_enc))
        except Exception:
            return None

        try:
            metadata_ = self.serde.loads_typed(self._decode_typed(metadata_enc))
        except Exception:
            metadata_ = {}

        writes_root = (data.get("writes") or {}).get(f"{checkpoint_ns}|{checkpoint_id}") or {}
        pending_writes = []
        for _, w in writes_root.items():
            if not isinstance(w, list) or len(w) < 4:
                continue
            task_id, channel, typed_v, _task_path = w
            try:
                pending_writes.append((task_id, channel, self.serde.loads_typed(self._decode_typed(typed_v))))
            except Exception:
                continue

        parent_config = (
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
            if parent_checkpoint_id
            else None
        )
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint_,
                "channel_values": self._load_blobs(thread_id, checkpoint_ns, checkpoint_.get("channel_versions") or {}),
            },
            metadata=metadata_,
            pending_writes=pending_writes,
            parent_config=parent_config,
        )

    def list(
        self,
        config: dict | None,
        *,
        filter: dict | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ):
        thread_ids = (config["configurable"]["thread_id"],) if config else tuple(self._cache.keys())
        config_checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        yielded = 0
        for thread_id in thread_ids:
            data = self._load_thread(thread_id)
            storage = data.get("storage") or {}
            for checkpoint_ns, ns_map in storage.items():
                if config_checkpoint_ns is not None and checkpoint_ns != config_checkpoint_ns:
                    continue
                if not isinstance(ns_map, dict):
                    continue
                for checkpoint_id in list(ns_map.keys())[::-1]:
                    if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                        continue
                    if before_checkpoint_id and checkpoint_id >= before_checkpoint_id:
                        continue
                    tup = self.get_tuple(
                        {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}}
                    )
                    if not tup:
                        continue
                    if filter:
                        md = tup.metadata or {}
                        ok = all(md.get(k) == v for k, v in filter.items())
                        if not ok:
                            continue
                    yield tup
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return

    def put(self, config: dict, checkpoint: dict, metadata: dict, new_versions: dict) -> dict:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        values: dict = c.pop("channel_values", {})  # type: ignore[assignment]

        data = self._load_thread(thread_id)
        blobs = data.get("blobs") or {}
        for k, v in (new_versions or {}).items():
            typed = self.serde.dumps_typed(values[k]) if k in values else ("empty", b"")
            blobs[f"{checkpoint_ns}|{k}|{v}"] = self._encode_typed(typed)
        data["blobs"] = blobs

        storage = data.get("storage") or {}
        ns_map = storage.get(checkpoint_ns) or {}
        ns_map[checkpoint["id"]] = (
            self._encode_typed(self.serde.dumps_typed(c)),
            self._encode_typed(self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))),
            config["configurable"].get("checkpoint_id"),
        )
        storage[checkpoint_ns] = ns_map
        data["storage"] = storage
        data["latest_id"] = checkpoint["id"]
        self._save_thread(thread_id)

        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint["id"]}}

    def put_writes(self, config: dict, writes, task_id: str, task_path: str = "") -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        data = self._load_thread(thread_id)
        writes_root = data.get("writes") or {}
        outer_key = f"{checkpoint_ns}|{checkpoint_id}"
        outer = writes_root.get(outer_key) or {}

        for idx, (c, v) in enumerate(writes or []):
            inner_idx = WRITES_IDX_MAP.get(c, idx)
            inner_key = f"{task_id}|{inner_idx}"
            if inner_idx >= 0 and inner_key in outer:
                continue
            outer[inner_key] = [task_id, c, self._encode_typed(self.serde.dumps_typed(v)), task_path]

        writes_root[outer_key] = outer
        data["writes"] = writes_root
        self._save_thread(thread_id)

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._cache.pop(thread_id, None)
            path = self._path_for_thread(thread_id)
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

    def delete_for_runs(self, run_ids) -> None:
        return

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        src = self._path_for_thread(source_thread_id)
        dst = self._path_for_thread(target_thread_id)
        if not src.exists():
            return
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    def prune(self, thread_ids, *, strategy: str = "keep_latest") -> None:
        if strategy != "keep_latest":
            for tid in thread_ids:
                self.delete_thread(tid)
            return
        for tid in thread_ids:
            data = self._load_thread(tid)
            latest = data.get("latest_id")
            if not latest:
                continue
            storage = data.get("storage") or {}
            for ns, ns_map in list(storage.items()):
                if not isinstance(ns_map, dict):
                    continue
                for cid in list(ns_map.keys()):
                    if cid != latest:
                        ns_map.pop(cid, None)
                storage[ns] = ns_map
            data["storage"] = storage
            self._save_thread(tid)

def get_checkpointer():
    """
    Pick a checkpointer based on environment variables.
    Uses memory when DATABASE_URL is not set.
    """
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("sqlite"):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            db_path = db_url.replace("sqlite:///", "")
            return SqliteSaver.from_conn_string(db_path)
        except ImportError:
            pass
    if db_url.startswith("postgresql"):
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            return PostgresSaver.from_conn_string(db_url)
        except ImportError:
            pass
    try:
        root = Path(os.environ.get("RESEARCH_AGENT_CHECKPOINT_DIR", ".research_agent_cache/checkpoints"))
        return FileCheckpointSaver(root)
    except Exception:
        return MemorySaver()
