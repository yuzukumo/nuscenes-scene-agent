from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from nusc_scene_agent.nuplan_replay import inspect_nuplan_dataset


DATASET_BACKEND_INVENTORY_SCHEMA = "dataset_backend_inventory_v1"


def inspect_dataset_backends(
    *,
    nuscenes_root: Path = Path("data/sets/nuscenes"),
    nuplan_root: Path = Path("data/nuplan/dataset"),
    index_root: Path = Path("artifacts/index"),
) -> Dict[str, Any]:
    return {
        "schema": DATASET_BACKEND_INVENTORY_SCHEMA,
        "backends": {
            "nuscenes": inspect_nuscenes_backend(nuscenes_root=nuscenes_root, index_root=index_root),
            "nuplan": inspect_nuplan_backend(nuplan_root=nuplan_root),
        },
    }


def inspect_nuscenes_backend(
    *,
    nuscenes_root: Path = Path("data/sets/nuscenes"),
    index_root: Path = Path("artifacts/index"),
) -> Dict[str, Any]:
    nuscenes_root = Path(nuscenes_root)
    version_payloads: Dict[str, Any] = {}
    for version in ["v1.0-mini", "v1.0-trainval"]:
        version_dir = nuscenes_root / version
        version_payloads[version] = _inspect_nuscenes_version(version_dir)

    index_payloads = {}
    if Path(index_root).exists():
        for index_path in sorted(Path(index_root).glob("*.sqlite")):
            index_payloads[index_path.name] = {
                "path": str(index_path),
                "exists": True,
                "size_bytes": index_path.stat().st_size,
            }

    map_roots = [nuscenes_root / "maps", nuscenes_root / "expansion"]
    map_counts = {root.name: _count_json_files(root) for root in map_roots}
    return {
        "dataset_root": str(nuscenes_root),
        "exists": nuscenes_root.exists(),
        "versions": version_payloads,
        "map_file_count": sum(map_counts.values()),
        "map_file_counts": map_counts,
        "indices": index_payloads,
    }


def inspect_nuplan_backend(nuplan_root: Path = Path("data/nuplan/dataset")) -> Dict[str, Any]:
    return inspect_nuplan_dataset(Path(nuplan_root))


def write_dataset_backend_inventory(inventory: Mapping[str, Any], output_path: Path) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(inventory)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _inspect_nuscenes_version(version_dir: Path) -> Dict[str, Any]:
    files = {
        "scene": version_dir / "scene.json",
        "sample": version_dir / "sample.json",
        "sample_annotation": version_dir / "sample_annotation.json",
        "instance": version_dir / "instance.json",
        "log": version_dir / "log.json",
    }
    counts = {
        name: _safe_json_count(path)
        for name, path in files.items()
    }
    return {
        "path": str(version_dir),
        "exists": version_dir.exists(),
        "counts": counts,
    }


def _safe_json_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _count_json_files(root: Path) -> int:
    return len(list(root.rglob("*.json"))) if root.exists() else 0
