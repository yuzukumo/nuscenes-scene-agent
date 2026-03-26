from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_DATAROOT = Path("data/sets/nuscenes")
TRAINVAL_TABLES_VERSION = "v1.0-trainval"
PREPARE_PROFILES = ("mini", "trainval", "trainval-full", "all")


@dataclass
class ArchiveInventory:
    mini: List[str] = field(default_factory=list)
    trainval_blobs: List[str] = field(default_factory=list)
    trainval_tables: List[str] = field(default_factory=list)
    maps: List[str] = field(default_factory=list)
    can_bus: List[str] = field(default_factory=list)
    legacy: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["mini_ready"] = bool(self.mini and self.maps)
        payload["trainval_ready"] = bool(self.trainval_tables and self.trainval_blobs and self.maps)
        payload["trainval_blob_count"] = len(self.trainval_blobs)
        return payload


def _safe_target(member_name: str, root: Path) -> Path:
    target = (root / member_name).resolve()
    root = root.resolve()
    if root == target or str(target).startswith(str(root) + "/"):
        return target
    raise ValueError("Unsafe archive member path: {0}".format(member_name))


def _extract_tar(archive_path: Path, dataroot: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            _safe_target(member.name, dataroot)
        tar.extractall(path=dataroot)


def _extract_zip(archive_path: Path, dataroot: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            _safe_target(member, dataroot)
        archive.extractall(path=dataroot)


def _archive_contains_version_tables(archive_path: Path, version: str) -> bool:
    lower_name = archive_path.name.lower()
    if version in lower_name and "blobs" not in lower_name:
        return True

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist()[:40]:
                    if member.startswith(version + "/") and member.endswith(".json"):
                        return True
            return False

        with tarfile.open(archive_path, "r:*") as archive:
            for index, member in enumerate(archive):
                if index >= 40:
                    break
                if member.name.startswith(version + "/") and member.name.endswith(".json"):
                    return True
    except Exception:
        return False
    return False


def discover_archive_inventory(workspace: Path) -> ArchiveInventory:
    workspace = workspace.resolve()
    inventory = ArchiveInventory()

    candidates: List[Path] = []
    candidates.extend(sorted((workspace / "archives").rglob("*")))
    candidates.extend(sorted(workspace.glob("*.tgz")))
    candidates.extend(sorted(workspace.glob("*.zip")))
    candidates.extend(sorted(workspace.glob("*.tar")))
    candidates.extend(sorted(workspace.glob("*.7z")))

    unique: List[Path] = []
    seen = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)

    for archive_path in unique:
        relative = str(archive_path.relative_to(workspace))
        name = archive_path.name.lower()

        if name == "v1.0-mini.tgz":
            inventory.mini.append(relative)
        elif "trainval" in name and "blobs" in name:
            inventory.trainval_blobs.append(relative)
        elif "trainval" in name and _archive_contains_version_tables(archive_path, TRAINVAL_TABLES_VERSION):
            inventory.trainval_tables.append(relative)
        elif "map-expansion" in name:
            inventory.maps.append(relative)
        elif "can_bus" in name:
            inventory.can_bus.append(relative)
        elif name.endswith(".7z"):
            inventory.legacy.append(relative)
        elif _archive_contains_version_tables(archive_path, TRAINVAL_TABLES_VERSION):
            inventory.trainval_tables.append(relative)
        else:
            inventory.unknown.append(relative)

    return inventory


def normalize_map_layout(dataroot: Path) -> None:
    dataroot = dataroot.resolve()
    expansion_dir = dataroot / "expansion"
    maps_dir = dataroot / "maps"
    target_expansion_dir = maps_dir / "expansion"

    if not expansion_dir.exists() or target_expansion_dir.exists():
        return

    maps_dir.mkdir(parents=True, exist_ok=True)
    try:
        target_expansion_dir.symlink_to(expansion_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(expansion_dir, target_expansion_dir, dirs_exist_ok=True)


def resolve_archives(
    workspace: Path,
    archives: Optional[Sequence[str]] = None,
    profile: str = "mini",
) -> List[Path]:
    if archives:
        resolved: List[Path] = []
        for item in archives:
            path = Path(item)
            if path.is_absolute():
                resolved.append(path)
                continue

            direct = (workspace / path).resolve()
            if direct.exists():
                resolved.append(direct)
                continue

            archive_relative = (workspace / "archives" / path).resolve()
            if archive_relative.exists():
                resolved.append(archive_relative)
                continue

            resolved.append(direct)
        return resolved

    inventory = discover_archive_inventory(workspace)
    candidates: List[Path] = []
    groups = inventory.to_dict()

    if profile == "mini":
        relative_paths = inventory.mini + inventory.maps
    elif profile == "trainval":
        relative_paths = inventory.trainval_tables + inventory.maps
    elif profile == "trainval-full":
        relative_paths = inventory.trainval_tables + inventory.trainval_blobs + inventory.maps
    elif profile == "all":
        relative_paths = (
            inventory.mini
            + inventory.trainval_tables
            + inventory.trainval_blobs
            + inventory.maps
            + inventory.can_bus
        )
    else:
        raise ValueError("Unsupported archive profile: {0}".format(profile))

    for relative_path in relative_paths:
        candidates.append((workspace / relative_path).resolve())

    resolved: List[Path] = []
    seen = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved_path = path.resolve()
        key = str(resolved_path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(resolved_path)
    return resolved


def _validate_profile_requirements(
    workspace: Path,
    resolved: Sequence[Path],
    profile: str,
) -> None:
    if profile not in {"trainval", "trainval-full", "all"}:
        return

    has_tables = any(_archive_contains_version_tables(path, TRAINVAL_TABLES_VERSION) for path in resolved)
    if has_tables:
        return

    inventory = discover_archive_inventory(workspace)
    raise FileNotFoundError(
        "Trainval preparation requires an archive that contains {0}/*.json tables. "
        "Current inventory has {1} trainval blob archives but no trainval tables archive.".format(
            TRAINVAL_TABLES_VERSION,
            len(inventory.trainval_blobs),
        )
    )


def prepare_data(
    workspace: Path,
    dataroot: Path,
    archives: Optional[Sequence[str]] = None,
    profile: str = "mini",
    force: bool = False,
) -> List[Path]:
    dataroot = dataroot.resolve()
    dataroot.mkdir(parents=True, exist_ok=True)
    resolved = resolve_archives(workspace.resolve(), archives, profile=profile)
    if not resolved:
        raise FileNotFoundError("No archives found. Pass --archive or place nuScenes archives in the workspace root.")
    _validate_profile_requirements(workspace.resolve(), resolved, profile=profile)

    extracted = []
    for archive_path in resolved:
        if not archive_path.exists():
            raise FileNotFoundError("Archive not found: {0}".format(archive_path))

        if archive_path.suffix == ".zip":
            _extract_zip(archive_path, dataroot)
        else:
            _extract_tar(archive_path, dataroot)
        extracted.append(archive_path)

    normalize_map_layout(dataroot)

    required_versions = ["v1.0-mini"] if profile == "mini" else [TRAINVAL_TABLES_VERSION] if profile == "trainval" else []
    for version in required_versions:
        version_dir = dataroot / version
        if not version_dir.exists() and not force:
            raise FileNotFoundError(
                "Data extraction completed but {0} is missing. Check that the selected archives contain nuScenes tables.".format(
                    version_dir
                )
            )
    return extracted
