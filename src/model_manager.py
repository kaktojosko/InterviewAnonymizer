import argparse
import os
import sys
import threading
from typing import Callable, Optional

from huggingface_hub import hf_hub_download, snapshot_download

from src.config import (
    ASR_MODEL_PROFILES,
    DEFAULT_ASR_MODEL,
    DIARIZATION_MODELS,
    MODELS_DIR,
    PRELOADED_MODELS_DIR,
)

_download_lock = threading.Lock()


class ModelManager:
    """
    Thread-safe local model manager for ASR and speaker diarization weights.
    Checks preloaded air-gapped directories first (/opt/preloaded_models), then
    the host-mounted cache directory (MODELS_DIR, default ./models), downloading
    and validating file sizes on first use if missing.
    """

    @staticmethod
    def _candidate_roots() -> list[str]:
        roots = []
        if PRELOADED_MODELS_DIR and os.path.isdir(PRELOADED_MODELS_DIR):
            roots.append(PRELOADED_MODELS_DIR)
        if MODELS_DIR and MODELS_DIR not in roots:
            roots.append(MODELS_DIR)
        return roots

    @staticmethod
    def _is_valid_file(path: str, min_bytes: int = 1) -> bool:
        try:
            return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
        except OSError:
            return False

    @staticmethod
    def _is_valid_asr_dir(model_dir: str, profile: dict) -> bool:
        if not os.path.isdir(model_dir):
            return False

        for req in profile.get("required_files", []):
            if not os.path.isfile(os.path.join(model_dir, req)):
                return False

        if profile.get("engine") == "faster_whisper":
            has_vocab = (
                os.path.isfile(os.path.join(model_dir, "vocabulary.json"))
                or os.path.isfile(os.path.join(model_dir, "vocabulary.txt"))
            )
            if not has_vocab:
                return False

        total_bytes = 0
        for entry in os.listdir(model_dir):
            full = os.path.join(model_dir, entry)
            if os.path.isfile(full):
                try:
                    total_bytes += os.path.getsize(full)
                except OSError:
                    pass

        return total_bytes >= profile.get("min_total_bytes", 1)

    @classmethod
    def ensure_diarization_models(
        cls, status_cb: Optional[Callable[[str], None]] = None
    ) -> tuple[str, str]:
        """
        Ensures segmentation and speaker embedding ONNX models exist locally.
        Returns (segmentation_onnx_path, embedding_onnx_path).
        """
        seg_spec = DIARIZATION_MODELS["segmentation"]
        emb_spec = DIARIZATION_MODELS["embedding"]

        seg_rel = os.path.join("diarization", "segmentation", seg_spec["filename"])
        emb_rel = os.path.join("diarization", "embedding", emb_spec["filename"])

        with _download_lock:
            seg_path = None
            emb_path = None

            for root in cls._candidate_roots():
                cand_seg = os.path.join(root, seg_rel)
                if seg_path is None and cls._is_valid_file(cand_seg, seg_spec["min_bytes"]):
                    seg_path = cand_seg

                cand_emb = os.path.join(root, emb_rel)
                if emb_path is None and cls._is_valid_file(cand_emb, emb_spec["min_bytes"]):
                    emb_path = cand_emb

            if seg_path is None:
                target_seg_dir = os.path.join(MODELS_DIR, "diarization", "segmentation")
                os.makedirs(target_seg_dir, exist_ok=True)
                if status_cb:
                    status_cb("Загрузка локальной модели сегментации спикеров (~6 МБ, однократно)...")
                downloaded_seg = hf_hub_download(
                    repo_id=seg_spec["repo_id"],
                    filename=seg_spec["filename"],
                    local_dir=target_seg_dir,
                )
                if not cls._is_valid_file(downloaded_seg, seg_spec["min_bytes"]):
                    raise RuntimeError(
                        f"Downloaded segmentation model is invalid or truncated: {downloaded_seg}"
                    )
                seg_path = downloaded_seg

            if emb_path is None:
                target_emb_dir = os.path.join(MODELS_DIR, "diarization", "embedding")
                os.makedirs(target_emb_dir, exist_ok=True)
                if status_cb:
                    status_cb("Загрузка локальной модели голосовых эмбеддингов (~28 МБ, однократно)...")
                downloaded_emb = hf_hub_download(
                    repo_id=emb_spec["repo_id"],
                    filename=emb_spec["filename"],
                    local_dir=target_emb_dir,
                )
                if not cls._is_valid_file(downloaded_emb, emb_spec["min_bytes"]):
                    raise RuntimeError(
                        f"Downloaded speaker embedding model is invalid or truncated: {downloaded_emb}"
                    )
                emb_path = downloaded_emb

            return seg_path, emb_path

    @classmethod
    def ensure_asr_model(
        cls,
        profile_id: str = DEFAULT_ASR_MODEL,
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Ensures the requested ASR model profile exists locally and is valid.
        Returns the local directory path containing the model weights.
        """
        if profile_id not in ASR_MODEL_PROFILES:
            raise ValueError(f"Unknown ASR model profile: '{profile_id}'")

        profile = ASR_MODEL_PROFILES[profile_id]
        asr_rel = os.path.join("asr", profile_id)

        with _download_lock:
            for root in cls._candidate_roots():
                cand_dir = os.path.join(root, asr_rel)
                if cls._is_valid_asr_dir(cand_dir, profile):
                    return cand_dir

            target_dir = os.path.join(MODELS_DIR, asr_rel)
            os.makedirs(target_dir, exist_ok=True)

            if status_cb:
                status_cb(
                    f"Первичная загрузка локальной модели распознавания речи "
                    f"({profile['short_name']})... В следующие запуски модель берётся из кэша."
                )

            snapshot_download(
                repo_id=profile["repo_id"],
                local_dir=target_dir,
                allow_patterns=profile.get("allow_patterns"),
            )

            if not cls._is_valid_asr_dir(target_dir, profile):
                raise RuntimeError(
                    f"Downloaded ASR model '{profile_id}' failed integrity check in {target_dir}"
                )

            return target_dir


def _cli():
    parser = argparse.ArgumentParser(
        description="Preload local ASR and speaker diarization models into MODELS_DIR."
    )
    parser.add_argument(
        "--preload",
        required=True,
        help="Comma-separated ASR profile IDs (turbo,large-v3,parakeet,small), 'all', or 'diarization'.",
    )
    args = parser.parse_args()

    raw_items = [item.strip().lower() for item in args.preload.split(",") if item.strip()]
    if not raw_items:
        print("Nothing to preload.")
        return

    print(f"[ModelManager] Target directory: {MODELS_DIR}")
    ModelManager.ensure_diarization_models(status_cb=print)

    if "all" in raw_items:
        profiles_to_load = list(ASR_MODEL_PROFILES.keys())
    else:
        profiles_to_load = [p for p in raw_items if p != "diarization"]

    for pid in profiles_to_load:
        if pid not in ASR_MODEL_PROFILES:
            print(f"Error: unknown ASR profile '{pid}'. Valid: {', '.join(ASR_MODEL_PROFILES)}", file=sys.stderr)
            sys.exit(1)
        path = ModelManager.ensure_asr_model(pid, status_cb=print)
        print(f"[ModelManager] Ready: {pid} -> {path}")


if __name__ == "__main__":
    _cli()
