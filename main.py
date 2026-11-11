import argparse
import os
import sys

from src.config import (
    ASR_MODEL_PROFILES,
    COMPUTE_DEVICE_OPTIONS,
    DEFAULT_ASR_MODEL,
    DEFAULT_COMPUTE_DEVICE,
    MAX_ALLOWED_WORKERS,
    MAX_WORKERS,
    SPEAKER_COUNT_OPTIONS,
    TRANSCRIPTION_ENABLED_DEFAULT,
)
from src.io_manager import IOManager
from src.orchestrator import Orchestrator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interview Anonymizer v4.1.1 (Blur faces, pitch shift audio, and local role-split transcription)"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input video (.mp4)")
    parser.add_argument("-o", "--output", required=True, help="Path to output anonymized video (.mp4)")
    parser.add_argument(
        "-t",
        "--transcript-output",
        default=None,
        help="Path to output transcript .txt file (default: <output_stem>_transcript.txt)",
    )
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Disable automatic local speech transcription (enabled by default)",
    )
    parser.add_argument(
        "--asr-model",
        choices=list(ASR_MODEL_PROFILES.keys()),
        default=DEFAULT_ASR_MODEL,
        help=f"Local ASR model profile (default: {DEFAULT_ASR_MODEL})",
    )
    parser.add_argument(
        "--speakers",
        choices=SPEAKER_COUNT_OPTIONS,
        default="auto",
        help="Speaker count hint for diarization (default: auto)",
    )
    parser.add_argument(
        "--device",
        choices=COMPUTE_DEVICE_OPTIONS,
        default=DEFAULT_COMPUTE_DEVICE,
        help=f"Compute device for transcription: cpu or cuda (default: {DEFAULT_COMPUTE_DEVICE})",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel CPU workers, 1..{MAX_ALLOWED_WORKERS} (default: {MAX_WORKERS})",
    )
    return parser


def process_video(
    input_path: str,
    output_path: str,
    transcribe: bool = TRANSCRIPTION_ENABLED_DEFAULT,
    transcript_path: str | None = None,
    asr_model: str = DEFAULT_ASR_MODEL,
    speaker_count: str = "auto",
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    workers: int = MAX_WORKERS,
) -> dict:
    io_manager = IOManager()
    job_dir = io_manager.create_job_dir()
    worker_count = max(1, min(MAX_ALLOWED_WORKERS, int(workers)))

    try:
        result = Orchestrator.process_video_job(
            input_path=input_path,
            output_path=output_path,
            job_dir=job_dir,
            worker_count=worker_count,
            transcribe=transcribe,
            transcript_path=transcript_path,
            asr_model=asr_model,
            speaker_count=speaker_count,
            compute_device=compute_device,
        )
        if result.get("transcript_ready") and result.get("transcript_path"):
            print(f"Transcript saved to: {result['transcript_path']}")
        return result
    finally:
        print("Cleaning up temporary files...")
        io_manager.cleanup_job_dir(job_dir)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    if args.workers < 1 or args.workers > MAX_ALLOWED_WORKERS:
        print(f"Error: --workers must be between 1 and {MAX_ALLOWED_WORKERS}.")
        sys.exit(1)

    transcribe_enabled = not args.no_transcribe
    process_video(
        input_path=args.input,
        output_path=args.output,
        transcribe=transcribe_enabled,
        transcript_path=args.transcript_output,
        asr_model=args.asr_model,
        speaker_count=args.speakers,
        compute_device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

