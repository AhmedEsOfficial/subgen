"""CLI entry point for subgen."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from subgen import __version__
from subgen.config import AppConfig, load_config
from subgen.orchestrator import Pipeline

app = typer.Typer(
    name="subgen",
    help="Generate subtitles from video files — transcribe, translate, emit SRT.",
    add_completion=False,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.command()
def generate(
    input_file: Path = typer.Argument(..., help="Input video file path", exists=True),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output SRT file path"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file path"),
    source_lang: Optional[str] = typer.Option(None, "--source-lang", "-s", help="Source language code (e.g. 'ar')"),
    target_lang: Optional[str] = typer.Option(None, "--target-lang", "-t", help="Target language code (e.g. 'en')"),
    asr: Optional[str] = typer.Option(None, "--asr", help="ASR engine: whisperx | whisper_hf | faster_whisper"),
    asr_model: Optional[str] = typer.Option(None, "--asr-model", help="ASR model name/path"),
    translator: Optional[str] = typer.Option(None, "--translator", help="Translation engine: llama_cpp | nllb | opus | local_llm | none"),
    translator_model: Optional[str] = typer.Option(None, "--translator-model", help="Translation model name/path"),
    llama_server_url: Optional[str] = typer.Option(None, "--llama-url", help="URL of running llama-server (e.g. http://127.0.0.1:8081)"),
    initial_prompt: Optional[str] = typer.Option(None, "--initial-prompt", help="Initial prompt for ASR (improves dialect accuracy)"),
    denoise: bool = typer.Option(False, "--denoise", help="Enable audio denoising via demucs"),
    device: Optional[str] = typer.Option(None, "--device", help="Device: auto | cuda | cpu"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
) -> None:
    """Generate subtitles from a video file."""
    _setup_logging(verbose)
    logger = logging.getLogger("subgen.cli")

    # Load config
    if config and config.exists():
        logger.info("Loading config from %s", config)
        cfg = load_config(config)
    else:
        cfg = AppConfig()

    # Apply CLI overrides
    if source_lang:
        cfg.language_detection.engine = "fixed"
        cfg.language_detection.fixed_language = source_lang
    if target_lang:
        cfg.translation.target_lang = target_lang
    if asr:
        cfg.asr.engine = asr  # type: ignore[assignment]
    if asr_model:
        cfg.asr.model = asr_model
    if translator:
        cfg.translation.engine = translator  # type: ignore[assignment]
    if translator_model:
        cfg.translation.model = translator_model
    if llama_server_url:
        cfg.translation.llama_server_url = llama_server_url
    if initial_prompt:
        cfg.asr.initial_prompt = initial_prompt
    if denoise:
        cfg.audio.denoise = True
    if device:
        cfg.asr.device = device  # type: ignore[assignment]

    # Determine output path
    if output is None:
        output = input_file.with_suffix(".srt")

    # Run pipeline
    logger.info("Starting subtitle generation: %s", input_file)
    pipeline = Pipeline(cfg)
    result = pipeline.run(
        video_path=input_file,
        output_path=output,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    typer.echo(f"Subtitles written to: {result}")


@app.command()
def voicenote(
    inputs: list[Path] = typer.Argument(..., help="Voice note files and/or directories of them", exists=True),
    config: Path = typer.Option(Path("config.voicenote.yaml"), "--config", "-c", help="YAML config file path"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-o", help="Output directory (default: next to each input)"),
    force: bool = typer.Option(False, "--force", "-f", help="Redo notes that already have a transcript"),
    translator: Optional[str] = typer.Option(None, "--translator", help="Translation engine override (e.g. none)"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging"),
) -> None:
    """Transcribe and translate voice notes into a bilingual Markdown transcript + SRTs."""
    from subgen.voicenote import VoiceNoteTranscriber, collect_inputs

    _setup_logging(verbose)
    if not config.exists():
        raise typer.BadParameter(f"config file not found: {config}", param_hint="--config")
    cfg = load_config(config)
    if translator:
        cfg.translation.engine = translator  # type: ignore[assignment]

    files = collect_inputs(inputs)
    if not files:
        typer.echo("No audio files found.")
        raise typer.Exit(1)

    notes, failed = VoiceNoteTranscriber(cfg).run(files, out_dir=out_dir, force=force)
    for note in notes:
        typer.echo(f"Transcript written to: {note.markdown_path}")
    if not notes and not failed:
        typer.echo("Nothing to do — every note already has a transcript (use --force to redo).")
    for path in failed:
        typer.echo(f"FAILED: {path}", err=True)
    if failed:
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show subgen version."""
    typer.echo(f"subgen v{__version__}")


if __name__ == "__main__":
    app()
