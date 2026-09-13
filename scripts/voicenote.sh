#!/usr/bin/env bash
# Transcribe (Egyptian Arabic) and translate (English) voice notes.
#
#   scripts/voicenote.sh                         # every new note in voicenotes/
#   scripts/voicenote.sh voicenotes/audio.ogg    # specific files or directories
#   scripts/voicenote.sh --force voicenotes/x.ogg
#
# Writes <name>.md (bilingual transcript), <name>.ar.srt and <name>.en.srt next to
# each note. Notes that already have a .md are skipped unless --force is given.
# Other `subgen voicenote` options pass through: scripts/voicenote.sh --help
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$repo/.venv/bin/python"
[[ -x "$python" ]] || { echo "error: virtualenv not found at $repo/.venv" >&2; exit 1; }

# Scan arguments: note whether a config or any input path was given.
has_config=false
has_input=false
expects_value=false
for arg in "$@"; do
  if $expects_value; then expects_value=false; continue; fi
  case "$arg" in
    -c|--config) has_config=true; expects_value=true ;;
    --config=*) has_config=true ;;
    -o|--out-dir|--translator) expects_value=true ;;
    -*) ;;
    *) has_input=true ;;
  esac
done

extra=()
$has_config || extra+=(--config "$repo/config.voicenote.yaml")
$has_input || extra+=("$repo/voicenotes")

# Run from the caller's directory so relative paths mean what the caller typed.
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" exec "$python" -m subgen.cli voicenote "${extra[@]}" "$@"
