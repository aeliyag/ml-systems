#!/usr/bin/env bash
# Serve the local HTML textbook with Read the Docs theme assets.
# Usage: ./docs/book/serve.sh
# Then open http://127.0.0.1:8765/ (Safari is more reliable than Cursor's preview)

set -euo pipefail

PORT="${PORT:-8765}"
BOOK_SRC="$(cd "$(dirname "$0")" && pwd)"
VENV="${TMPDIR:-/tmp}/ml-book-venv"
PREVIEW="${TMPDIR:-/tmp}/ml-book-preview"

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q sphinx sphinx-rtd-theme
THEME="$("$VENV/bin/python" -c "import sphinx_rtd_theme, pathlib; print(pathlib.Path(sphinx_rtd_theme.__file__).parent / 'static')")"

rm -rf "$PREVIEW"
mkdir -p "$PREVIEW"
rsync -a --exclude 'serve.sh' --exclude '.buildinfo' "$BOOK_SRC/" "$PREVIEW/"
rsync -a "$THEME/" "$PREVIEW/static/"

if [ -f "$PREVIEW/static/jquery-3.5.1.js" ] && [ ! -f "$PREVIEW/static/jquery.js" ]; then
  cp "$PREVIEW/static/jquery-3.5.1.js" "$PREVIEW/static/jquery.js"
fi

mkdir -p "$PREVIEW/static/js" "$PREVIEW/images"
[ -f "$PREVIEW/static/nbsphinx-code-cells.css" ] || echo '/* nbsphinx code cells */' > "$PREVIEW/static/nbsphinx-code-cells.css"
[ -f "$PREVIEW/static/js/html5shiv.min.js" ] || echo '/* html5shiv placeholder */' > "$PREVIEW/static/js/html5shiv.min.js"
[ -f "$PREVIEW/static/bridge.ico" ] || : > "$PREVIEW/static/bridge.ico"

echo "Serving textbook at http://127.0.0.1:${PORT}/"
echo "Open it in Safari. Press Ctrl-C to stop."
if command -v open >/dev/null 2>&1; then
  (sleep 0.4 && open -a Safari "http://127.0.0.1:${PORT}/") &
fi
exec python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$PREVIEW"
