#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HOME}/.local"

while (($#)); do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || { echo "--prefix requires a directory" >&2; exit 2; }
            PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--prefix DIR]"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

APP_HOME="$PREFIX/lib/meshcore-vanity-keygen"
BIN_PATH="$PREFIX/bin/meshcore-vanity-keygen"
DESKTOP_PATH="${XDG_DATA_HOME:-$HOME/.local/share}/applications/meshcore-vanity-keygen.desktop"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/meshcore-vanity-keygen"

rm -f -- "$BIN_PATH" "$DESKTOP_PATH"
if [[ -d "$APP_HOME" ]]; then
    find "$APP_HOME" -mindepth 1 -maxdepth 1 -type f -delete
    rmdir "$APP_HOME"
fi
echo "Application removed. Generated private keys were preserved in: $DATA_HOME/results"
