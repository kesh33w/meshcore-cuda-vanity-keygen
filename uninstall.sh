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
ICON_HOME="$PREFIX/share/icons/hicolor"

rm -f -- "$BIN_PATH" "$DESKTOP_PATH" \
    "$ICON_HOME/scalable/apps/meshcore-vanity-keygen.svg" \
    "$ICON_HOME/256x256/apps/meshcore-vanity-keygen.png"
if [[ -d "$APP_HOME" ]]; then
    find "$APP_HOME" -type f -delete
    find "$APP_HOME" -depth -type d -empty -delete
fi
echo "Application removed. Generated private keys were preserved in: $DATA_HOME/results"
