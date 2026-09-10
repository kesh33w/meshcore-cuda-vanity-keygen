#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${HOME}/.local"
INSTALL_PACKAGES=1
BUILD=1
DESKTOP=1

usage() {
    echo "Usage: $0 [--prefix DIR] [--skip-packages] [--skip-build] [--no-desktop]"
}

while (($#)); do
    case "$1" in
        --prefix)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            PREFIX="$2"
            shift 2
            ;;
        --skip-packages) INSTALL_PACKAGES=0; shift ;;
        --skip-build) BUILD=0; shift ;;
        --no-desktop) DESKTOP=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ((INSTALL_PACKAGES)); then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Automatic package installation currently supports apt-based Linux systems." >&2
        echo "Install Python 3, Tk, libsodium, GNU Make/C++, and the CUDA toolkit, then use --skip-packages." >&2
        exit 2
    fi
    packages=(build-essential libsodium23 python3 python3-tk)
    if ! command -v nvcc >/dev/null 2>&1; then
        packages+=(nvidia-cuda-toolkit)
    fi
    echo "Installing system dependencies (sudo may ask for your password)..."
    sudo apt-get update
    sudo apt-get install -y "${packages[@]}"
fi

if ((BUILD)); then
    make -C "$PROJECT_DIR"
fi

[[ -x "$PROJECT_DIR/meshcore_cuda_vanity" ]] || {
    echo "CUDA engine is missing. Run make, or omit --skip-build." >&2
    exit 2
}

APP_HOME="$PREFIX/lib/meshcore-vanity-keygen"
BIN_HOME="$PREFIX/bin"
ICON_HOME="$PREFIX/share/icons/hicolor"
install -d -m 0755 "$APP_HOME/assets" "$BIN_HOME"
install -d -m 0755 "$ICON_HOME/scalable/apps" "$ICON_HOME/256x256/apps"
install -m 0755 "$PROJECT_DIR/meshcore_vanity.py" "$APP_HOME/meshcore_vanity.py"
install -m 0755 "$PROJECT_DIR/meshcore_key_audit.py" "$APP_HOME/meshcore_key_audit.py"
install -m 0755 "$PROJECT_DIR/meshcore_cuda_vanity" "$APP_HOME/meshcore_cuda_vanity"
install -m 0644 "$PROJECT_DIR/VERSION" "$APP_HOME/VERSION"
install -m 0644 "$PROJECT_DIR/LICENSE" "$PROJECT_DIR/THIRD_PARTY_NOTICES.md" "$APP_HOME/"
install -m 0644 "$PROJECT_DIR/assets/meshcore-vanity-keygen.png" "$APP_HOME/assets/"
install -m 0644 "$PROJECT_DIR/assets/meshcore-vanity-keygen.svg" "$ICON_HOME/scalable/apps/meshcore-vanity-keygen.svg"
install -m 0644 "$PROJECT_DIR/assets/meshcore-vanity-keygen.png" "$ICON_HOME/256x256/apps/meshcore-vanity-keygen.png"
install -m 0755 "$PROJECT_DIR/meshcore-vanity-keygen" "$BIN_HOME/meshcore-vanity-keygen"
install -m 0755 "$PROJECT_DIR/meshcore-key-audit" "$BIN_HOME/meshcore-key-audit"

if ((DESKTOP)); then
    APPLICATIONS_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    install -d -m 0755 "$APPLICATIONS_HOME"
    desktop_temp="$(mktemp)"
    trap 'rm -f -- "$desktop_temp"' EXIT
    escaped_exec="${BIN_HOME//&/\\&}/meshcore-vanity-keygen"
    sed "s&@EXEC@&$escaped_exec&" "$PROJECT_DIR/meshcore-vanity-keygen.desktop.in" > "$desktop_temp"
    install -m 0644 "$desktop_temp" "$APPLICATIONS_HOME/meshcore-vanity-keygen.desktop"
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$APPLICATIONS_HOME" >/dev/null 2>&1 || true
    fi
    if command -v gtk-update-icon-cache >/dev/null 2>&1; then
        gtk-update-icon-cache -f -t "$ICON_HOME" >/dev/null 2>&1 || true
    fi
fi

"$BIN_HOME/meshcore-vanity-keygen" --self-test
echo
echo "Installed MeshCore Vanity Key Generator $(<"$PROJECT_DIR/VERSION")"
echo "Command: $BIN_HOME/meshcore-vanity-keygen --gui"
echo "Saved-key audit: $BIN_HOME/meshcore-key-audit PATH"
if [[ ":$PATH:" != *":$BIN_HOME:"* ]]; then
    echo "Add $BIN_HOME to PATH to use 'meshcore-vanity-keygen' directly."
fi
