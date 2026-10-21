#!/usr/bin/env bash
# ggml-loader installer -- Termux (Android) and Linux.
#
# Configures (if needed), builds, and installs the ggml-loader binary:
#   Termux : $PREFIX/bin        (no sudo, no root needed)
#   Linux  : /usr/local/bin     (sudo when not root), or ~/.local/bin with --user
#
# Usage:
#   ./install_ggml_loader.sh [--user] [--no-vulkan] [--prefix PATH] [-j N] [--build-only]
#
#   --user       install to ~/.local/bin (Linux only; Termux already installs user-local)
#   --no-vulkan  CPU-only build (default: Vulkan ON -- the project's raison d'etre)
#   --prefix P   install to P/bin instead of the detected default
#   -j N         parallel build jobs (default: all cores)
#   --build-only configure+build but skip the install step

set -euo pipefail

YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${YELLOW}[*]${NC} $*"; }
ok()   { echo -e "${GREEN}[+]${NC} $*"; }
die()  { echo -e "${RED}[!]${NC} $*" >&2; exit 1; }

USER_INSTALL=0
VULKAN=ON
JOBS="$( (nproc 2>/dev/null) || (getconf _NPROCESSORS_ONLN 2>/dev/null) || echo 8 )"
PREFIX_OVERRIDE=""
BUILD_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)       USER_INSTALL=1 ;;
        --no-vulkan)  VULKAN=OFF ;;
        --prefix)     shift; [[ $# -gt 0 ]] || die "--prefix needs a path"; PREFIX_OVERRIDE="$1" ;;
        -j)           shift; [[ $# -gt 0 ]] || die "-j needs a number"; JOBS="$1" ;;
        --build-only) BUILD_ONLY=1 ;;
        -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
        *)            die "unknown option '$1' (see --help)" ;;
    esac
    shift
done

# --- where are we installing? ------------------------------------------------
IS_TERMUX=0
[[ -n "${PREFIX:-}" && "$PREFIX" == *"com.termux"* ]] && IS_TERMUX=1

if [[ -n "$PREFIX_OVERRIDE" ]]; then
    DEST_DIR="$PREFIX_OVERRIDE/bin"
elif [[ $IS_TERMUX -eq 1 ]]; then
    DEST_DIR="$PREFIX/bin"
elif [[ $USER_INSTALL -eq 1 ]]; then
    DEST_DIR="$HOME/.local/bin"
else
    DEST_DIR="/usr/local/bin"
fi

# --- build -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command -v cmake >/dev/null 2>&1 || die "cmake not found (Termux: pkg install cmake; Debian/Ubuntu: sudo apt install cmake)"

if [[ ! -f build/CMakeCache.txt ]]; then
    log "Configuring (Vulkan=$VULKAN, static libs for a relocatable binary)..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release \
                  -DBUILD_SHARED_LIBS=OFF \
                  -DGGML_LOADER_VULKAN=$VULKAN
else
    ok "Existing build/ cache found -- reusing its configuration."
fi

log "Building with -j$JOBS ..."
cmake --build build -j"$JOBS"
[[ -x build/ggml-loader ]] || die "build finished but build/ggml-loader is missing"
ok "Build succeeded."

[[ $BUILD_ONLY -eq 1 ]] && { ok "--build-only: skipping install."; exit 0; }

# --- install -----------------------------------------------------------------
mkdir -p "$DEST_DIR" 2>/dev/null || true
if [[ ! -d "$DEST_DIR" ]]; then
    die "cannot create $DEST_DIR (pick --user or --prefix)"
fi

SUDO=""
if [[ ! -w "$DEST_DIR" ]]; then
    command -v sudo >/dev/null 2>&1 || die "$DEST_DIR is not writable and sudo is unavailable (use --user or --prefix)"
    SUDO="sudo"
fi

$SUDO cp build/ggml-loader "$DEST_DIR/ggml-loader"
$SUDO chmod +x "$DEST_DIR/ggml-loader"
ok "Installed: $DEST_DIR/ggml-loader"

if command -v ggml-loader >/dev/null 2>&1; then
    ok "ggml-loader is in PATH: $(command -v ggml-loader)"
else
    log "Note: $DEST_DIR is not in your PATH -- add it, e.g.:"
    echo "    export PATH=\"$DEST_DIR:\$PATH\""
fi
