#!/bin/sh
# Installs the pinned official Stockfish release into /usr/local/bin/stockfish, checking its SHA256.
# Usage: install_stockfish.sh <docker TARGETARCH: amd64 | arm64>; PYTHON=<interpreter> when there is no `python` on PATH
set -eu

TAG=sf_19
case "$1" in
  amd64) ASSET=stockfish-linux-x86-64-universal; SHA256=9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f ;;
  arm64) ASSET=stockfish-linux-arm64-universal; SHA256=fe26cfd1d9db4c8af3d21e24d9ff34cacb31c1f940085a7583da11796f2bac01 ;;
  *) echo "no pinned Stockfish build for architecture '$1'" >&2; exit 1 ;;
esac

WORK=$(mktemp -d)
${PYTHON:-python} - "https://github.com/official-stockfish/Stockfish/releases/download/$TAG/$ASSET.tar.gz" "$WORK/sf.tar.gz" <<'EOF'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
EOF
echo "$SHA256  $WORK/sf.tar.gz" | sha256sum -c -
tar -xzf "$WORK/sf.tar.gz" -C "$WORK" "stockfish/$ASSET"
install -m 0755 "$WORK/stockfish/$ASSET" /usr/local/bin/stockfish
rm -rf "$WORK"
