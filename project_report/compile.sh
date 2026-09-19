#!/usr/bin/env bash
# Script to compile the LaTeX project report
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[1/2] Compiling main.tex with pdflatex..."
pdflatex -interaction=nonstopmode main.tex > /dev/null

echo "[2/2] Running second pass for references and cross-links..."
pdflatex -interaction=nonstopmode main.tex > /dev/null

echo "==> Compilation successful: main.pdf generated in $DIR"
