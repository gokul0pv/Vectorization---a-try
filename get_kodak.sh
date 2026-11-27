#!/bin/sh
# Downloads the 24 Kodak test photos (768x512) used by bench_kodak.py.
mkdir -p kodak
for i in $(seq -w 1 24); do
  curl -sS -m 90 -o kodak/kodim$i.png https://r0k.us/graphics/kodak/kodak/kodim$i.png
done
