#!/usr/bin/env bash
# Reproduces every numerical claim in "The Embedded-Observer No-Go Theorem"
# (Sections 3.1, 3.2, 3.4, 4.2-4.3), in order. Run from the repository root.
# Total runtime: a few seconds.
set -euo pipefail

OUT=results/log.txt
mkdir -p results
: > "$OUT"

echo "############################################################" | tee -a "$OUT"
echo "# Lemma A (§3.1): purification non-uniqueness"                 | tee -a "$OUT"
echo "############################################################" | tee -a "$OUT"
python3 src/lemma_a_purification.py | tee -a "$OUT"

echo | tee -a "$OUT"
echo "############################################################" | tee -a "$OUT"
echo "# Lemma B (§3.2): no autonomous unitary under active coupling" | tee -a "$OUT"
echo "############################################################" | tee -a "$OUT"
python3 src/lemma_b_autonomy.py | tee -a "$OUT"

echo | tee -a "$OUT"
echo "############################################################" | tee -a "$OUT"
echo "# Stock vs. flow separation (§3.4)"                            | tee -a "$OUT"
echo "############################################################" | tee -a "$OUT"
python3 src/stock_flow_separation.py | tee -a "$OUT"

echo | tee -a "$OUT"
echo "############################################################" | tee -a "$OUT"
echo "# Exact Light Cone theorem (§4.2-4.3)"                         | tee -a "$OUT"
echo "############################################################" | tee -a "$OUT"
python3 src/exact_light_cone.py | tee -a "$OUT"

echo | tee -a "$OUT"
echo "All checks passed. Full log written to $OUT"
