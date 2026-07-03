#!/bin/bash
# Board-in-the-loop regression test: CMM teardown on Init-failure paths.
#
# Forces a deterministic mid-load abort (AXLLM_TEST_ABORT_AFTER_LAYER) and asserts the
# partially-loaded model releases all its CMM cleanly -- guarding against the teardown-
# order leak fixed in commit "fix(mem-guard): free model before engine/SYS teardown".
#
# Run ON an AX650 board (needs /proc/ax_proc/mem_cmm_info + /soc/lib):
#   AXLLM=/root/axllm MODEL=/root/models/qwen3-0.6b-ax650 ./test_teardown_cmm_leak.sh
# Exit code 0 = PASS, non-zero = FAIL (wireable into board CI).
set -uo pipefail
AXLLM="${AXLLM:-/root/axllm}"
MODEL="${MODEL:?set MODEL=<model dir>}"
PORT="${PORT:-18099}"
ABORT_AFTER="${ABORT_AFTER:-1}"          # abort after N layers loaded (1 = customer scenario)
SYSLOG_DIR="${SYSLOG_DIR:-/opt/data/AXSyslog/syslog}"
MARGIN_MB="${MARGIN_MB:-16}"

read_cmm_kb() { grep -oE 'remain=[0-9]+KB' /proc/ax_proc/mem_cmm_info | head -1 | grep -oE '[0-9]+'; }
LOG=$(ls -t "$SYSLOG_DIR"/* 2>/dev/null | head -1)
fail=0

base=$(read_cmm_kb)
n0=$(grep -acE "Free failed|not ready|not inited" "$LOG" 2>/dev/null || echo 0)

AXLLM_TEST_ABORT_AFTER_LAYER="$ABORT_AFTER" LD_LIBRARY_PATH=/soc/lib \
  timeout 90 "$AXLLM" serve "$MODEL" --port "$PORT" >/tmp/teardown_test.log 2>&1
sync; sleep 4
after=$(read_cmm_kb)
n1=$(grep -acE "Free failed|not ready|not inited" "$LOG" 2>/dev/null || echo 0)
delta_mb=$(( (base - after) / 1024 ))

# 1) the forced mid-load abort must actually trigger
grep -q "LLM.Init failed" /tmp/teardown_test.log || { echo "FAIL: forced abort did not trigger"; fail=1; }
# 2) the CMM sentry must not report a leak
if grep -q "suspected leak" /tmp/teardown_test.log; then echo "FAIL: cmm-sentry reported a leak:"; grep -a cmm-sentry /tmp/teardown_test.log; fail=1; fi
# 3) no NEW driver free-failures in the board syslog
if [ "${n1:-0}" -gt "${n0:-0}" ]; then echo "FAIL: $((n1-n0)) new 'Free failed/not ready/not inited' driver errors in syslog"; fail=1; fi
# 4) CMM must return to baseline
if [ "$delta_mb" -gt "$MARGIN_MB" ]; then echo "FAIL: CMM not reclaimed: ${delta_mb} MB down (base=${base}KB after=${after}KB)"; fail=1; fi

if [ "$fail" -eq 0 ]; then
  echo "PASS: teardown clean (abort after ${ABORT_AFTER} layer(s); CMM delta ${delta_mb}MB; no new free-failures; sentry balanced)"
else
  echo "RESULT: FAIL"
fi
exit $fail
