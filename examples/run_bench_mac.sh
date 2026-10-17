#!/bin/sh
# examples/run_bench_mac.sh —— 在 macOS（或 Linux）上编译并运行 bench_accel，同时记下运行前后的机器负载
#
#   sh examples/run_bench_mac.sh                      # 结果写到 build/bench/bench_mac.json 和 bench_mac.log
#   sh examples/run_bench_mac.sh 输出目录              # 换一个输出目录
#   sh examples/run_bench_mac.sh 输出目录 --quick      # 第二个参数起原样传给 bench_accel
#
# 可以用环境变量改的东西：
#   CXX      编译器（默认 c++）
#   THREADS  多线程各项要测的线程数（默认 "1,2,4,6,8"：M1 Pro 是 6 个性能核 + 2 个能效核）
#   MACHINE  写进 JSON 的处理器名称（默认取系统报告的型号）
#   TAG      输出文件的主名（默认 bench_mac）
#
# 日志里除了 bench_accel 的进度和每一轮的原始读数，前后各有一段“当时的负载”：
# 平均负载、整机 CPU 占用、占 CPU 最多的几个进程。别的程序在抢核心时，多线程的读数会偏慢，要对照着看。
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/build/bench"}
if [ $# -gt 0 ]; then shift; fi
CXX=${CXX:-c++}
THREADS=${THREADS:-"1,2,4,6,8"}
TAG=${TAG:-bench_mac}
MACHINE=${MACHINE:-$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)}
mkdir -p "$OUT" "$ROOT/build"
EXE="$ROOT/build/bench_accel"

PTHREAD=""
if [ "$(uname -s)" != "Darwin" ]; then PTHREAD="-pthread"; fi
# shellcheck disable=SC2086
"$CXX" -std=c++17 -O2 -DBENCH_OPT=-O2 $PTHREAD -I "$ROOT/include" -I "$ROOT/examples" \
    "$ROOT/examples/bench_accel.cpp" -o "$EXE"

# 当时的负载：平均负载、整机 CPU 占用（macOS 的 top 第二次采样才准）、占 CPU 最多的 6 个进程
load() {
    echo "==== load $1: $(date '+%Y-%m-%dT%H:%M:%S')"
    uptime
    if [ "$(uname -s)" = "Darwin" ]; then
        top -l 2 -n 0 -s 1 | grep 'CPU usage' | tail -1
    fi
    ps -A -o %cpu= -o comm= | sort -nr | head -6 | sed -E 's|^ *([0-9.]+) +(.*/)?([^/]+)$|  \1%  \3|'
}

LOG="$OUT/$TAG.log"
load "before the run" > "$LOG" 2>&1
RC=0
# 先写到临时文件，成功了才换成正式的文件名：运行失败时不会把上一次的结果清空
"$EXE" --threads "$THREADS" --machine "$MACHINE" "$@" > "$OUT/$TAG.json.tmp" 2>> "$LOG" || RC=$?
load "after the run" >> "$LOG" 2>&1
echo "exit code $RC" >> "$LOG"
if [ "$RC" -eq 0 ]; then
    mv "$OUT/$TAG.json.tmp" "$OUT/$TAG.json"
    echo "wrote $OUT/$TAG.json and $LOG" >&2
else
    rm -f "$OUT/$TAG.json.tmp"
    echo "bench_accel failed (exit code $RC), $OUT/$TAG.json left untouched; see $LOG" >&2
fi
exit $RC
