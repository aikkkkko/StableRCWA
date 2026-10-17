#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# 系统 Python 在 Apple 芯片上是原生 arm64（NumPy 走 Accelerate），优先于可能经 Rosetta 转译的 x86 Anaconda。
# 未装命令行工具时 /usr/bin/python3 只是会弹安装窗的占位程序，此时跳过。
rcwa_system_python=""
/usr/bin/xcode-select -p >/dev/null 2>&1 && rcwa_system_python=/usr/bin/python3
rcwa_candidates=("${RCWA_PYTHON:-}" "$PWD/.venv/bin/python" "$rcwa_system_python" "$HOME/opt/anaconda3/bin/python" "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" "$(command -v python3)")
for rcwa_python in "${rcwa_candidates[@]}"; do
  if [[ -n "$rcwa_python" && -x "$rcwa_python" ]] && "$rcwa_python" -c 'import numpy, scipy, matplotlib' >/dev/null 2>&1; then
    exec "$rcwa_python" -u gui.py
  fi
done
echo '未找到包含 NumPy、SciPy、Matplotlib 的 Python。'
echo '请在此文件夹运行：python3 -m pip install -r requirements.txt'
echo '然后运行：python3 gui.py'
read -k 1 '?按任意键关闭…'
