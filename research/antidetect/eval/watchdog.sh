#!/bin/zsh
# Restart run_eval.py (with --resume) whenever it stalls >240s, until done.
cd "$(dirname "$0")"
for i in $(seq 1 40); do
  if grep -q "done in" results/run.log 2>/dev/null; then echo "WATCHDOG: complete"; exit 0; fi
  if [ ! -f results/run.log ] || [ $(($(date +%s) - $(stat -f %m results/run.log))) -gt 240 ]; then
    echo "WATCHDOG: stall detected (loop $i), restarting with --resume"
    pkill -f "run_eval.py" 2>/dev/null; sleep 5
    nohup .venv/bin/python -u run_eval.py --resume >> results/run.log 2>&1 &
  fi
  sleep 30
done
echo "WATCHDOG: gave up after 40 loops"
