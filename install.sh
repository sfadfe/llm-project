#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
UV=${UV:-$HOME/.local/bin/uv}

"$UV" sync --frozen --python /usr/bin/python3.12

echo 'import os; os.environ.setdefault("HIP_VISIBLE_DEVICES", "0,1")' \
  > .venv/lib/python3.12/site-packages/_hip_visible_devices.pth
