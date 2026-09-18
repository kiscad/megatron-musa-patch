#!/usr/bin/env bash
# Level 2（git push）与 CI：CPU/轻量单元测试。
# 硬件 smoke（*_smoke.py）不在 pytest 默认收集范围内；8 卡分布式、
# 真实训练等重型验证属于专用硬件轮次（见 CONTRIBUTING「测试」一节），
# 不进入本地 pre-push。
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 -m pytest tests -q
