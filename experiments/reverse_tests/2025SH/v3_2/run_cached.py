#!/usr/bin/env python3
"""
Launcher：根据当前目录推导 exp_id 与模型，调用 reverse_tests 下通用入口 train_fluProfiler.main()。
可原样复制到任意 reverse_tests/<季节>/<版本> 目录使用。
"""
import os
import sys
from pathlib import Path

_script = Path(__file__).resolve()
_reverse_tests = _script.parent.parent.parent
sys.path.insert(0, str(_reverse_tests))
from experiment_tools import find_repo_root, default_exp_id

_repo = find_repo_root(_script)
_exp_id = os.environ.get("FLUPROFILER_EXP_ID") or default_exp_id(_script, _repo)
_model = os.environ.get("FLUPROFILER_MODEL") or _exp_id.strip().split("/")[-1]

os.environ.setdefault("FLUPROFILER_EXP_ID", _exp_id)
os.environ.setdefault("FLUPROFILER_MODEL", _model)

from train_fluProfiler import main

main()
