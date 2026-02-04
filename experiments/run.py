from pathlib import Path
import json
import argparse
import sys

sys.path.append('../src/')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", required=True, help="configs/runs/*.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    cfg_path = Path(args.run_config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # 路径统一相对 repo_root 解析（config 里写相对路径更方便）
    def resolve(p):
        if p is None:
            return None
        p = Path(p)
        return str(p if p.is_absolute() else (repo_root / p))

    cfg["train_csv"] = resolve(cfg.get("train_csv"))
    cfg["test_csv"] = resolve(cfg.get("test_csv"))
    cfg["embedding_dir"] = resolve(cfg.get("embedding_dir"))
    cfg["config_json"] = resolve(cfg.get("config_json"))
    cfg["args_pkl"] = resolve(cfg.get("args_pkl"))
    cfg["run_root"] = resolve(cfg.get("run_root", "runs"))

    from fluprofiler.runners.v0_1_csv import run_v0_1_from_csv
    run_dir = run_v0_1_from_csv(**cfg)

    print("Run saved to:", run_dir)

if __name__ == "__main__":
    main()
