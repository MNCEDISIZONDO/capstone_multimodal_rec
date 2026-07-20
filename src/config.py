from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(PROJECT_ROOT / rel)
    return cfg


if __name__ == "__main__":
    c = load_config()
    print("Config loaded.")
    print("Ghost fraction :", c["data"]["ghost_fraction"])
    print("Split seed     :", c["seeds"]["split_construction"])
    print("SBERT dim      :", c["features"]["sbert_dim"])
    print("Features path  :", c["paths"]["features"])