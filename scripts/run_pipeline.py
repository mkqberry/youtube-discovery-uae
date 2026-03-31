from pathlib import Path
import importlib
import sys


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    for mod_name in ("main", "app", "run"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "main"):
                mod.main()
                return
        except Exception:
            continue
    print("No runnable entrypoint found in src/.")


if __name__ == "__main__":
    main()
