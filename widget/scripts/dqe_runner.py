#!/usr/bin/env python3
import os, sys, json, importlib.util, contextlib, warnings

def load_module(path, name="DeterministicQueryEngine"):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def main():
    try:
        payload = json.loads(sys.stdin.readline() or "{}")
        query = payload.get("query","")
        engine_path = os.environ.get("DQE_ENGINE_PATH","")
        config_dir  = os.environ.get("DQE_CONFIG_DIR",".")
        master_path = os.environ.get("DQS_MASTER","")

        if not engine_path or not os.path.isfile(engine_path):
            print(json.dumps({})); return 0

        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(sys.stderr):
            os.chdir(config_dir)
            mod = load_module(engine_path)
            Cls = getattr(mod, "DeterministicQueryEngine", None)
            eng = Cls(data_path=master_path) if master_path else Cls()
            res = None
            for name in ("query","run","handle_query","__call__"):
                fn = getattr(eng, name, None) or getattr(mod, name, None)
                if callable(fn): res = fn(query); break

        try:
            from dataclasses import asdict, is_dataclass
            if is_dataclass(res): res = asdict(res)
        except Exception:
            pass

        print(json.dumps(res if res is not None else {}))
        return 0
    except Exception:
        print(json.dumps({})); return 0

if __name__ == "__main__":
    raise SystemExit(main())
