import time, importlib, traceback


def timed(name, fn):
    t = time.time()
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print(f"[run_all] {name} FAILED")
    print(f"[run_all] {name}: {time.time()-t:.0f}s", flush=True)


def main():
    t0 = time.time()
    try:
        uci = importlib.import_module("tasks.uci_classification")
        if hasattr(uci, "main"):
            timed("uci_classification", uci.main)
    except Exception:
        traceback.print_exc(); print("[run_all] uci_classification skipped")
    for m in ["segment_classification", "temporal", "temporal_odometry", "regression", "clustering"]:
        mod = importlib.import_module(f"tasks.{m}")
        timed(m, mod.main)
    print(f"[run_all] total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
