import builtins
import sys
import traceback

original_open = builtins.open


def proxy_open(file, *args, **kwargs):
    try:
        fname = str(file)
        if "MagicMock" in fname:
            print(f"\n[LEAK DETECTED!] Attempting to open: {fname}", file=sys.stderr)
            traceback.print_stack(file=sys.stderr)
    except Exception:
        pass
    return original_open(file, *args, **kwargs)


builtins.open = proxy_open
