import sys

sys.path.insert(0, r"C:\BIZRA-DATA-LAKE")
try:
    import uvicorn

    print(f"uvicorn {uvicorn.__version__}")
except ImportError:
    print("uvicorn NOT FOUND")
try:
    import fastapi

    print(f"fastapi {fastapi.__version__}")
except ImportError:
    print("fastapi NOT FOUND")
