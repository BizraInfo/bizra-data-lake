path = r"C:\BIZRA-DATA-LAKE\bizra-omega\bizra-resourcepool\src\lib.rs"
terms = ["zakat", "gini", "harberger", "distribute", "split", "fn "]
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        lower = line.lower()
        if any(t in lower for t in ["zakat", "gini", "harberger"]):
            print(f"{i}: {line.rstrip()}")
        elif "fn " in line and ("pub" in line or "async" in line):
            print(f"{i}: {line.rstrip()}")
