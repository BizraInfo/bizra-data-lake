#!/usr/bin/env python3
import argparse, json, os, hashlib, base64, subprocess, tempfile, sys

BASE = os.path.dirname(__file__)
ART = os.path.join(BASE, "artifacts")
KEYS = os.path.join(BASE, "keys")

GENESIS_JSON = os.path.join(ART, "genesis.built.json")
GENESIS_ROOT = os.path.join(ART, "genesis_merkle_root.txt")
PACK_JSON = os.path.join(ART, "pack.manifest.json")
PACK_SHA = os.path.join(ART, "pack.sha256")
POI_JSON = os.path.join(ART, "poi_attestation.json")
REPLAY_DB = os.path.join(ART, "replay.json")

PRIVKEY = os.path.join(KEYS, "ed25519_private.pem")
PUBKEY = os.path.join(KEYS, "ed25519_public.pem")

EVID_METRICS = os.path.join(ART, "evidence.metrics.json")
EVID_LOG = os.path.join(ART, "evidence.log")

CHAIN_ID = "BIZRA-MSSC-DEVNET-0"
WINDOW = "2026-02-03"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ensure_dirs():
    os.makedirs(ART, exist_ok=True)
    os.makedirs(KEYS, exist_ok=True)


def ensure_keys():
    if not os.path.exists(PRIVKEY):
        subprocess.run(["openssl", "genpkey", "-algorithm", "Ed25519", "-out", PRIVKEY], check=True)
    if not os.path.exists(PUBKEY):
        subprocess.run(["openssl", "pkey", "-in", PRIVKEY, "-pubout", "-out", PUBKEY], check=True)


def read_pubkey_pem() -> str:
    with open(PUBKEY, "r", encoding="utf-8") as f:
        return f.read().strip()


def sign_bytes(data: bytes) -> str:
    ensure_keys()
    p = subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", PRIVKEY], input=data, stdout=subprocess.PIPE, check=True)
    return base64.b64encode(p.stdout).decode("utf-8")


def verify_sig(data: bytes, sig_b64: str) -> bool:
    ensure_keys()
    sig = base64.b64decode(sig_b64)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(sig)
        tf.flush()
        sigfile = tf.name
    try:
        p = subprocess.run(["openssl", "pkeyutl", "-verify", "-pubin", "-inkey", PUBKEY, "-sigfile", sigfile], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode == 0
    finally:
        try:
            os.unlink(sigfile)
        except OSError:
            pass


def genesis_build():
    ensure_dirs()
    ensure_keys()
    genesis = {
        "chain_id": CHAIN_ID,
        "version": "0.1",
        "hash_suite": "SHA256",
        "sig_suite": "Ed25519",
        "reward_multiplier_bp": 10000,
        "impact_score_scale": "int",
        "policy": {
            "local_first": True,
            "no_raw_data": True,
            "deterministic": True
        }
    }
    canon = canonical(genesis)
    root = sha256_bytes(canon)
    sig = sign_bytes(canon)
    built = {
        "genesis": genesis,
        "genesis_merkle_root": root,
        "signature_ed25519": sig,
        "pubkey_ed25519": read_pubkey_pem()
    }
    with open(GENESIS_JSON, "w", encoding="utf-8") as f:
        json.dump(built, f, ensure_ascii=False, indent=2)
    with open(GENESIS_ROOT, "w", encoding="utf-8") as f:
        f.write(root)
    print("OK genesis build ->", GENESIS_JSON)


def contribute_run():
    ensure_dirs()
    # Deterministic contribution
    n = 10000
    result = sum(range(1, n + 1))
    metrics = {
        "task": "MSSC:sum_range",
        "n": n,
        "result": result,
        "seed": 42
    }
    with open(EVID_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(EVID_LOG, "w", encoding="utf-8") as f:
        f.write("MSSC contribution log: sum_range executed deterministically.\n")
    # Pack manifest (hashes only)
    evid = []
    for path in [EVID_METRICS, EVID_LOG]:
        with open(path, "rb") as f:
            evid.append({"name": os.path.basename(path), "sha256": sha256_bytes(f.read())})
    manifest = {"version": "1", "evidence": evid}
    with open(PACK_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    pack_hash = sha256_bytes(canonical(manifest))
    with open(PACK_SHA, "w", encoding="utf-8") as f:
        f.write(pack_hash)
    print("OK contribution ->", PACK_JSON)


def load_genesis_root():
    if not os.path.exists(GENESIS_JSON):
        raise SystemExit("Missing genesis.built.json; run genesis build first.")
    with open(GENESIS_JSON, "r", encoding="utf-8") as f:
        built = json.load(f)
    genesis = built.get("genesis")
    root = built.get("genesis_merkle_root")
    # recompute root for safety
    recomputed = sha256_bytes(canonical(genesis))
    return built, root, recomputed


def poi_attest():
    ensure_dirs()
    ensure_keys()
    if not os.path.exists(PACK_JSON) or not os.path.exists(PACK_SHA):
        raise SystemExit("Missing pack manifest; run contribute run first.")
    built, root, recomputed = load_genesis_root()
    if root != recomputed:
        raise SystemExit("Genesis root mismatch; rebuild genesis.")
    with open(PACK_SHA, "r", encoding="utf-8") as f:
        pack_hash = f.read().strip()
    with open(EVID_METRICS, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    score = int(metrics.get("result", 0) // 1000)
    att = {
        "anchor": {"chain_id": CHAIN_ID, "genesis_merkle_root": root},
        "attester": {"id": "node0-local", "pubkey_ed25519": read_pubkey_pem()},
        "evidence": {"pack_sha256": pack_hash},
        "impact": {"score": score, "method": "MSSC:sum_range", "inputs": {"n": metrics.get("n")}},
        "contribution_id": sha256_bytes((pack_hash + ":" + "node0-local").encode("utf-8")),
        "window": WINDOW
    }
    sig = sign_bytes(canonical(att))
    att["signature_ed25519"] = sig
    with open(POI_JSON, "w", encoding="utf-8") as f:
        json.dump(att, f, ensure_ascii=False, indent=2)
    print("OK attestation ->", POI_JSON)


def verify_attestation(att: dict, allow_replay: bool = False):
    built, root, recomputed = load_genesis_root()
    if root != recomputed:
        return False, "GENESIS_ROOT_MISMATCH", None
    if att.get("anchor", {}).get("genesis_merkle_root") != root:
        return False, "ANCHOR_MISMATCH", None
    # canonical JSON check + signature
    sig = att.get("signature_ed25519")
    if not sig:
        return False, "MISSING_SIGNATURE", None
    att_no_sig = dict(att)
    att_no_sig.pop("signature_ed25519", None)
    if not verify_sig(canonical(att_no_sig), sig):
        return False, "SIG_INVALID", None
    # pack hash check
    if not os.path.exists(PACK_JSON):
        return False, "MISSING_PACK", None
    with open(PACK_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    pack_hash = sha256_bytes(canonical(manifest))
    if att.get("evidence", {}).get("pack_sha256") != pack_hash:
        return False, "PACK_HASH_MISMATCH", None
    # replay check
    contrib_id = att.get("contribution_id")
    if not allow_replay:
        if os.path.exists(REPLAY_DB):
            with open(REPLAY_DB, "r", encoding="utf-8") as f:
                seen = set(json.load(f))
        else:
            seen = set()
        if contrib_id in seen:
            return False, "REPLAY", None
        seen.add(contrib_id)
        with open(REPLAY_DB, "w", encoding="utf-8") as f:
            json.dump(sorted(seen), f, ensure_ascii=False, indent=2)
    # reward quote
    score = int(att.get("impact", {}).get("score", 0))
    multiplier_bp = built.get("genesis", {}).get("reward_multiplier_bp", 10000)
    stable = score * multiplier_bp // 10000
    growth = stable * 2
    quote = {"stable": stable, "growth": growth}
    return True, "OK", quote


def poi_verify(allow_replay: bool = False):
    if not os.path.exists(POI_JSON):
        raise SystemExit("Missing poi_attestation.json; run poi attest first.")
    with open(POI_JSON, "r", encoding="utf-8") as f:
        att = json.load(f)
    valid, reason, quote = verify_attestation(att, allow_replay=allow_replay)
    out = {"valid": valid, "reason": reason, "score": att.get("impact", {}).get("score"), "reward_quote": quote}
    print(json.dumps(out, ensure_ascii=False, indent=2))


def api_up(port: int = 8808):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/api/v1/proof-of-impact/verify":
                self.send_response(404); self.end_headers(); return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            try:
                att = json.loads(body.decode("utf-8"))
            except Exception:
                self.send_response(400); self.end_headers(); return
            valid, reason, quote = verify_attestation(att)
            resp = json.dumps({"valid": valid, "reason": reason, "score": att.get("impact", {}).get("score"), "reward_quote": quote})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.encode("utf-8"))
    print(f"API up on http://127.0.0.1:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser("mssc")
    sub = parser.add_subparsers(dest="cmd")

    g = sub.add_parser("genesis")
    gsub = g.add_subparsers(dest="gcmd")
    gsub.add_parser("build")

    a = sub.add_parser("api")
    asub = a.add_subparsers(dest="acmd")
    up = asub.add_parser("up")
    up.add_argument("--port", type=int, default=8808)

    c = sub.add_parser("contribute")
    csub = c.add_subparsers(dest="ccmd")
    csub.add_parser("run")

    p = sub.add_parser("poi")
    psub = p.add_subparsers(dest="pcmd")
    psub.add_parser("attest")
    v = psub.add_parser("verify")
    v.add_argument("--allow-replay", action="store_true")

    args = parser.parse_args()

    if args.cmd == "genesis" and args.gcmd == "build":
        genesis_build()
    elif args.cmd == "api" and args.acmd == "up":
        api_up(port=args.port)
    elif args.cmd == "contribute" and args.ccmd == "run":
        contribute_run()
    elif args.cmd == "poi" and args.pcmd == "attest":
        poi_attest()
    elif args.cmd == "poi" and args.pcmd == "verify":
        poi_verify(allow_replay=args.allow_replay)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
