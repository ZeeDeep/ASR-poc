#!/usr/bin/env python3
import argparse, json, requests, sys, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="local audio file")
    ap.add_argument("--url", default="http://localhost:8080/v1/transcribe")
    ap.add_argument("--model", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--no-sep", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print("File not found.", file=sys.stderr); sys.exit(1)

    params = {
        "model_size": args.model,
        "language_hint": args.lang,
        "separation_enabled": not args.no-sep if hasattr(args, "no-sep") else True
    }

    files = {"file": open(args.path, "rb")}
    data = {"params_json": json.dumps({k:v for k,v in params.items() if v is not None})}
    r = requests.post(args.url, files=files, data=data, timeout=600)
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

