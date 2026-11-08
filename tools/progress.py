#!/usr/bin/env python3
"""Update docs/progress.json, the data behind the GitHub Pages progress site.

  progress.py now "<what is happening>"
  progress.py item <id> <todo|active|done> [note]
  progress.py media <docs-relative-path> "<caption>" [--item <id>] [--poster <path>]
  progress.py stat "<label>" "<value>"
  progress.py log "<text>"
  progress.py push "<commit message>"     (commits docs/ and pushes to origin main)

Media files must already be inside docs/ (usually docs/media/).
"""
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
PATH = os.path.join(DOCS, "progress.json")


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load():
    with open(PATH) as f:
        return json.load(f)


def save(d):
    d["updated"] = now_iso()
    fd, tmp = tempfile.mkstemp(dir=DOCS)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    os.replace(tmp, PATH)


def find_item(d, item_id):
    for phase in d["phases"]:
        for it in phase["items"]:
            if it["id"] == item_id:
                return it
    sys.exit(f"unknown item id: {item_id}")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout.strip()


def main(a):
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    if cmd == "push":
        msg = a[1] if len(a) > 1 else "Update progress site"
        git("add", "docs")
        if subprocess.run(["git", "-C", ROOT, "diff", "--cached", "--quiet"]).returncode == 0:
            print("nothing to commit")
        else:
            git("commit", "-m", msg)
        print(git("push", "origin", "main"))
        return
    d = load()
    if cmd == "now":
        d["now"] = a[1]
    elif cmd == "item":
        it = find_item(d, a[1])
        if a[2] not in ("todo", "active", "done"):
            sys.exit("status must be todo, active or done")
        it["status"] = a[2]
        if len(a) > 3:
            it["note"] = a[3]
        if a[2] == "done":
            it["done"] = now_iso()
            d["log"].insert(0, {"time": now_iso(), "text": f"Done: {it['name']}" + (f". {a[3]}" if len(a) > 3 else "")})
    elif cmd == "media":
        rel = a[1]
        if not os.path.exists(os.path.join(DOCS, rel)):
            sys.exit(f"missing file docs/{rel}")
        entry = {"src": rel, "caption": a[2], "time": now_iso(),
                 "type": "video" if rel.lower().endswith((".mp4", ".webm")) else "image"}
        rest = a[3:]
        while rest:
            flag, val = rest[0], rest[1]
            rest = rest[2:]
            if flag == "--item":
                find_item(d, val).setdefault("media", []).insert(0, rel)
                entry["item"] = val
            elif flag == "--poster":
                entry["poster"] = val
            else:
                sys.exit(f"unknown flag {flag}")
        d["gallery"] = [g for g in d["gallery"] if g["src"] != rel]
        d["gallery"].insert(0, entry)
    elif cmd == "stat":
        d["stats"][a[1]] = a[2]
    elif cmd == "log":
        d["log"].insert(0, {"time": now_iso(), "text": a[1]})
    else:
        sys.exit(__doc__)
    save(d)


main(sys.argv[1:])
