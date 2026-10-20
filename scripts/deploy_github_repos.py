#!/usr/bin/env python3
"""Stage 7: deploy the public and private repositories to GitHub.

The token is read in-memory from apis/GITHUB_fine_grained 2.txt and passed to
`gh` and `git` only through the process environment (GH_TOKEN). It is never
printed, logged, or written into .git/config. Remotes use clean
https://github.com/... URLs.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = ROOT / "apis" / "GITHUB_fine_grained 2.txt"

REPOS = {
    "public": {
        "dir": ROOT / "reproducibility" / "public_repo",
        "name": "Javad-Alizargar/nanopore-pino-surrogate",
        "private": False,
        "message": "Public release v1.3.5: final DOCX code breaks, Section 1 prose, abstract range, and table notes (Stage 17)",
        "tag": "v1.3.5",
    },
    "private": {
        "dir": ROOT / "reproducibility" / "github",
        "name": "Javad-Alizargar/nanopore-pino-surrogate-private",
        "private": True,
        "message": "Complete private research archive v1.3.5 (Stage 1-17 verified)",
        "tag": "v1.3.5",
    },
}

IDENTITY = ["-c", "user.name=Javad Alizargar", "-c", "user.email=8javad@ntunhs.edu.tw"]
CRED = ["-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential"]


def load_env() -> dict:
    if not TOKEN_FILE.exists():
        sys.exit("token file missing")
    token = TOKEN_FILE.read_text().strip()
    if not token:
        sys.exit("token file empty")
    return dict(os.environ, GH_TOKEN=token, GH_PROMPT_DISABLED="1", GIT_TERMINAL_PROMPT="0")


def run(args: list[str], env: dict, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if check and p.returncode != 0:
        msg = (p.stdout + p.stderr).strip()
        msg = re.sub(r"(gh[pousr]_|github_pat_)[A-Za-z0-9_]+", r"\1***", msg)
        raise RuntimeError(f"{' '.join(args[:3])} failed ({p.returncode}): {msg[:400]}")
    return p


def git(args: list[str], env: dict, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", *args], env, cwd, check)


def ensure_repo(spec: dict, env: dict) -> str:
    cwd = spec["dir"]
    if not (cwd / ".git").exists():
        git(["init", "-b", "main"], env, cwd)
    git([*IDENTITY, "add", "-A"], env, cwd)
    status = git(["status", "--porcelain"], env, cwd).stdout.strip()
    if status:
        git([*IDENTITY, "commit", "-m", spec["message"]], env, cwd)
    sha = git(["rev-parse", "HEAD"], env, cwd).stdout.strip()
    return sha


def remote_url(cwd: Path, env: dict) -> str:
    p = git(["remote", "get-url", "origin"], env, cwd, check=False)
    return p.stdout.strip()


def create_or_verify(spec: dict, env: dict, push: bool) -> None:
    args = ["gh", "repo", "create", spec["name"],
            "--private" if spec["private"] else "--public",
            "--source", str(spec["dir"]), "--remote", "origin"]
    exists = subprocess.run(["gh", "api", f"repos/{spec['name']}"],
                            env=env, capture_output=True, text=True).returncode == 0
    if not exists:
        if push:
            args.append("--push")
        run(args, env, spec["dir"])
    else:
        git(["remote", "remove", "origin"], env, spec["dir"], check=False)
        git(["remote", "add", "origin", f"https://github.com/{spec['name']}.git"], env, spec["dir"])
    url = remote_url(spec["dir"], env)
    assert url == f"https://github.com/{spec['name']}.git", f"unexpected remote {url}"
    assert "@" not in url, "token must never appear in the remote URL"


def push_branch(spec: dict, env: dict, force: bool = False) -> None:
    mid = ["push"] + (["--force"] if force else []) + ["-u", "origin", "main"]
    git([*CRED, *mid], env, spec["dir"])


def verify_private_flag(name: str, expected: bool, env: dict) -> bool:
    p = run(["gh", "api", f"repos/{name}", "--jq", ".private"], env, ROOT)
    return p.stdout.strip().lower() == ("true" if expected else "false")


def enable_pages(name: str, env: dict) -> dict:
    body = ["gh", "api", "-X", "POST", f"repos/{name}/pages",
            "-f", "source[branch]=main", "-f", "source[path]=/"]
    p = run(body, env, ROOT, check=False)
    if p.returncode != 0 and "409" not in (p.stdout + p.stderr):
        print("Pages POST note:", (p.stdout + p.stderr).strip()[:200])
    for _ in range(10):
        info = run(["gh", "api", f"repos/{name}/pages"], env, ROOT, check=False)
        if info.returncode == 0:
            import json
            return json.loads(info.stdout)
        time.sleep(6)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["public", "private"])
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--force", action="store_true", help="force-push main (after a rebuilt history)")
    args = ap.parse_args()
    env = load_env()
    user = run(["gh", "api", "user", "--jq", ".login"], env, ROOT).stdout.strip()
    print("authenticated user:", user)
    out = {}
    names = [args.only] if args.only else ["public", "private"]
    for key in names:
        spec = REPOS[key]
        sha = ensure_repo(spec, env)
        create_or_verify(spec, env, push=not args.no_push)
        ok_private = verify_private_flag(spec["name"], spec["private"], env)
        print(f"{key}: private={ok_private} head={sha}")
        if not ok_private:
            sys.exit(f"visibility mismatch for {spec['name']}")
        if not args.no_push:
            push_branch(spec, env, force=args.force)
            if spec["tag"]:
                git([*IDENTITY, "tag", "-f", spec["tag"]], env, spec["dir"])
                git([*CRED, "push", "-f", "origin", spec["tag"]], env, spec["dir"])
            sha = git(["rev-parse", "HEAD"], env, spec["dir"]).stdout.strip()
        out[key] = {"repo": spec["name"], "sha": sha,
                    "visibility": "private" if spec["private"] else "public",
                    "visibility_ok": ok_private}
        if key == "public":
            pages = enable_pages(spec["name"], env)
            out[key]["pages"] = {"html_url": pages.get("html_url"),
                                 "status": pages.get("status"),
                                 "url": "https://javad-alizargar.github.io/nanopore-pino-surrogate/web_simulator/"}
            out[key]["portal"] = "https://pino.statwhy.com/"
    import json
    (ROOT / "results" / "stage17_release.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
