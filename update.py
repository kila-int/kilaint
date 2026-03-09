"""
Kila Int -- Auto Updater
========================
Runs ingestion, reparses, and pushes to GitHub (triggering Render redeploy).

Usage:
    python update.py              # ingest + reparse + push
    python update.py --no-push    # ingest + reparse only (local)
"""

import subprocess
import sys
import os
import argparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd, description):
    print(f"\n[update] {description}...")
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"[error] {description} failed:")
        if result.stderr:
            print(result.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Kila Int Auto Updater")
    parser.add_argument("--no-push", action="store_true", help="Skip git push to Render")
    args = parser.parse_args()

    print("=" * 50)
    print("  KILA INT — AUTO UPDATE")
    print("=" * 50)

    # 1. Ingest new messages
    if not run([sys.executable, "tg_ingest.py", "--backfill"], "Ingesting from Telegram"):
        print("[!] Ingestion failed, continuing with existing data...")

    # 2. Reparse all messages
    if not run([sys.executable, "reparse.py"], "Reparsing all messages"):
        print("[!] Reparse failed, aborting.")
        return

    if args.no_push:
        print("\n[done] Local update complete (--no-push).")
        return

    # 3. Git commit and push
    print("\n[update] Pushing to GitHub...")
    subprocess.run(["git", "add", "kila.db"], cwd=PROJECT_DIR)

    # Check if there are changes to commit
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_DIR)
    if status.returncode == 0:
        print("[update] No new data to push.")
        return

    run(["git", "commit", "-m", "Auto-update: fresh data"], "Committing")
    run(["git", "push"], "Pushing to GitHub")

    print("\n" + "=" * 50)
    print("  UPDATE COMPLETE — Render will redeploy shortly")
    print("=" * 50)


if __name__ == "__main__":
    main()
