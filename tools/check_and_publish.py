"""Automatic approval of a contribution (runs in GitHub Actions on every new issue).

Reads the issue (from environment variables — never from the shell), downloads the attached
.zip, keeps only what passes the checks, and adds it to packs/ + index.json:

* a valid Sort by Member members pack (manifest format 1), size limits (zip bombs refused);
* group / member names usable as folder names on macOS and Windows;
* every picture is a JPEG with exactly the labelled face in it (YuNet face detector, MIT):
  pictures without that face are dropped, the rest is kept;
* pictures already in the collection are skipped; members / groups left empty are dropped;
* contributors' face numbers (embeddings) are removed — apps compute their own from the crops.

It can't know whether a face is labelled with the right member.
Writes a Markdown report to stdout; exit code 0 = published, 2 = nothing usable.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "tools" / "model" / "face_detection_yunet_2023mar.onnx"
MAX_DOWNLOAD = 30 << 20             # GitHub attachments are ≤ 25 MB
MAX_ENTRY = 16 << 20
MAX_TOTAL = 400 << 20
MAX_REFS = 3000
FORMAT = 1
BAD_CHARS = set('/\\:*?"<>|') | {chr(i) for i in range(32)}
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10))}
ATTACHMENT = re.compile(r"https://github\.com/(?:user-attachments/files/\d+/|[\w.-]+/[\w.-]+/files/\d+/)"
                        r"[^\s)\]>\"']+?\.zip")


def ok_name(name) -> bool:
    if not isinstance(name, str):
        return False
    n = name.strip()
    return (0 < len(n) <= 100 and n not in (".", "..") and not (BAD_CHARS & set(n))
            and not n.endswith(".") and n.split(".")[0].upper() not in RESERVED)


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "kpop-shared-references-bot"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(MAX_DOWNLOAD + 1)
    if len(data) > MAX_DOWNLOAD:
        raise ValueError("the file is larger than 25 MB")
    return data


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter > 0 else 0.0


class FaceCheck:
    def __init__(self) -> None:
        self.det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), 0.6, 0.3, 5000)

    def labelled_face_present(self, img: Image.Image, box) -> bool:
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or not len(faces):
            return False
        if not (isinstance(box, list) and len(box) == 4):
            return True
        want = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        return any(iou(want, [f[0] / w, f[1] / h, f[2] / w, f[3] / h]) >= 0.3 for f in faces)


def existing_crops() -> set[str]:
    have = set()
    for p in (ROOT / "packs").glob("*.sbmpack"):
        try:
            with zipfile.ZipFile(p) as z:
                m = json.loads(z.read("manifest.json"))
            have |= {r.get("sha256") for g in m["groups"] for mm in g["members"]
                     for r in mm["references"]}
        except Exception:
            continue
    return have


def check(data: bytes, faces: FaceCheck, have: set[str]):
    """→ (cleaned pack bytes or None, report lines, kept counts)"""
    notes: list[str] = []
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        infos = z.infolist()
        if sum(i.file_size for i in infos) > MAX_TOTAL or any(i.file_size > MAX_ENTRY for i in infos):
            return None, ["The file is unusually large inside — not accepted."], None
        man = json.loads(z.read("manifest.json"))
    except Exception:
        return None, ["That isn't a Sort by Member references file (save it with *Contribute "
                      "Your References…* in the app)."], None
    if man.get("format") != FORMAT or not isinstance(man.get("groups"), list):
        return None, ["This file was made by a different version of the app."], None
    names = set(z.namelist())
    out_groups, files, dropped = [], {}, {"no face": 0, "already shared": 0, "unreadable": 0,
                                          "bad name": 0}
    total = 0
    for g in man["groups"][:200]:
        if not isinstance(g, dict) or not ok_name(g.get("name")):
            dropped["bad name"] += 1
            continue
        members = []
        for m in (g.get("members") or [])[:200]:
            if not isinstance(m, dict) or not ok_name(m.get("name")):
                dropped["bad name"] += 1
                continue
            refs = []
            for r in (m.get("references") or []):
                if total >= MAX_REFS:
                    break
                f = r.get("file") if isinstance(r, dict) else None
                if not (isinstance(f, str) and re.fullmatch(r"refs/[\w.-]+\.jpg", f) and f in names):
                    dropped["unreadable"] += 1
                    continue
                raw = z.read(f)
                sha = hashlib.sha256(raw).hexdigest()
                if sha in have:
                    dropped["already shared"] += 1
                    continue
                try:
                    img = Image.open(io.BytesIO(raw))
                    if img.format != "JPEG" or not (64 <= min(img.size) and max(img.size) <= 2000):
                        raise ValueError
                    img.load()
                except Exception:
                    dropped["unreadable"] += 1
                    continue
                if not faces.labelled_face_present(img, r.get("face_box")):
                    dropped["no face"] += 1
                    continue
                have.add(sha)
                files[f] = raw
                refs.append({"file": f, "sha256": sha,
                             "source_sha256": str(r.get("source_sha256") or sha)[:64],
                             "face_box": r.get("face_box"), "review_status": "unreviewed",
                             "det_score": r.get("det_score"), "face_px": r.get("face_px"),
                             "frontalness": r.get("frontalness")})
                total += 1
            if refs:
                members.append({"name": m["name"].strip(), "references": refs})
        if members:
            out_groups.append({"name": g["name"].strip(), "members": members})
    for why, n in dropped.items():
        if n:
            notes.append(f"- left out: {n} ({why})")
    if not out_groups:
        return None, ["Nothing usable was found — every picture was left out."] + notes, None
    clean = {"format": FORMAT, "app_version": str(man.get("app_version", ""))[:20],
             "model_key": None, "created": str(man.get("created", ""))[:40], "groups": out_groups}
    clean["id"] = hashlib.sha256(json.dumps(clean, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as out:
        for f, raw in files.items():
            out.writestr(f, raw)
        out.writestr("manifest.json", json.dumps(clean, ensure_ascii=False, indent=1))
    counts = (len(out_groups), sum(len(g["members"]) for g in out_groups), total)
    return buf.getvalue(), notes, counts


def publish(pack: bytes, title: str, contributor: str, groups: list[str], counts) -> dict:
    with zipfile.ZipFile(io.BytesIO(pack)) as z:
        pid = json.loads(z.read("manifest.json"))["id"]
    (ROOT / "packs").mkdir(exist_ok=True)
    target = ROOT / "packs" / f"{pid}.sbmpack"
    target.write_bytes(pack)
    entry = {"id": pid, "title": title, "file": f"packs/{target.name}",
             "sha256": hashlib.sha256(pack).hexdigest(), "bytes": len(pack), "groups": groups,
             "members": counts[1], "references": counts[2], "added": date.today().isoformat(),
             "contributor": contributor}
    index = ROOT / "index.json"
    raw = json.loads(index.read_text(encoding="utf-8"))
    raw["packs"] = [p for p in raw.get("packs", []) if p.get("id") != pid] + [entry]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=ROOT, delete=False) as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    Path(f.name).replace(index)
    return entry


def main() -> int:
    body = os.environ.get("ISSUE_BODY", "")
    title = re.sub(r"\s+", " ", os.environ.get("ISSUE_TITLE", "")).strip()[:100] or "Contribution"
    user = re.sub(r"[^\w-]", "", os.environ.get("ISSUE_USER", ""))[:40]
    urls = list(dict.fromkeys(ATTACHMENT.findall(body)))[:5]
    if not urls:
        print("No .zip attached — drag the file saved by *Contribute Your References…* into the "
              "issue and edit it.")
        return 2
    faces, have = FaceCheck(), existing_crops()
    report, published = [], 0
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        try:
            data = download(url)
        except Exception as exc:
            report.append(f"**{name}**: couldn't download it ({exc}).")
            continue
        pack, notes, counts = check(data, faces, have)
        if pack is None:
            report.append(f"**{name}**: not added.\n" + "\n".join(notes))
            continue
        with zipfile.ZipFile(io.BytesIO(pack)) as z:
            groups = [g["name"] for g in json.loads(z.read("manifest.json"))["groups"]]
        publish(pack, title, user, groups, counts)
        published += 1
        report.append(f"**{name}**: added — {counts[0]} group(s), {counts[1]} member(s), "
                      f"{counts[2]} face(s) ({', '.join(groups)}).\n" + "\n".join(notes))
    print("\n\n".join(report) + ("\n\nEveryone gets it the next time they press *Check for Shared "
                                 "References* in the app. Thank you!" if published else ""))
    return 0 if published else 2


if __name__ == "__main__":
    sys.exit(main())
