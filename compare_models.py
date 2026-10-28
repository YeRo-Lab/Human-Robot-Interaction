# compare_models.py v4.1 (Python 3) - runs the same labeled test photos through DeepFace models (and optionally InsightFace)
# and saves comparable results, including a STRANGER test (people the system must NOT recognize).
#
# Folders (same as before):
#   face_database/<Person>/*.jpg   enrolled people (never modified; the script works on a copy called compare_db)
#   test_captures/<Person>/*.jpg   labeled test photos. Photos byte-identical to an enrolled photo are skipped.
#
# Stranger test: --strangers Hasaan   (comma list of people from your folders)
#   Those people are REMOVED from the copy of the database, and all their photos (from face_database/<name> and
#   test_captures/<name>) become "stranger" photos. A good model should reject them. Nothing is moved or deleted in
#   your real folders. Pick someone with 9+ photos; the remaining people stay enrolled.
#
# Usage examples:
#   py -3.13 compare_models.py --models all --detectors all --strangers Hasaan --out results_v2 --max-hours 9
#   py -3.13 compare_models.py --models InsightFace --strangers Hasaan --out results_insight     (needs the InsightFace venv)
#   py -3.13 compare_models.py --models all --detectors all --strangers Hasaan --out results_v2 --resume   (after a crash)
#
# Output (in --out): per_photo.csv and summary.csv, rewritten after every combination so nothing is lost.
import argparse
import csv
import hashlib
import os
import re
import shutil
import statistics
import time
from collections import defaultdict


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def list_images(folder):
    out = []
    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(os.path.join(root, name))
    return out


def person_of(path):
    return os.path.basename(os.path.dirname(os.path.normpath(path)))


def visit_of(path):
    """face_capture_<timestamp>_<n>.jpg -> '<timestamp>' so the 3 photos of one visit vote together."""
    m = re.search(r"face_capture_(\d+)_\d+", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def pick_person(results):
    """Same rule as server.py: majority vote, ties go to the lower average distance."""
    votes = {}
    for name, distance in results:
        if name is not None:
            votes.setdefault(name, []).append(distance)
    if not votes:
        return None
    return min(votes, key=lambda n: (-len(votes[n]), sum(votes[n]) / len(votes[n])))


# ---------------------------------------------------------------- DeepFace engine

def load_model(model_name):
    from deepface import DeepFace
    t0 = time.time()
    try:
        DeepFace.build_model(model_name=model_name, task="facial_recognition")
    except TypeError:
        DeepFace.build_model(model_name)
    return time.time() - t0


def default_threshold(model_name, metric):
    try:
        from deepface.modules import verification
        return verification.find_threshold(model_name, metric)
    except Exception:
        return None


def deepface_query(img_path, db_path, model_name, detector, metric):
    """Returns [(person, distance, db_file), ...] sorted closest first, over every photo in the database."""
    from deepface import DeepFace
    kwargs = dict(img_path=img_path, db_path=db_path, model_name=model_name, detector_backend=detector,
                  distance_metric=metric, enforce_detection=False, silent=True)
    try:
        res = DeepFace.find(threshold=10.0, **kwargs)  # huge threshold = return every photo, not only close ones
    except TypeError:
        res = DeepFace.find(**kwargs)
    if not res or len(res[0]) == 0:
        return []
    rows = res[0][["identity", "distance"]].values.tolist()
    rows = [(person_of(p), float(d), os.path.basename(p)) for p, d in rows]
    rows.sort(key=lambda r: r[1])
    return rows


# ---------------------------------------------------------------- InsightFace engine (optional, not part of DeepFace)

class InsightBackend(object):
    """InsightFace buffalo_l (SCRFD detector + ArcFace r50 recognizer). Embeds every enrolled photo once, then
    answers queries with cosine distance = 1 - similarity. A photo with no detected face has no embedding."""

    def __init__(self, db_path):
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
        self.cv2, self.np = cv2, np
        t0 = time.time()
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.load_s = time.time() - t0
        t1 = time.time()
        self.entries = []
        for p in list_images(db_path):
            emb = self.embed(p)
            if emb is not None:
                self.entries.append((person_of(p), os.path.basename(p), emb))
        self.index_s = time.time() - t1

    def embed(self, path):
        img = self.cv2.imread(path)
        if img is None:
            return None
        faces = self.app.get(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = self.np.asarray(face.normed_embedding, dtype="float32")
        return emb / (self.np.linalg.norm(emb) or 1.0)

    def query(self, path):
        emb = self.embed(path)
        if emb is None:
            return []
        rows = [(person, 1.0 - float(self.np.dot(emb, v)), name) for person, name, v in self.entries]
        rows.sort(key=lambda r: r[1])
        return rows


INSIGHT_THRESHOLD = 0.6  # assumed: InsightFace users commonly accept cosine similarity >= ~0.4 (distance <= 0.6)


# ---------------------------------------------------------------- scoring

def analyse(rows, true_person):
    """Top person, runner-up person, margin, and how well the true person separates from the best wrong person."""
    best = {}
    for person, d, _name in rows:
        if person not in best or d < best[person]:
            best[person] = d
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    top = ranked[0] if ranked else (None, None)
    second = ranked[1] if len(ranked) > 1 else (None, None)
    margin = (second[1] - top[1]) if second[1] is not None and top[1] is not None else None
    others = [d for p, d in best.items() if p != true_person]
    true_d = best.get(true_person)
    sep = (min(others) - true_d) if others and true_d is not None else None
    return top, second, margin, sep


def open_set(genuine, impostors, threshold):
    """genuine: [(identified_correctly, top_distance)] for enrolled people; impostors: [top_distance or None] for
    strangers. Returns FAR/FRR at the model's default threshold, the equal-error point, and the strictest cutoff that
    lets no stranger in, with the share of enrolled photos still recognized at that cutoff."""
    out = {"far_default": None, "frr_default": None, "eer": None, "eer_threshold": None,
           "strict_threshold": None, "genuine_ok_at_strict": None}
    if not genuine or not impostors:
        return out

    def far(t):
        return sum(1 for d in impostors if d is not None and d <= t) / float(len(impostors))

    def frr(t):
        return 1.0 - sum(1 for ok, d in genuine if ok and d is not None and d <= t) / float(len(genuine))

    if threshold is not None:
        out["far_default"], out["frr_default"] = far(threshold), frr(threshold)
    cands = sorted({d for _ok, d in genuine if d is not None} | {d for d in impostors if d is not None})
    if cands:
        best_t = min(cands, key=lambda t: (abs(far(t) - frr(t)), t))
        out["eer"], out["eer_threshold"] = (far(best_t) + frr(best_t)) / 2.0, best_t
    seen = [d for d in impostors if d is not None]
    if seen:
        strict = min(seen) - 1e-6  # just under the closest stranger: nobody unknown gets in
        out["strict_threshold"] = strict
        out["genuine_ok_at_strict"] = 1.0 - frr(strict)
    return out


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(v, digits=3):
    return "" if v is None else round(v, digits)


# ---------------------------------------------------------------- one combination

def run_combo(model, detector, tests, work_db, metric, db_images):
    """tests: [(path, true_name, kind)] with kind 'enrolled' or 'stranger'. Returns (summary row, per-photo rows)."""
    if model == "InsightFace":
        backend = InsightBackend(work_db)
        load_s, index_s = backend.load_s, backend.index_s
        threshold = INSIGHT_THRESHOLD
        run_query = backend.query
    else:
        load_s, index_s = load_model(model), 0.0
        threshold = default_threshold(model, metric)
        run_query = lambda path: deepface_query(path, work_db, model, detector, metric)

    first_s, times, per_photo, photo_rows = None, [], [], []
    for i, (path, truth, kind) in enumerate(tests):
        t0 = time.time()
        try:
            rows = run_query(path)
            err = ""
        except Exception as e:
            if i == 0:
                raise  # the very first query builds the index; if that fails the whole combination is unusable
            rows, err = [], str(e)[:120]
        took = time.time() - t0
        if i == 0:
            first_s = took + index_s  # includes building the index for this model
        else:
            times.append(took)
        top, second, margin, sep = analyse(rows, truth if kind == "enrolled" else None)
        top_file = rows[0][2] if rows else ""
        top3 = "; ".join("%s:%s" % (r[2], fmt(r[1], 3)) for r in rows[:3])
        accepted = top[1] is not None and (threshold is None or top[1] <= threshold)
        pred = top[0] if accepted else None
        correct = int(pred == truth) if kind == "enrolled" else int(pred is None)  # stranger: correct = rejected
        per_photo.append((path, truth, kind, pred, top[1], margin, sep, top[0]))
        photo_rows.append({
            "model": model, "detector": detector, "kind": kind, "file": path, "true": truth, "pred": pred or "",
            "correct": correct, "top_person": top[0] or "", "top_distance": fmt(top[1], 4),
            "second_person": second[0] or "", "second_distance": fmt(second[1], 4), "closest_db_file": top_file,
            "closest_3": top3, "margin": fmt(margin, 4), "true_vs_best_other": fmt(sep, 4),
            "seconds": fmt(took, 3), "error": err})
        print("  %-8s %-45s true=%-8s pred=%-8s d=%s" % (kind[:8], os.path.basename(path), truth, pred, fmt(top[1], 3)))

    enrolled = [r for r in per_photo if r[2] == "enrolled"]
    strangers = [r for r in per_photo if r[2] == "stranger"]
    correct = [r for r in enrolled if r[3] == r[1]]
    wrong = [r for r in enrolled if r[3] != r[1]]
    visits = defaultdict(list)
    for path, truth, _k, pred, d, _m, _s, _t in enrolled:
        visits[(truth, visit_of(path))].append((pred, d if d is not None else 9.0))
    visit_ok = sum(1 for (truth, _v), res in visits.items() if pick_person(res) == truth)

    # open-set view: an enrolled photo counts as recognized when the closest person is the true one AND is within the cutoff
    o = open_set([(r[7] == r[1], r[4]) for r in enrolled], [r[4] for r in strangers], threshold)

    summary = {
        "model": model, "detector": detector, "status": "ok", "metric": metric, "db_images": db_images,
        "default_threshold": fmt(threshold, 3), "test_photos": len(enrolled),
        "photo_accuracy": fmt(len(correct) / float(len(enrolled)), 3) if enrolled else "",
        "visits": len(visits), "visit_accuracy_vote": fmt(visit_ok / float(len(visits)), 3) if visits else "",
        "mean_margin_when_correct": fmt(mean([r[5] for r in correct]), 3),
        "mean_margin_when_wrong": fmt(mean([r[5] for r in wrong]), 3),
        "median_true_vs_best_other": fmt(median([r[6] for r in enrolled]), 3),
        "stranger_photos": len(strangers),
        "stranger_false_accept_default": fmt(o["far_default"], 3),
        "enrolled_rejected_default": fmt(o["frr_default"], 3),
        "eer": fmt(o["eer"], 3), "eer_threshold": fmt(o["eer_threshold"], 3),
        "strict_threshold_no_stranger_in": fmt(o["strict_threshold"], 3),
        "enrolled_recognized_at_strict": fmt(o["genuine_ok_at_strict"], 3),
        "model_load_s": fmt(load_s, 2), "index_build_plus_first_query_s": fmt(first_s, 2),
        "mean_seconds_per_photo": fmt(mean(times), 3)}
    return summary, photo_rows


SUMMARY_FIELDS = ["model", "detector", "status", "metric", "db_images", "default_threshold", "test_photos",
                  "photo_accuracy", "visits", "visit_accuracy_vote", "mean_margin_when_correct",
                  "mean_margin_when_wrong", "median_true_vs_best_other", "stranger_photos",
                  "stranger_false_accept_default", "enrolled_rejected_default", "eer", "eer_threshold",
                  "strict_threshold_no_stranger_in", "enrolled_recognized_at_strict", "model_load_s",
                  "index_build_plus_first_query_s", "mean_seconds_per_photo"]
PHOTO_FIELDS = ["model", "detector", "kind", "file", "true", "pred", "correct", "top_person", "top_distance",
                "second_person", "second_distance", "closest_db_file", "closest_3", "margin", "true_vs_best_other",
                "seconds", "error"]


def save(out, summary, photo_rows):
    """Rewritten after every combination, so a crash or Ctrl+C never loses finished results."""
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(out, "per_photo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PHOTO_FIELDS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(photo_rows)


def load_previous(out):
    """For --resume: finished ('ok') combinations and their photo rows from an earlier run in the same folder."""
    done, photo_rows = [], []
    sp, pp = os.path.join(out, "summary.csv"), os.path.join(out, "per_photo.csv")
    if os.path.isfile(sp):
        with open(sp, newline="", encoding="utf-8", errors="replace") as f:
            done = [r for r in csv.DictReader(f) if r.get("status") == "ok"]
    ok_keys = {(r["model"], r["detector"]) for r in done}
    if os.path.isfile(pp):
        with open(pp, newline="", encoding="utf-8", errors="replace") as f:
            photo_rows = [r for r in csv.DictReader(f) if (r["model"], r["detector"]) in ok_keys]
    return done, photo_rows


ALL_MODELS = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace",
              "GhostFaceNet", "Buffalo_L"]
ALL_DETECTORS = ["yunet", "retinaface", "opencv", "mtcnn", "ssd", "centerface", "fastmtcnn", "mediapipe", "yolov8n",
                 "yolov11n"]  # most useful first, so --max-hours cuts the least useful ones


def expand(arg, everything):
    out = []
    for item in [x.strip() for x in arg.split(",") if x.strip()]:
        if item.lower() == "all":
            out.extend(everything)
        else:
            out.append(item)
    return list(dict.fromkeys(out))  # no duplicates, keep order


def gather_tests(db, tests_dir, strangers):
    """(enrolled test photos, stranger photos). Test photos that are byte-identical to an enrolled photo are skipped."""
    stranger_names = set(strangers)
    db_hashes = {md5(p) for p in list_images(db) if person_of(p) not in stranger_names}
    enrolled, skipped = [], 0
    for p in list_images(tests_dir):
        if person_of(p) in stranger_names:
            continue
        if md5(p) in db_hashes:
            skipped += 1
        else:
            enrolled.append((p, person_of(p), "enrolled"))
    stranger_photos, seen = [], set()
    for name in strangers:
        found = [p for p in list_images(os.path.join(db, name))] + [p for p in list_images(os.path.join(tests_dir, name))]
        if not found:
            print("WARNING: no photos found for stranger '%s' in %s or %s" % (name, db, tests_dir))
        for p in found:
            h = md5(p)
            if h not in seen:
                seen.add(h)
                stranger_photos.append((p, name, "stranger"))
    return enrolled, stranger_photos, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="face_database")
    ap.add_argument("--tests", default="test_captures")
    ap.add_argument("--models", default="VGG-Face,Facenet512,ArcFace",
                    help="comma list; 'all' = every DeepFace model; add InsightFace by name (needs its own install)")
    ap.add_argument("--detectors", default="opencv", help="comma list, or 'all'")
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--out", default="results")
    ap.add_argument("--strangers", default="", help="comma list of people to treat as unknown (removed from the DB copy)")
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this long (0 = no limit)")
    ap.add_argument("--resume", action="store_true", help="skip combinations already finished in --out")
    args = ap.parse_args()

    if not os.path.isdir(args.db) or not os.path.isdir(args.tests):
        raise SystemExit("Need '%s' and '%s' folders here. Put test photos in %s/<Person>/*.jpg" % (args.db, args.tests, args.tests))

    strangers = [s.strip() for s in args.strangers.split(",") if s.strip()]
    enrolled, stranger_photos, skipped = gather_tests(args.db, args.tests, strangers)
    if skipped:
        print("Skipped %d test photo(s) that are already in the face database." % skipped)
    tests = enrolled + stranger_photos
    if not enrolled:
        raise SystemExit("No usable test photos for enrolled people.")
    print("Enrolled test photos:", {p: sum(1 for t in enrolled if t[1] == p) for p in sorted({t[1] for t in enrolled})})
    print("Stranger photos:", {p: sum(1 for t in stranger_photos if t[1] == p) for p in sorted({t[1] for t in stranger_photos})} or "none")

    work_db = "compare_db"  # fresh copy each run, so index build times are real and face_database is never touched
    if os.path.isdir(work_db):
        shutil.rmtree(work_db)
    shutil.copytree(args.db, work_db, ignore=shutil.ignore_patterns("*.pkl"))
    for name in strangers:
        shutil.rmtree(os.path.join(work_db, name), ignore_errors=True)  # the stranger must be unknown to the model
    db_images = len(list_images(work_db))
    print("Database copy: %d photos of %s" % (db_images, sorted(d for d in os.listdir(work_db) if os.path.isdir(os.path.join(work_db, d)))))
    os.makedirs(args.out, exist_ok=True)

    models = expand(args.models, ALL_MODELS)
    detectors = expand(args.detectors, ALL_DETECTORS)
    combos = [(m, d) for d in detectors for m in models if m != "InsightFace"]
    if "InsightFace" in models:
        combos.append(("InsightFace", "scrfd"))  # InsightFace brings its own detector

    summary, photo_rows = ([], [])
    if args.resume:
        summary, photo_rows = load_previous(args.out)
        finished = {(r["model"], r["detector"]) for r in summary}
        combos = [c for c in combos if c not in finished]
        print("Resuming: %d combination(s) already done, %d left." % (len(finished), len(combos)))

    start = time.time()
    for n, (model, detector) in enumerate(combos, start=1):
        if args.max_hours and (time.time() - start) > args.max_hours * 3600:
            print("\nTime limit reached (%.1f h). Stopping with %d combination(s) not run." % (args.max_hours, len(combos) - n + 1))
            break
        print("\n=== [%d/%d] %s / %s ===" % (n, len(combos), model, detector))
        try:
            row, rows = run_combo(model, detector, tests, work_db, args.metric, db_images)
            summary.append(row)
            photo_rows.extend(rows)
        except KeyboardInterrupt:
            print("Stopped by user. Saving what finished.")
            save(args.out, summary, photo_rows)
            raise SystemExit(1)
        except Exception as e:
            reason = str(e).replace("\n", " ").encode("ascii", "replace").decode("ascii")[:150]  # plain ASCII: DeepFace puts emoji in errors
            print("  SKIPPED (%s: %s)" % (type(e).__name__, reason))
            summary.append({"model": model, "detector": detector, "status": "failed: " + reason,
                            "metric": args.metric, "db_images": db_images})
        save(args.out, summary, photo_rows)

    print("\n=== SUMMARY ===")
    for s in summary:
        if s["status"] != "ok":
            print("%s / %s: %s" % (s["model"], s["detector"], s["status"]))
            continue
        print("%s / %s: acc %s, strangers let in %s (default cutoff), EER %s, %s s/photo" % (
            s["model"], s["detector"], s.get("photo_accuracy"), s.get("stranger_false_accept_default"),
            s.get("eer"), s.get("mean_seconds_per_photo")))
    print("\nSaved %s and %s" % (os.path.join(args.out, "summary.csv"), os.path.join(args.out, "per_photo.csv")))


if __name__ == "__main__":
    main()