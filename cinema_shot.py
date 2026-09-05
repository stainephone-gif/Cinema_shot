#!/usr/bin/env python3
"""
cinema_shot — раскадровка видео по ключевым кадрам.

Программа проходит по видео и сохраняет в отдельную папку JPG-кадры в тех местах,
где что-то меняется:

  * склейка (монтажный стык) — сохраняется последний кадр до склейки и первый после;
  * смена крупности внутри плана (деталь / крупный / средний / общий) — по размеру
    лица в кадре;
  * изменение количества героев в кадре — по числу найденных лиц (и, опционально, фигур).

Дополнительно сохраняются первый и последний кадры видео, а по желанию — кадр
раз в N секунд внутри длинных статичных планов.

Рядом с кадрами кладутся manifest.csv / manifest.json с описанием каждого кадра
и, по желанию, contact_sheet.jpg — общая «простыня» из миниатюр.

Зависимости: opencv-python (или opencv-python-headless) >= 4.5.4, numpy.
Модель лиц: models/face_detection_yunet_2023mar.onnx (скачивается автоматически, если нет).

Пример:
    python cinema_shot.py clip.mp4
    python cinema_shot.py clip.mp4 -o storyboard --contact-sheet
    python cinema_shot.py clip.mp4 --cut-threshold 0.25 --stable 8 --bodies
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, deque
from dataclasses import dataclass, asdict
from typing import Iterable

import cv2
import numpy as np

try:  # убрать служебные предупреждения OpenCV из вывода
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

ANALYSIS_WIDTH = 640  # ширина кадра, на которой идёт анализ (ускоряет работу)

# Классы крупности по отношению высоты лица к высоте кадра.
# Порог перечислен от самого крупного к самому общему.
SHOT_SIZE_BY_FACE = [
    (0.55, "ECU", "деталь"),
    (0.28, "CU", "крупный"),
    (0.12, "MS", "средний"),
    (0.00, "WS", "общий"),
]
NO_FACE_SIZE = ("NF", "без лиц")

REASON_RU = {
    "start": "начало видео",
    "end": "конец видео",
    "cut_before": "до склейки",
    "cut_after": "после склейки",
    "size_change": "смена крупности",
    "count_change": "изменение числа героев",
    "interval": "кадр по интервалу",
}


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------


@dataclass
class FrameInfo:
    """Результаты анализа одного кадра (на уменьшенном разрешении)."""

    index: int
    cut_score: float
    faces: int
    bodies: int
    face_ratio: float  # высота самого большого лица / высота кадра
    body_ratio: float  # высота самой большой фигуры / высота кадра

    @property
    def people(self) -> int:
        return max(self.faces, self.bodies)

    @property
    def size_code(self) -> str:
        return classify_size(self.face_ratio, self.body_ratio)


@dataclass
class Keyframe:
    index: int
    time_sec: float
    timecode: str
    shot: int
    reason: str
    reason_ru: str
    people: int
    size_code: str
    size_ru: str
    filename: str = ""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def classify_size(face_ratio: float, body_ratio: float) -> str:
    """Определить крупность по размеру лица; если лиц нет — по размеру фигуры."""
    if face_ratio > 0:
        for threshold, code, _ in SHOT_SIZE_BY_FACE:
            if face_ratio > threshold:
                return code
        return "WS"
    if body_ratio > 0:
        # Фигура почти во весь кадр — средний план, иначе общий.
        return "MS" if body_ratio > 0.8 else "WS"
    return NO_FACE_SIZE[0]


def size_code_to_ru(code: str) -> str:
    for _, c, ru in SHOT_SIZE_BY_FACE:
        if c == code:
            return ru
    return NO_FACE_SIZE[1]


def format_timecode(seconds: float, sep: str = ":") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}{sep}{m:02d}{sep}{s:02d}.{ms:03d}"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Детекторы
# ---------------------------------------------------------------------------


MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/" + MODEL_NAME
)


def find_model(explicit: str | None) -> str:
    """Найти ONNX-модель YuNet: аргумент --model, папка models/ рядом со скриптом, иначе скачать."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [os.path.join(here, "models", MODEL_NAME), os.path.join(here, MODEL_NAME)]
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) > 10_000:
            return c
    target = candidates[-2]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    log(f"  модель лиц не найдена, скачиваю {MODEL_URL} → {target}")
    import urllib.request

    urllib.request.urlretrieve(MODEL_URL, target)
    return target


class Detectors:
    """Детектор лиц YuNet (cv2.FaceDetectorYN) и, если доступен, HOG-детектор фигур."""

    def __init__(self, use_bodies: bool, model_path: str | None = None, score_thr: float = 0.7):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Нужен OpenCV >= 4.5.4 с cv2.FaceDetectorYN (pip install -U opencv-python)")
        self.model_path = find_model(model_path)
        self.face = cv2.FaceDetectorYN.create(self.model_path, "", (320, 320), score_thr, 0.3, 5000)
        self.input_size: tuple[int, int] | None = None
        self.hog = None
        if use_bodies:
            if hasattr(cv2, "HOGDescriptor"):
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            else:
                log("  предупреждение: в этой сборке OpenCV нет HOG-детектора, --bodies игнорируется")

    def faces(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        if self.input_size != (w, h):
            self.face.setInputSize((w, h))
            self.input_size = (w, h)
        _, found = self.face.detect(bgr)
        if found is None:
            return []
        return [tuple(int(v) for v in row[:4]) for row in found]

    def bodies(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.hog is None:
            return []
        rects, weights = self.hog.detectMultiScale(
            bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        out = []
        for r, w in zip(rects, weights):
            if float(w) >= 0.5:
                out.append(tuple(int(v) for v in r))
        return merge_rects(out)


def merge_rects(rects: list[tuple[int, int, int, int]], iou_thr: float = 0.3):
    """Убрать сильно перекрывающиеся прямоугольники (дубли одной фигуры)."""
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for r in rects:
        if all(iou(r, k) < iou_thr for k in kept):
            kept.append(r)
    return kept


def iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Анализ видео (первый проход)
# ---------------------------------------------------------------------------


def frame_signature(bgr: np.ndarray):
    """Гистограмма HSV и размытая серая картинка — для оценки различия кадров."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)
    return hist, small


def cut_score(prev_sig, cur_sig) -> float:
    """0 — кадры одинаковые, ~1 — совсем разные."""
    hist_d = cv2.compareHist(prev_sig[0], cur_sig[0], cv2.HISTCMP_BHATTACHARYYA)
    pix_d = float(np.mean(np.abs(prev_sig[1] - cur_sig[1]))) / 255.0
    return 0.5 * float(hist_d) + 0.5 * min(1.0, pix_d * 4.0)


def analyze(path: str, detectors: Detectors, detect_every: int) -> tuple[list[FrameInfo], float, tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    infos: list[FrameInfo] = []
    prev_sig = None
    last_det = (0, 0, 0.0, 0.0)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scale = ANALYSIS_WIDTH / frame.shape[1]
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        fh = frame.shape[0]

        sig = frame_signature(frame)
        score = cut_score(prev_sig, sig) if prev_sig is not None else 0.0
        prev_sig = sig

        if idx % detect_every == 0:
            faces = detectors.faces(frame)
            bodies = detectors.bodies(frame)
            face_ratio = max((r[3] / fh for r in faces), default=0.0)
            body_ratio = max((r[3] / fh for r in bodies), default=0.0)
            last_det = (len(faces), len(bodies), face_ratio, body_ratio)

        infos.append(FrameInfo(idx, score, *last_det))
        idx += 1
        if total and idx % 50 == 0:
            log(f"  анализ: {idx}/{total} кадров ({100 * idx // total}%)")
    cap.release()
    log(f"  анализ завершён: {idx} кадров, {fps:.2f} к/с, {width}x{height}")
    return infos, fps, (width, height)


# ---------------------------------------------------------------------------
# Поиск склеек и изменений внутри плана
# ---------------------------------------------------------------------------


def detect_cuts(infos: list[FrameInfo], threshold: float, ratio: float, min_shot: int) -> list[int]:
    """Вернуть индексы кадров, с которых начинается новый план."""
    scores = np.array([f.cut_score for f in infos], dtype=np.float32)
    n = len(scores)
    cuts: list[int] = []
    win = 10
    for i in range(1, n):
        s = scores[i]
        if s < threshold:
            continue
        lo, hi = max(1, i - win), min(n, i + win + 1)
        neighbours = np.concatenate([scores[lo:i], scores[i + 1:hi]])
        local = float(np.median(neighbours)) if neighbours.size else 0.0
        if s < ratio * max(local, 0.02):
            continue  # быстрое движение/панорама: соседи тоже «шумят»
        if cuts and i - cuts[-1] < min_shot:
            # Две подряд "склейки" ближе минимальной длины плана — оставляем сильнейшую.
            if s > scores[cuts[-1]]:
                cuts[-1] = i
            continue
        if i < min_shot:
            continue
        cuts.append(i)
    return cuts


def detect_state_changes(
    infos: list[FrameInfo], shot_bounds: list[tuple[int, int]], stable: int
) -> list[tuple[int, str]]:
    """Внутри каждого плана найти кадры, где устойчиво меняется крупность или число людей.

    Состояние кадра = (число людей, код крупности). Новое состояние считается
    подтверждённым, когда оно преобладает (>= 70 %) в окне из `stable` кадров.
    Возвращает список (индекс кадра, причина).
    """
    events: list[tuple[int, str]] = []
    need = max(1, int(math.ceil(stable * 0.7)))
    for start, end in shot_bounds:
        window: deque[tuple[int, tuple[int, str]]] = deque(maxlen=stable)
        confirmed: tuple[int, str] | None = None
        for i in range(start, end):
            f = infos[i]
            window.append((i, (f.people, f.size_code)))
            if len(window) < min(stable, end - start):
                continue
            mode, cnt = Counter(st for _, st in window).most_common(1)[0]
            if cnt < need:
                continue
            if confirmed is None:
                confirmed = mode  # первое устойчивое состояние плана — событие не нужно
                continue
            if mode == confirmed:
                continue
            first_idx = next(j for j, st in window if st == mode)
            if mode[0] != confirmed[0]:
                reason = "count_change"
            else:
                reason = "size_change"
            events.append((first_idx, reason))
            confirmed = mode
            window.clear()
    return events


# ---------------------------------------------------------------------------
# Сборка списка ключевых кадров
# ---------------------------------------------------------------------------


def build_keyframes(
    infos: list[FrameInfo],
    fps: float,
    cuts: list[int],
    stable: int,
    interval_sec: float,
) -> list[Keyframe]:
    n = len(infos)
    bounds = []
    prev = 0
    for c in cuts:
        bounds.append((prev, c))
        prev = c
    bounds.append((prev, n))

    # index -> список причин (один кадр может быть и «после склейки», и «начало»)
    reasons: dict[int, list[str]] = {}

    def add(i: int, reason: str) -> None:
        if 0 <= i < n:
            reasons.setdefault(i, []).append(reason)

    add(0, "start")
    add(n - 1, "end")
    for c in cuts:
        add(c - 1, "cut_before")
        add(c, "cut_after")
    for i, reason in detect_state_changes(infos, bounds, stable):
        add(i, reason)
    if interval_sec > 0:
        step = max(1, int(round(interval_sec * fps)))
        for start, end in bounds:
            for i in range(start + step, end, step):
                add(i, "interval")

    shot_of = np.zeros(n, dtype=int)
    for shot_no, (start, end) in enumerate(bounds, start=1):
        shot_of[start:end] = shot_no

    keyframes = []
    for i in sorted(reasons):
        f = infos[i]
        reason = "+".join(dict.fromkeys(reasons[i]))
        reason_ru = ", ".join(REASON_RU[r] for r in dict.fromkeys(reasons[i]))
        t = i / fps
        keyframes.append(
            Keyframe(
                index=i,
                time_sec=round(t, 3),
                timecode=format_timecode(t),
                shot=int(shot_of[i]),
                reason=reason,
                reason_ru=reason_ru,
                people=f.people,
                size_code=f.size_code,
                size_ru=size_code_to_ru(f.size_code),
            )
        )
    return keyframes


# ---------------------------------------------------------------------------
# Сохранение кадров (второй проход) и отчётов
# ---------------------------------------------------------------------------


def save_frames(path: str, keyframes: list[Keyframe], out_dir: str, quality: int) -> None:
    wanted = {k.index: k for k in keyframes}
    cap = cv2.VideoCapture(path)
    idx = 0
    saved = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            k = wanted.pop(idx)
            k.filename = (
                f"{saved + 1:04d}_shot{k.shot:02d}_{format_timecode(k.time_sec, '-')}_{k.reason}.jpg"
            )
            cv2.imwrite(os.path.join(out_dir, k.filename), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved += 1
        idx += 1
    cap.release()
    if wanted:
        log(f"  предупреждение: не удалось прочитать {len(wanted)} кадров при сохранении")


def write_manifest(keyframes: list[Keyframe], out_dir: str, video: str, fps: float, size) -> None:
    rows = [asdict(k) for k in keyframes if k.filename]
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["index"], delimiter=";")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"video": os.path.abspath(video), "fps": fps, "width": size[0], "height": size[1], "keyframes": rows},
            fh,
            ensure_ascii=False,
            indent=2,
        )


def contact_sheet(keyframes: list[Keyframe], out_dir: str, cols: int = 4, thumb_w: int = 320) -> str | None:
    frames = [k for k in keyframes if k.filename]
    if not frames:
        return None
    thumbs = []
    for k in frames:
        img = cv2.imread(os.path.join(out_dir, k.filename))
        if img is None:
            continue
        scale = thumb_w / img.shape[1]
        img = cv2.resize(img, (thumb_w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        label_h = 34
        canvas = np.full((img.shape[0] + label_h, thumb_w, 3), 24, dtype=np.uint8)
        canvas[:img.shape[0]] = img
        # cv2.putText не умеет кириллицу — подписи латиницей.
        line1 = f"#{k.shot} {k.timecode} {k.size_code} x{k.people}"
        line2 = k.reason
        cv2.putText(canvas, line1, (4, img.shape[0] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2, (4, img.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 220, 255), 1, cv2.LINE_AA)
        thumbs.append(canvas)
    if not thumbs:
        return None
    h = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)) for t in thumbs]
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(thumbs[0], 24))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out = os.path.join(out_dir, "contact_sheet.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Раскадровка видео: сохраняет ключевые кадры (склейки, смена крупности, число героев) в JPG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", help="путь к видеофайлу")
    p.add_argument("-o", "--out", help="папка для кадров (по умолчанию <имя видео>_storyboard рядом с видео)")
    p.add_argument("--cut-threshold", type=float, default=0.30, help="порог различия кадров для склейки (0..1)")
    p.add_argument("--cut-ratio", type=float, default=3.0, help="во сколько раз различие должно превышать соседние кадры")
    p.add_argument("--min-shot", type=int, default=6, help="минимальная длина плана в кадрах")
    p.add_argument("--stable", type=int, default=12, help="сколько кадров подряд состояние должно держаться, чтобы считаться сменой")
    p.add_argument("--detect-every", type=int, default=1, help="искать лица не в каждом кадре, а в каждом N-м (ускорение)")
    p.add_argument("--model", help="путь к ONNX-модели YuNet (по умолчанию models/ рядом со скриптом; при отсутствии скачивается)")
    p.add_argument("--face-score", type=float, default=0.7, help="порог уверенности детектора лиц (0..1)")
    p.add_argument("--bodies", action="store_true", help="дополнительно считать фигуры людей (HOG), медленнее")
    p.add_argument("--interval", type=float, default=0.0, help="дополнительно сохранять кадр каждые N секунд внутри плана (0 — выкл.)")
    p.add_argument("--quality", type=int, default=92, help="качество JPG (1..100)")
    p.add_argument("--contact-sheet", action="store_true", help="собрать общую картинку-простыню из миниатюр")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    video = args.video
    if not os.path.isfile(video):
        log(f"Файл не найден: {video}")
        return 1
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(video)),
        os.path.splitext(os.path.basename(video))[0] + "_storyboard",
    )
    os.makedirs(out_dir, exist_ok=True)

    log(f"Видео: {video}")
    log("1/3 Анализ кадров…")
    detectors = Detectors(use_bodies=args.bodies, model_path=args.model, score_thr=args.face_score)
    infos, fps, size = analyze(video, detectors, max(1, args.detect_every))
    if not infos:
        log("В видео нет кадров.")
        return 1

    log("2/3 Поиск склеек и изменений…")
    cuts = detect_cuts(infos, args.cut_threshold, args.cut_ratio, args.min_shot)
    keyframes = build_keyframes(infos, fps, cuts, args.stable, args.interval)
    log(f"  планов: {len(cuts) + 1}, ключевых кадров: {len(keyframes)}")

    log(f"3/3 Сохранение кадров в {out_dir}…")
    save_frames(video, keyframes, out_dir, args.quality)
    write_manifest(keyframes, out_dir, video, fps, size)
    if args.contact_sheet:
        sheet = contact_sheet(keyframes, out_dir)
        if sheet:
            log(f"  простыня: {sheet}")

    print(f"{'кадр':>6}  {'время':>12}  {'план':>4}  {'люди':>4}  {'крупность':<9}  причина")
    for k in keyframes:
        if k.filename:
            print(f"{k.index:>6}  {k.timecode:>12}  {k.shot:>4}  {k.people:>4}  {k.size_ru:<9}  {k.reason_ru}")
    print(f"\nСохранено {sum(1 for k in keyframes if k.filename)} кадров в {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
