#!/usr/bin/env python3
"""
cinema_shot — раскадровка видео по ключевым кадрам.

Программа проходит по видео и сохраняет в отдельную папку JPG-кадры в тех местах,
где что-то меняется:

  * склейка (монтажный стык) — сохраняется последний кадр до склейки и первый после;
  * смена крупности внутри плана (деталь / крупный / средний / общий) — по размеру
    лица в кадре;
  * изменение состава героев в кадре — люди и животные, найденные YOLOX (или лица);
    мелкие фигуры на фоне считаются массовкой и в «героев» не входят.

Дополнительно сохраняются первый и последний кадры видео, а по желанию — кадр
раз в N секунд внутри длинных статичных планов.

Рядом с кадрами кладутся manifest.csv / manifest.json с описанием каждого кадра
и, по желанию, contact_sheet.jpg — общая «простыня» из миниатюр.

Зависимости: opencv-python (или opencv-python-headless) >= 4.5.4, numpy.
Модели (папка models/, скачиваются автоматически, если их нет):
    face_detection_yunet_2023mar.onnx   — лица (YuNet, ~230 КБ)
    object_detection_yolox_2022nov.onnx — люди (YOLOX-S, ~35 МБ)

Пример:
    python cinema_shot.py clip.mp4
    python cinema_shot.py clip.mp4 -o storyboard --contact-sheet
    python cinema_shot.py clip.mp4 --cut-threshold 0.25 --stable 8 --no-persons
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
NO_FACE_SIZE = ("NF", "без героев")

# Классы COCO, которые считаем «героями»: люди и животные.
HERO_CLASSES = {
    0: "человек", 14: "птица", 15: "кошка", 16: "собака", 17: "лошадь", 18: "овца",
    19: "корова", 20: "слон", 21: "медведь", 22: "зебра", 23: "жираф",
}

REASON_RU = {
    "start": "начало видео",
    "end": "конец видео",
    "cut_before": "до склейки",
    "cut_after": "после склейки",
    "size_change": "смена крупности",
    "count_change": "изменение состава героев",
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
    faces: int          # найдено лиц
    heroes: int         # «герои» — крупные фигуры (или лица, если детектор людей выключен)
    people_total: int   # все люди в кадре, включая массовку
    face_ratio: float   # высота самого большого лица / высота кадра
    body_ratio: float   # высота самой большой фигуры-героя / высота кадра
    body_full: bool     # фигура-герой видна целиком (не обрезана нижним краем кадра)
    composition: str    # кто в кадре, например «1 человек, 1 собака»

    @property
    def people(self) -> int:
        return self.heroes

    @property
    def size_code(self) -> str:
        return classify_size(self.face_ratio, self.body_ratio, self.body_full)


@dataclass
class Keyframe:
    index: int
    time_sec: float
    timecode: str
    shot: int
    reason: str
    reason_ru: str
    people: int
    people_total: int
    composition: str
    size_code: str
    size_ru: str
    filename: str = ""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def classify_size(face_ratio: float, body_ratio: float, body_full: bool = False) -> str:
    """Определить крупность.

    Если лицо достаточно крупное (средний план и ближе) — по лицу. Иначе — по
    фигуре-герою: фигура видна целиком — общий план; фигура обрезана кадром и
    занимает почти всю высоту — средний. Мелкое лицо без фигуры — общий;
    ничего нет — «без героев».
    """
    if face_ratio > SHOT_SIZE_BY_FACE[-2][0]:
        for threshold, code, _ in SHOT_SIZE_BY_FACE:
            if face_ratio > threshold:
                return code
    if body_ratio > 0:
        return "MS" if body_ratio > 0.85 and not body_full else "WS"
    if face_ratio > 0:
        return "WS"
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


# OpenCV на Windows не открывает файлы по путям с кириллицей, поэтому чтение и
# запись картинок/моделей идут через Python, а OpenCV получает только байты.


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def imread_u(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_u(path: str, img: np.ndarray, quality: int = 92) -> bool:
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)
    return bool(ok)


def open_video(path: str) -> tuple[cv2.VideoCapture, str]:
    """Открыть видео; если OpenCV не справляется с путём (кириллица на Windows) —
    скопировать файл во временную папку с латинским именем и открыть копию."""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, path
    cap.release()
    import shutil
    import tempfile

    tmp_dir = tempfile.gettempdir()
    if not tmp_dir.isascii() and os.name == "nt":
        for candidate in (r"C:\Temp", r"C:\Windows\Temp"):
            try:
                os.makedirs(candidate, exist_ok=True)
                tmp_dir = candidate
                break
            except OSError:
                continue
    tmp = os.path.join(tmp_dir, "cinema_shot_input" + os.path.splitext(path)[1].lower())
    log(f"  OpenCV не открыл видео по этому пути, копирую во временный файл {tmp}")
    shutil.copyfile(path, tmp)
    cap = cv2.VideoCapture(tmp)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {path}")
    return cap, tmp


# ---------------------------------------------------------------------------
# Детекторы
# ---------------------------------------------------------------------------


FACE_MODEL = "face_detection_yunet_2023mar.onnx"
PERSON_MODEL = "object_detection_yolox_2022nov.onnx"
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODEL_URLS = {
    FACE_MODEL: ZOO + "face_detection_yunet/" + FACE_MODEL,
    PERSON_MODEL: ZOO + "object_detection_yolox/" + PERSON_MODEL,
}


def find_model(name: str, explicit: str | None) -> str:
    """Найти модель: явный путь, папка models/ рядом со скриптом, иначе скачать из opencv_zoo."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [c for c in (explicit, os.path.join(here, "models", name), os.path.join(here, name)) if c]
    for c in candidates:
        if os.path.isfile(c) and os.path.getsize(c) > 10_000:
            return c
    target = os.path.join(here, "models", name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    log(f"  модель {name} не найдена, скачиваю {MODEL_URLS[name]}")
    import urllib.request

    urllib.request.urlretrieve(MODEL_URLS[name], target)
    return target


class PersonDetector:
    """YOLOX-S (COCO) через cv2.dnn — находит людей (в том числе спиной к камере) и животных."""

    INPUT = 640
    STRIDES = (8, 16, 32)

    def __init__(self, model_path: str, conf: float = 0.4, nms: float = 0.5):
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(read_bytes(model_path), np.uint8))
        self.conf, self.nms = conf, nms
        grids, strides = [], []
        for st in self.STRIDES:
            n = self.INPUT // st
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), st))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        """Вернуть (x, y, w, h, класс COCO) для людей и животных из HERO_CLASSES."""
        h, w = bgr.shape[:2]
        r = min(self.INPUT / h, self.INPUT / w)
        resized = cv2.resize(bgr, (int(w * r), int(h * r)))
        canvas = np.full((self.INPUT, self.INPUT, 3), 114, np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]
        self.net.setInput(blob)
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        cls_ids = np.array(sorted(HERO_CLASSES), dtype=int)
        cls_scores = out[:, 4:5] * out[:, 5 + cls_ids]  # objectness * вероятность класса
        best = cls_scores.argmax(axis=1)
        scores = cls_scores[np.arange(len(best)), best]
        keep = scores >= self.conf
        if not keep.any():
            return []
        boxes = np.concatenate([xy[keep] - wh[keep] / 2, wh[keep]], axis=1) / r
        classes = cls_ids[best[keep]]
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores[keep].tolist(), classes.tolist(), self.conf, self.nms)
        if len(idx) == 0:
            return []
        return [tuple(int(v) for v in boxes[int(i)]) + (int(classes[int(i)]),) for i in np.asarray(idx).flatten()]


class Detectors:
    """Детектор лиц YuNet (cv2.FaceDetectorYN) и детектор людей YOLOX (cv2.dnn)."""

    def __init__(
        self,
        use_persons: bool,
        face_model: str | None = None,
        person_model: str | None = None,
        score_thr: float = 0.7,
    ):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("Нужен OpenCV >= 4.5.4 с cv2.FaceDetectorYN (pip install -U opencv-python)")
        model_bytes = read_bytes(find_model(FACE_MODEL, face_model))
        self.face = cv2.FaceDetectorYN.create("onnx", model_bytes, b"", (320, 320), score_thr, 0.3, 5000)
        self.input_size: tuple[int, int] | None = None
        self.persons: PersonDetector | None = None
        if use_persons:
            try:
                self.persons = PersonDetector(find_model(PERSON_MODEL, person_model))
            except Exception as exc:  # noqa: BLE001 — нет сети/модели: работаем только по лицам
                log(f"  предупреждение: детектор людей недоступен ({exc}), считаю героев по лицам")

    def faces(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        if self.input_size != (w, h):
            self.face.setInputSize((w, h))
            self.input_size = (w, h)
        _, found = self.face.detect(bgr)
        if found is None:
            return []
        return [tuple(int(v) for v in row[:4]) for row in found]

    def bodies(self, bgr: np.ndarray) -> list[tuple[int, int, int, int, int]]:
        return self.persons.detect(bgr) if self.persons else []


def split_heroes(bodies: list, frame_h: int, min_height: float, rel_area: float) -> list:
    """Отделить героев от массовки: фигура достаточно высокая и по площади сопоставима с самой крупной."""
    if not bodies:
        return []
    largest = max(b[2] * b[3] for b in bodies)
    return [b for b in bodies if b[3] >= min_height * frame_h and b[2] * b[3] >= rel_area * largest]


def describe(bodies: list) -> str:
    """«1 человек, 1 собака» по списку фигур с классом COCO в последнем элементе."""
    counts = Counter(HERO_CLASSES.get(b[-1], "человек") for b in bodies)
    order = sorted(counts, key=lambda k: (k != "человек", k))
    return ", ".join(f"{counts[k]} {k}" for k in order)


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


def analyze(
    path: str,
    detectors: Detectors,
    detect_every: int,
    hero_min_height: float,
    hero_rel_area: float,
    likely_cut: float = 0.2,
) -> tuple[list[FrameInfo], float, tuple[int, int], str]:
    cap, opened_path = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    infos: list[FrameInfo] = []
    prev_sig = None
    last_det = (0, 0, 0, 0.0, 0.0, False, "")
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

        # Детекция в каждом N-м кадре, а также сразу после вероятной склейки,
        # чтобы кадр «после склейки» не унаследовал результаты предыдущего плана.
        if idx % detect_every == 0 or score >= likely_cut:
            faces = detectors.faces(frame)
            face_ratio = max((r[3] / fh for r in faces), default=0.0)
            body_full = False
            if detectors.persons:
                bodies = detectors.bodies(frame)
                heroes = split_heroes(bodies, fh, hero_min_height, hero_rel_area)
                body_ratio = max((min(1.0, r[3] / fh) for r in heroes), default=0.0)
                if heroes:
                    largest = max(heroes, key=lambda r: r[3])
                    body_full = largest[1] + largest[3] < fh * 0.97  # ноги не срезаны кадром
                n_heroes, n_total, comp = len(heroes), len(bodies), describe(heroes)
            else:
                # Без детектора людей: герой — лицо, сопоставимое по размеру с самым крупным.
                hero_faces = [r for r in faces if r[3] >= 0.35 * face_ratio * fh]
                n_heroes, n_total, body_ratio = len(hero_faces), len(faces), 0.0
                face_ratio = max((r[3] / fh for r in hero_faces), default=0.0)
                comp = f"{n_heroes} человек" if n_heroes else ""
            last_det = (len(faces), n_heroes, n_total, face_ratio, body_ratio, body_full, comp)

        infos.append(FrameInfo(idx, score, *last_det))
        idx += 1
        if total and idx % 50 == 0:
            log(f"  анализ: {idx}/{total} кадров ({100 * idx // total}%)")
    cap.release()
    log(f"  анализ завершён: {idx} кадров, {fps:.2f} к/с, {width}x{height}")
    return infos, fps, (width, height), opened_path


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

    Состояние кадра = (состав героев, код крупности). Новое состояние считается
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
            window.append((i, (f.composition, f.size_code)))
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
                people_total=f.people_total,
                composition=f.composition,
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
    cap, _ = open_video(path)
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
            if not imwrite_u(os.path.join(out_dir, k.filename), frame, quality):
                log(f"  предупреждение: не удалось записать {k.filename}")
                k.filename = ""
                continue
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
        img = imread_u(os.path.join(out_dir, k.filename))
        if img is None:
            continue
        scale = thumb_w / img.shape[1]
        img = cv2.resize(img, (thumb_w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        label_h = 34
        canvas = np.full((img.shape[0] + label_h, thumb_w, 3), 24, dtype=np.uint8)
        canvas[:img.shape[0]] = img
        # cv2.putText не умеет кириллицу — подписи латиницей.
        line1 = f"#{k.shot} {k.timecode} {k.size_code} heroes:{k.people} all:{k.people_total}"
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
    imwrite_u(out, sheet, 90)
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
    p.add_argument("--detect-every", type=int, default=2, help="искать людей и лица не в каждом кадре, а в каждом N-м (ускорение)")
    p.add_argument("--no-persons", action="store_true", help="не использовать детектор людей YOLOX, считать героев только по лицам (быстрее)")
    p.add_argument("--hero-min-height", type=float, default=0.2, help="фигура ниже этой доли высоты кадра — массовка")
    p.add_argument("--hero-rel-area", type=float, default=0.3, help="фигура с площадью меньше этой доли от самой крупной — массовка")
    p.add_argument("--face-model", help="путь к ONNX-модели YuNet (по умолчанию models/ рядом со скриптом; при отсутствии скачивается)")
    p.add_argument("--person-model", help="путь к ONNX-модели YOLOX (по умолчанию models/ рядом со скриптом; при отсутствии скачивается, ~35 МБ)")
    p.add_argument("--face-score", type=float, default=0.7, help="порог уверенности детектора лиц (0..1)")
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
    detectors = Detectors(
        use_persons=not args.no_persons,
        face_model=args.face_model,
        person_model=args.person_model,
        score_thr=args.face_score,
    )
    infos, fps, size, opened_path = analyze(
        video, detectors, max(1, args.detect_every), args.hero_min_height, args.hero_rel_area
    )
    if not infos:
        log("В видео нет кадров.")
        return 1

    log("2/3 Поиск склеек и изменений…")
    cuts = detect_cuts(infos, args.cut_threshold, args.cut_ratio, args.min_shot)
    keyframes = build_keyframes(infos, fps, cuts, args.stable, args.interval)
    log(f"  планов: {len(cuts) + 1}, ключевых кадров: {len(keyframes)}")

    log(f"3/3 Сохранение кадров в {out_dir}…")
    save_frames(opened_path, keyframes, out_dir, args.quality)
    if opened_path != video:
        try:
            os.remove(opened_path)
        except OSError:
            pass
    write_manifest(keyframes, out_dir, video, fps, size)
    if args.contact_sheet:
        sheet = contact_sheet(keyframes, out_dir)
        if sheet:
            log(f"  простыня: {sheet}")

    print(f"{'кадр':>6}  {'время':>12}  {'план':>4}  {'всего':>5}  {'крупность':<11}  {'герои':<22}  причина")
    for k in keyframes:
        if k.filename:
            print(
                f"{k.index:>6}  {k.timecode:>12}  {k.shot:>4}  {k.people_total:>5}  "
                f"{k.size_ru:<11}  {k.composition or '—':<22}  {k.reason_ru}"
            )
    print(f"\nСохранено {sum(1 for k in keyframes if k.filename)} кадров в {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
