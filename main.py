import json
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from models.llm import describe_scene_from_frame
from models.yolo import classify_hazard, detect_image_bytes, format_hazard_alert

RUNTIME_CONFIG_YAML_PATH = Path(__file__).resolve().parent / "config" / "runtime_config.yaml"
RUNTIME_CONFIG_JSON_PATH = Path(__file__).resolve().parent / "config" / "runtime_config.json"


def _load_json_with_comments(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] != "\n":
                i += 1
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        out.append(ch)
        i += 1

    return json.loads("".join(out))


def load_runtime_config() -> dict[str, Any]:
    if RUNTIME_CONFIG_YAML_PATH.exists():
        data = yaml.safe_load(RUNTIME_CONFIG_YAML_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    if RUNTIME_CONFIG_JSON_PATH.exists():
        return _load_json_with_comments(RUNTIME_CONFIG_JSON_PATH)
    return {
        "pipeline": {"yolo_target_fps": 6.0, "danger_threshold": 0.85},
        "llm": {
            "max_new_tokens": 80,
            "calls_per_second_constant": 0.8,
            "max_calls_per_second": 1.5,
            "min_interval_ms": 300,
            "probability_scale": 1.0,
            "base_probability_bias": 0.1,
            "similarity_block_enabled": True,
            "similarity_threshold": 0.9,
            "rng_seed": 42,
        },
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _smooth_positive_to_unit(x: float, tau: float = 0.35) -> float:
    """
    Smoothly map x>=0 to [0,1):
      f(x) = x / (x + tau), tau>0
    Compared to hard clamp, this avoids sharp saturation and keeps continuity.
    """
    x = max(0.0, float(x))
    tau = max(1e-6, float(tau))
    return x / (x + tau)


def preprocess_image(img_bytes: bytes, target_size: int = 256) -> bytes:
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _build_detection_signature(detections: list[dict[str, Any]]) -> dict[str, float]:
    """Map class_name -> max confidence for similarity comparison."""
    sig: dict[str, float] = {}
    for det in detections:
        name = str(det.get("class_name", "")).lower()
        conf = float(det.get("confidence", 0.0))
        if not name:
            continue
        if name not in sig or conf > sig[name]:
            sig[name] = conf
    return sig


def _signature_similarity(sig_a: dict[str, float], sig_b: dict[str, float]) -> float:
    """
    Weighted Jaccard-like similarity on class-confidence signatures:
    sum(min(conf))/sum(max(conf)).
    """
    keys = set(sig_a) | set(sig_b)
    if not keys:
        return 1.0
    num = 0.0
    den = 0.0
    for k in keys:
        a = sig_a.get(k, 0.0)
        b = sig_b.get(k, 0.0)
        num += min(a, b)
        den += max(a, b)
    if den <= 1e-8:
        return 1.0
    return _clamp01(num / den)


def _llm_worker_process(
    llm_cfg: dict[str, Any],
    in_queue: "mp.Queue[dict[str, Any] | None]",
    out_queue: "mp.Queue[dict[str, Any]]",
) -> None:
    while True:
        task = in_queue.get()
        if task is None:
            break
        frame_id = task["frame_id"]
        try:
            start = time.perf_counter()
            desc = describe_scene_from_frame(
                task["processed_bytes"],
                task["detections"],
                max_new_tokens=int(llm_cfg.get("max_new_tokens", 80)),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            out_queue.put(
                {
                    "frame_id": frame_id,
                    "description": desc,
                    "error": None,
                    "llm_ms": round(elapsed_ms, 1),
                }
            )
        except Exception as e:  # noqa: BLE001
            out_queue.put(
                {
                    "frame_id": frame_id,
                    "description": None,
                    "error": str(e),
                    "llm_ms": 0.0,
                }
            )


class FrameSequencePipeline:
    """
    Two-queue pipeline:
    1) YOLO queue (main thread): always runs frame-by-frame without waiting for LLM.
    2) LLM queue (worker thread): only sent=True frames are queued and dequeued on completion.
    """

    def __init__(
        self,
        *,
        context: str = "walking",
        use_depth: bool = True,
        preprocess_size: int = 256,
        imgsz: int = 256,
        return_annotated: bool = True,
        llm_worker_mode: str = "thread",
    ) -> None:
        self.cfg = load_runtime_config()
        self.context = context
        self.use_depth = use_depth
        self.preprocess_size = preprocess_size
        self.imgsz = imgsz
        self.return_annotated = return_annotated
        self.llm_worker_mode = llm_worker_mode

        self.llm_cfg = self.cfg.get("llm", {})
        self._rng = random.Random(int(self.llm_cfg.get("rng_seed", 42)))
        self._rng_lock = threading.Lock()

        self._last_llm_call_ts = 0.0
        self._last_llm_lock = threading.Lock()

        self._llm_process: mp.Process | None = None
        if self.llm_worker_mode == "process":
            self._llm_in = mp.Queue()
            self._llm_out = mp.Queue()
            self._llm_process = mp.Process(
                target=_llm_worker_process,
                args=(self.llm_cfg, self._llm_in, self._llm_out),
                daemon=True,
            )
            self._llm_process.start()
            self._worker: threading.Thread | None = None
        else:
            self._llm_in = queue.Queue()
            self._llm_out = queue.Queue()
            self._worker = threading.Thread(target=self._llm_worker, daemon=True)
            self._worker.start()

        self._results_by_id: dict[int, dict[str, Any]] = {}
        self._ordered_ids: list[int] = []
        self._next_frame_id = 0
        self._danger_active = False
        self._prev_signature: dict[str, float] | None = None
        self._llm_pending = 0
        self._llm_pending_lock = threading.Lock()

    def close(self) -> None:
        self._llm_in.put(None)
        if self._llm_process is not None:
            self._llm_process.join(timeout=5)
            return
        if self._worker is not None:
            self._worker.join(timeout=5)

    def llm_send_probability(self, yolo_score: float) -> float:
        """
        Bias outside scale:
        P = bias + (1 - bias) * smooth(yolo_score * constant * probability_scale)
        """
        constant = float(self.llm_cfg.get("calls_per_second_constant", 0.8))
        score_scale = float(self.llm_cfg.get("probability_scale", 1.0))
        bias = _clamp01(float(self.llm_cfg.get("base_probability_bias", 0.1)))
        tau = float(self.llm_cfg.get("probability_smoothing_tau", 0.35))
        scaled_input = float(yolo_score) * constant * score_scale
        scaled_part = _smooth_positive_to_unit(scaled_input, tau=tau)
        return bias + (1.0 - bias) * scaled_part

    def _rate_allowed(self) -> bool:
        max_calls_per_sec = float(self.llm_cfg.get("max_calls_per_second", 1.5))
        min_interval_sec = float(self.llm_cfg.get("min_interval_ms", 300)) / 1000.0
        per_call_interval = 1.0 / max_calls_per_sec if max_calls_per_sec > 0 else 1.0
        now = time.monotonic()

        with self._last_llm_lock:
            if self._last_llm_call_ts and (now - self._last_llm_call_ts) < max(min_interval_sec, per_call_interval):
                return False
            self._last_llm_call_ts = now
            return True

    def should_send_to_llm(self, yolo_score: float) -> tuple[bool, float, bool, bool]:
        probability = self.llm_send_probability(yolo_score)
        with self._rng_lock:
            sampled = self._rng.random() < probability
        rate_allowed = self._rate_allowed() if sampled else False
        return sampled and rate_allowed, probability, sampled, rate_allowed

    def _llm_worker(self) -> None:
        while True:
            task = self._llm_in.get()
            if task is None:
                self._llm_in.task_done()
                break

            frame_id = task["frame_id"]
            try:
                start = time.perf_counter()
                desc = describe_scene_from_frame(
                    task["processed_bytes"],
                    task["detections"],
                    max_new_tokens=int(self.llm_cfg.get("max_new_tokens", 80)),
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._llm_out.put(
                    {
                        "frame_id": frame_id,
                        "description": desc,
                        "error": None,
                        "llm_ms": round(elapsed_ms, 1),
                    }
                )
            except Exception as e:  # noqa: BLE001
                self._llm_out.put(
                    {
                        "frame_id": frame_id,
                        "description": None,
                        "error": str(e),
                        "llm_ms": 0.0,
                    }
                )
            finally:
                self._llm_in.task_done()

    def _drain_llm_out(self) -> int:
        updated = 0
        while True:
            try:
                out = self._llm_out.get_nowait()
            except queue.Empty:
                break

            frame_id = out["frame_id"]
            result = self._results_by_id.get(frame_id)
            if result is not None:
                result["llm"]["queued"] = False
                result["llm"]["done"] = True
                result["llm"]["description"] = out["description"]
                result["llm"]["error"] = out["error"]
                result["latency"]["llm_ms"] = out["llm_ms"]
                updated += 1
                with self._llm_pending_lock:
                    self._llm_pending = max(0, self._llm_pending - 1)
            if self._llm_process is None:
                self._llm_out.task_done()
        return updated

    def submit_frames(self, frames: list[bytes]) -> list[dict[str, Any]]:
        for frame_bytes in frames:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
            self._ordered_ids.append(frame_id)

            total_start = time.perf_counter()

            preprocess_start = time.perf_counter()
            processed_bytes = preprocess_image(frame_bytes, target_size=self.preprocess_size)
            preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

            yolo_start = time.perf_counter()
            yolo_result = detect_image_bytes(
                processed_bytes,
                return_image=self.return_annotated,
                imgsz=self.imgsz,
                use_depth=self.use_depth,
            )
            yolo_ms = (time.perf_counter() - yolo_start) * 1000
            detections = yolo_result["detections"]

            hazard_start = time.perf_counter()
            hazard = classify_hazard(detections, context=self.context)
            hazard_ms = (time.perf_counter() - hazard_start) * 1000

            yolo_score = float(hazard.get("hazard_score", 0.0))
            danger_threshold = float(self.cfg.get("pipeline", {}).get("danger_threshold", 0.85))
            is_danger_now = yolo_score >= danger_threshold
            just_entered_danger = is_danger_now and not self._danger_active
            just_exited_danger = (not is_danger_now) and self._danger_active
            self._danger_active = is_danger_now

            current_sig = _build_detection_signature(detections)
            similarity_to_prev = None
            similarity_blocked = False
            if self._prev_signature is not None:
                similarity_to_prev = _signature_similarity(current_sig, self._prev_signature)
                sim_enabled = bool(self.llm_cfg.get("similarity_block_enabled", True))
                sim_threshold = float(self.llm_cfg.get("similarity_threshold", 0.9))
                similarity_blocked = sim_enabled and (similarity_to_prev >= sim_threshold)
            self._prev_signature = current_sig

            if similarity_blocked:
                llm_prob = self.llm_send_probability(yolo_score)
                send_to_llm, llm_sampled, llm_rate_allowed = False, False, False
            else:
                send_to_llm, llm_prob, llm_sampled, llm_rate_allowed = self.should_send_to_llm(yolo_score)
            queued = False
            if send_to_llm:
                self._llm_in.put(
                    {
                        "frame_id": frame_id,
                        "processed_bytes": processed_bytes,
                        "detections": detections,
                    }
                )
                with self._llm_pending_lock:
                    self._llm_pending += 1
                queued = True

            total_ms = (time.perf_counter() - total_start) * 1000
            result = {
                "frame_id": frame_id,
                "detections": detections,
                "image_base64": yolo_result.get("image_base64"),
                "hazard": hazard,
                "hazard_alert": format_hazard_alert(hazard),
                "danger": {
                    "threshold": round(danger_threshold, 4),
                    "is_danger": is_danger_now,
                    "active": self._danger_active,
                    "just_entered": just_entered_danger,
                    "just_exited": just_exited_danger,
                    "immediate_alert": "DANGER: frame marked dangerous." if just_entered_danger else None,
                },
                "llm": {
                    "sent": send_to_llm,
                    "queued": queued,
                    "done": False if queued else True,
                    "probability": round(llm_prob, 4),
                    "sampled": llm_sampled,
                    "rate_allowed": llm_rate_allowed,
                    "similarity_to_prev": round(float(similarity_to_prev), 4) if similarity_to_prev is not None else None,
                    "similarity_blocked": similarity_blocked,
                    "description": None,
                    "error": None,
                },
                "latency": {
                    "preprocess_ms": round(preprocess_ms, 1),
                    "yolo_ms": round(yolo_ms, 1),
                    "depth_ms": round(float(yolo_result.get("depth_ms", 0.0)), 1),
                    "hazard_ms": round(hazard_ms, 1),
                    "llm_ms": 0.0,
                    "total_ms": round(total_ms, 1),
                },
                "runtime": {
                    "yolo_pid": os.getpid(),
                    "llm_pid": self._llm_process.pid if self._llm_process is not None else os.getpid(),
                    "llm_worker_mode": self.llm_worker_mode,
                },
            }
            self._results_by_id[frame_id] = result
            self._drain_llm_out()

        return self.get_results()

    def wait_for_all_llm(self) -> None:
        if self._llm_process is not None:
            while True:
                self._drain_llm_out()
                with self._llm_pending_lock:
                    if self._llm_pending == 0:
                        break
                time.sleep(0.01)
            return
        self._llm_in.join()
        self._drain_llm_out()

    def poll_llm(self) -> int:
        """Drain completed LLM results without blocking."""
        return self._drain_llm_out()

    def get_queue_status(self) -> dict[str, int]:
        """Return current queue/throughput counters for terminal monitoring."""
        self._drain_llm_out()
        results = self.get_results()
        llm_sent = 0
        llm_done = 0
        llm_queued = 0
        for r in results:
            llm = r.get("llm", {})
            if llm.get("sent"):
                llm_sent += 1
            if llm.get("done"):
                llm_done += 1
            if llm.get("queued"):
                llm_queued += 1
        with self._llm_pending_lock:
            pending = int(self._llm_pending)
        return {
            "total_frames": len(results),
            "yolo_queue": 0,
            "llm_sent": llm_sent,
            "llm_done": llm_done,
            "llm_queued": llm_queued,
            "llm_pending": pending,
        }

    def get_results(self) -> list[dict[str, Any]]:
        return [self._results_by_id[i] for i in self._ordered_ids]


def run_frame_sequence(
    frames: list[bytes],
    *,
    context: str = "walking",
    use_depth: bool = True,
    preprocess_size: int = 256,
    imgsz: int = 256,
    return_annotated: bool = True,
    wait_for_llm: bool = True,
    llm_worker_mode: str = "thread",
) -> list[dict[str, Any]]:
    pipeline = FrameSequencePipeline(
        context=context,
        use_depth=use_depth,
        preprocess_size=preprocess_size,
        imgsz=imgsz,
        return_annotated=return_annotated,
        llm_worker_mode=llm_worker_mode,
    )
    try:
        results = pipeline.submit_frames(frames)
        if wait_for_llm:
            pipeline.wait_for_all_llm()
            results = pipeline.get_results()
        return results
    finally:
        pipeline.close()


def run_image_pipeline(
    image_bytes: bytes,
    *,
    context: str = "walking",
    use_depth: bool = True,
    preprocess_size: int = 256,
    imgsz: int = 256,
    return_annotated: bool = True,
    wait_for_llm: bool = True,
    llm_worker_mode: str = "thread",
) -> dict[str, Any]:
    """Backward-compatible single-image wrapper."""
    results = run_frame_sequence(
        [image_bytes],
        context=context,
        use_depth=use_depth,
        preprocess_size=preprocess_size,
        imgsz=imgsz,
        return_annotated=return_annotated,
        wait_for_llm=wait_for_llm,
        llm_worker_mode=llm_worker_mode,
    )
    return results[0]
