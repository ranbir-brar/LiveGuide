import multiprocessing as mp
import os
import queue
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from models.llm import describe_scene_from_frame
from models.yolo import detect_image_bytes

RUNTIME_CONFIG_YAML_PATH = Path(__file__).resolve().parent / "config" / "runtime_config.yaml"


def load_runtime_config() -> dict[str, Any]:
    if RUNTIME_CONFIG_YAML_PATH.exists():
        data = yaml.safe_load(RUNTIME_CONFIG_YAML_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {
        "pipeline": {"yolo_target_fps": 6.0},
        "llm": {
            "max_new_tokens": 80,
            "llm_call_interval_sec": 2.5,
            "similarity_threshold": 0.85,
            "similarity_min_confidence": 0.25,
        },
    }


def preprocess_image(img_bytes: bytes, target_size: int = 256) -> bytes:
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    resized = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


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
                provider=task.get("llm_provider"),
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
        context: str = "general",
        use_depth: bool = True,
        preprocess_size: int = 256,
        imgsz: int = 256,
        return_annotated: bool = True,
        llm_worker_mode: str = "thread",
        max_results: int = 300,
    ) -> None:
        self.cfg = load_runtime_config()
        self.context = context
        self.use_depth = use_depth
        self.preprocess_size = preprocess_size
        self.imgsz = imgsz
        self.return_annotated = return_annotated
        self.llm_worker_mode = llm_worker_mode
        self.max_results = max(0, int(max_results))

        self.llm_cfg = self.cfg.get("llm", {})
        self.llm_provider = str(self.llm_cfg.get("provider", "qwen_local")).strip().lower()
        self._detection_sensitivity = 5.0

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
        self._llm_pending = 0
        self._llm_pending_lock = threading.Lock()
        self._prev_detection_signature: set[str] | None = None

    def _prune_results(self) -> None:
        if self.max_results <= 0:
            return
        overflow = len(self._ordered_ids) - self.max_results
        if overflow <= 0:
            return
        stale_ids = self._ordered_ids[:overflow]
        self._ordered_ids = self._ordered_ids[overflow:]
        for stale_id in stale_ids:
            self._results_by_id.pop(stale_id, None)

    @staticmethod
    def _build_detection_signature(
        detections: list[dict[str, Any]],
        *,
        min_confidence: float = 0.25,
    ) -> set[str]:
        sig: set[str] = set()
        for det in detections:
            conf = float(det.get("confidence", 0.0))
            if conf < min_confidence:
                continue
            name = str(det.get("class_name", "unknown"))
            x1, y1, x2, y2 = [float(v) for v in det.get("xyxy", [0.0, 0.0, 0.0, 0.0])]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            x_bin = int(cx // 32)
            y_bin = int(cy // 32)
            area = w * h
            if area < 32 * 32:
                size_bin = "s"
            elif area < 96 * 96:
                size_bin = "m"
            else:
                size_bin = "l"
            sig.add(f"{name}:{x_bin}:{y_bin}:{size_bin}")
        return sig

    @staticmethod
    def _signature_jaccard(a: set[str] | None, b: set[str] | None) -> float:
        if a is None or b is None:
            return 0.0
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 0.0
        inter = a & b
        return len(inter) / len(union)

    def close(self) -> None:
        self._llm_in.put(None)
        if self._llm_process is not None:
            self._llm_process.join(timeout=5)
            return
        if self._worker is not None:
            self._worker.join(timeout=5)

    def set_llm_provider(self, provider: str) -> None:
        provider_norm = str(provider).strip().lower()
        if provider_norm in {"qwen_local", "gemini_api"}:
            self.llm_provider = provider_norm

    def set_detection_sensitivity(self, sensitivity: float) -> None:
        """
        Set runtime detection sensitivity (1.0 to 10.0).
        Higher values increase LLM trigger sensitivity.
        """
        try:
            value = float(sensitivity)
        except Exception:
            value = 5.0
        self._detection_sensitivity = min(10.0, max(1.0, value))

    def _resolve_llm_gating(self) -> tuple[float, float, float]:
        """
        Resolve runtime gating parameters from base config + sensitivity.
        """
        base_interval = float(self.llm_cfg.get("llm_call_interval_sec", 2.5))
        base_threshold = float(self.llm_cfg.get("similarity_threshold", 0.85))
        base_min_conf = float(self.llm_cfg.get("similarity_min_confidence", 0.25))

        sensitivity = min(10.0, max(1.0, float(self._detection_sensitivity)))
        norm = (sensitivity - 1.0) / 9.0

        # Overall shifted to be less sensitive across the full slider range.
        interval_scale = 1.9 - (1.1 * norm)  # [1.9, 0.8]
        llm_call_interval = max(0.3, base_interval * interval_scale)

        threshold_offset = -0.10 + (0.12 * norm)  # [-0.10, +0.02]
        similarity_threshold = min(0.95, max(0.20, base_threshold + threshold_offset))

        min_conf_scale = 1.5 - (0.5 * norm)  # [1.5, 1.0]
        similarity_min_confidence = min(0.9, max(0.08, base_min_conf * min_conf_scale))

        return llm_call_interval, similarity_threshold, similarity_min_confidence

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
                    provider=task.get("llm_provider"),
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

            llm_call_interval, similarity_threshold, similarity_min_confidence = self._resolve_llm_gating()
            current_sig = self._build_detection_signature(
                detections, min_confidence=similarity_min_confidence
            )
            similarity_score = self._signature_jaccard(current_sig, self._prev_detection_signature)
            similarity_blocked = (
                self._prev_detection_signature is not None and similarity_score >= similarity_threshold
            )
            self._prev_detection_signature = current_sig

            now = time.monotonic()
            with self._last_llm_lock:
                elapsed = now - self._last_llm_call_ts if self._last_llm_call_ts else float("inf")
                rate_allowed = elapsed >= llm_call_interval
                send_to_llm = rate_allowed and not similarity_blocked
                if send_to_llm:
                    self._last_llm_call_ts = now
            llm_prob, llm_sampled, llm_rate_allowed = 1.0, True, rate_allowed
            queued = False
            if send_to_llm:
                self._llm_in.put(
                    {
                        "frame_id": frame_id,
                        "processed_bytes": processed_bytes,
                        "detections": detections,
                        "llm_provider": self.llm_provider,
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
                "llm": {
                    "sent": send_to_llm,
                    "queued": queued,
                    "done": False if queued else True,
                    "probability": round(llm_prob, 4),
                    "sampled": llm_sampled,
                    "rate_allowed": llm_rate_allowed,
                    "detection_sensitivity": round(float(self._detection_sensitivity), 2),
                    "llm_call_interval_sec": round(llm_call_interval, 3),
                    "similarity_score": round(similarity_score, 4),
                    "similarity_threshold": similarity_threshold,
                    "similarity_min_confidence": similarity_min_confidence,
                    "similarity_blocked": similarity_blocked,
                    "description": None,
                    "error": None,
                },
                "latency": {
                    "preprocess_ms": round(preprocess_ms, 1),
                    "yolo_ms": round(yolo_ms, 1),
                    "depth_ms": round(float(yolo_result.get("depth_ms", 0.0)), 1),
                    "llm_ms": 0.0,
                    "total_ms": round(total_ms, 1),
                },
                "runtime": {
                    "yolo_pid": os.getpid(),
                    "llm_pid": self._llm_process.pid if self._llm_process is not None else os.getpid(),
                    "llm_worker_mode": self.llm_worker_mode,
                    "llm_provider": self.llm_provider,
                },
            }
            self._results_by_id[frame_id] = result
            self._prune_results()
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
    context: str = "general",
    use_depth: bool = True,
    preprocess_size: int = 256,
    imgsz: int = 256,
    return_annotated: bool = True,
    wait_for_llm: bool = True,
    llm_worker_mode: str = "thread",
    max_results: int = 300,
) -> list[dict[str, Any]]:
    pipeline = FrameSequencePipeline(
        context=context,
        use_depth=use_depth,
        preprocess_size=preprocess_size,
        imgsz=imgsz,
        return_annotated=return_annotated,
        llm_worker_mode=llm_worker_mode,
        max_results=max_results,
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
    context: str = "general",
    use_depth: bool = True,
    preprocess_size: int = 256,
    imgsz: int = 256,
    return_annotated: bool = True,
    wait_for_llm: bool = True,
    llm_worker_mode: str = "thread",
    max_results: int = 300,
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
        max_results=max_results,
    )
    return results[0]
