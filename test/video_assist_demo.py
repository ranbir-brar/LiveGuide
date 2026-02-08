"""Offline video-assist demo for local files in test/test_videos/."""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import FrameSequencePipeline  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _emit_log(path: Path, event: str, payload: dict) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_video(video_path: Path, fps: float = 6.0) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"video_events_{video_path.stem}.jsonl"
    log_path.write_text("", encoding="utf-8")

    pipeline = FrameSequencePipeline(
        context="general",
        use_depth=True,
        return_annotated=False,
        llm_worker_mode="process",
    )

    frame_idx = 0
    seen_llm = set()
    frame_interval = 1.0 / fps if fps > 0 else 0.0

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ok_jpg, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok_jpg:
                continue

            result = pipeline.submit_frames([buf.tobytes()])[-1]
            pipeline.poll_llm()

            det_count = len(result.get("detections", []))
            llm = result.get("llm", {})
            runtime = result.get("runtime", {})
            print(
                f"[YOLO] video={video_path.name} frame={frame_idx} det={det_count} "
                f"llm_sent={llm.get('sent')} sim_block={llm.get('similarity_blocked')} "
                f"provider={runtime.get('llm_provider')}"
            )
            _emit_log(
                log_path,
                "yolo_video",
                {
                    "video": video_path.name,
                    "frame": frame_idx,
                    "det_count": det_count,
                    "llm_sent": llm.get("sent"),
                    "sim_block": llm.get("similarity_blocked"),
                    "llm_provider": runtime.get("llm_provider"),
                },
            )

            for r in pipeline.get_results():
                fid = int(r["frame_id"])
                if fid in seen_llm:
                    continue
                l = r.get("llm", {})
                if l.get("sent") and l.get("done"):
                    seen_llm.add(fid)
                    desc = (l.get("description") or "").strip()
                    err = (l.get("error") or "").strip()
                    if desc or err:
                        print(f"[LLM ] video={video_path.name} frame_id={fid} text={desc} err={err}")
                        _emit_log(
                            log_path,
                            "llm_video",
                            {
                                "video": video_path.name,
                                "frame_id": fid,
                                "description": desc,
                                "error": err,
                                "llm_ms": r.get("latency", {}).get("llm_ms"),
                            },
                        )

            elapsed = time.time() - t0
            if frame_interval > elapsed:
                time.sleep(frame_interval - elapsed)

        pipeline.wait_for_all_llm()
        for r in pipeline.get_results():
            fid = int(r["frame_id"])
            if fid in seen_llm:
                continue
            l = r.get("llm", {})
            if l.get("sent") and l.get("done"):
                desc = (l.get("description") or "").strip()
                err = (l.get("error") or "").strip()
                if desc or err:
                    print(f"[LLM ] video={video_path.name} frame_id={fid} text={desc} err={err}")
                    _emit_log(
                        log_path,
                        "llm_video",
                        {
                            "video": video_path.name,
                            "frame_id": fid,
                            "description": desc,
                            "error": err,
                            "llm_ms": r.get("latency", {}).get("llm_ms"),
                        },
                    )

    finally:
        cap.release()
        pipeline.close()

    print(f"[DONE] {video_path.name} frames={frame_idx} log={log_path}")


def main() -> int:
    test_dir = Path(__file__).resolve().parent / "test_videos"
    videos = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print(f"No videos found in {test_dir}. Put files in test/test_videos/")
        return 1

    for v in videos:
        process_video(v, fps=6.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
