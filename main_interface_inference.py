#**********************************************************
# Main Interface Streaming Example                        #
# Streams YOLO/LLM events to console + logs               #
#**********************************************************

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from main_interface import FrameSequencePipeline


def main() -> int:
    here = Path(__file__).resolve().parent
    test_dir = here / "data" / "test_images"
    logs_dir = here / "logs"
    logs_dir.mkdir(exist_ok=True)
    event_log = logs_dir / "pipeline_events.jsonl"

    image_paths = sorted(
        p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        print(f"No test images found in {test_dir}")
        return 1

    def emit(event_type: str, payload: dict[str, Any], message: str) -> None:
        print(message)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        with event_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pipeline = FrameSequencePipeline(
        context="walking",
        use_depth=True,
        return_annotated=True,
        llm_worker_mode="process",
    )
    results: list[dict[str, Any]] = []
    printed_llm_ids: set[int] = set()
    frame_name_by_id: dict[int, str] = {}

    try:
        event_log.write_text("", encoding="utf-8")
        print(f"logging to {event_log}")
        print(f"num_frames = {len(image_paths)}")

        for path in image_paths:
            frame_bytes = path.read_bytes()
            result = pipeline.submit_frames([frame_bytes])[-1]

            frame_id = int(result["frame_id"])
            frame_name_by_id[frame_id] = path.name
            result["source_image"] = path.name

            emit(
                "yolo",
                {
                    "frame_id": frame_id,
                    "source_image": path.name,
                    "det_count": len(result["detections"]),
                    "hazard_level": result["hazard"]["hazard_level"],
                    "hazard_score": result["hazard"]["hazard_score"],
                    "danger_active": result["danger"]["active"],
                    "llm_sent": result["llm"]["sent"],
                    "sim_block": result["llm"]["similarity_blocked"],
                    "yolo_pid": result["runtime"]["yolo_pid"],
                    "llm_pid": result["runtime"]["llm_pid"],
                    "mode": result["runtime"]["llm_worker_mode"],
                },
                (
                    f"[YOLO] frame={frame_id} image={path.name} det={len(result['detections'])} "
                    f"hazard={result['hazard']['hazard_level']}({result['hazard']['hazard_score']}) "
                    f"danger_active={result['danger']['active']} llm_sent={result['llm']['sent']} "
                    f"sim_block={result['llm']['similarity_blocked']} "
                    f"yolo_pid={result['runtime']['yolo_pid']} llm_pid={result['runtime']['llm_pid']}"
                ),
            )

            for r in pipeline.get_results():
                rid = int(r["frame_id"])
                if rid in printed_llm_ids:
                    continue
                llm = r["llm"]
                if llm.get("sent") and llm.get("done"):
                    printed_llm_ids.add(rid)
                    emit(
                        "llm",
                        {
                            "frame_id": rid,
                            "source_image": frame_name_by_id.get(rid, r.get("source_image")),
                            "description": llm.get("description"),
                            "error": llm.get("error"),
                            "llm_ms": r["latency"]["llm_ms"],
                        },
                        (
                            f"[LLM ] frame={rid} image={frame_name_by_id.get(rid, r.get('source_image', 'unknown'))} "
                            f"done={llm.get('done')} error={bool(llm.get('error'))} "
                            f"text={llm.get('description') or ''}"
                        ),
                    )

        pipeline.wait_for_all_llm()
        for r in pipeline.get_results():
            rid = int(r["frame_id"])
            if rid in printed_llm_ids:
                continue
            llm = r["llm"]
            if llm.get("sent") and llm.get("done"):
                printed_llm_ids.add(rid)
                emit(
                    "llm",
                    {
                        "frame_id": rid,
                        "source_image": frame_name_by_id.get(rid, r.get("source_image")),
                        "description": llm.get("description"),
                        "error": llm.get("error"),
                        "llm_ms": r["latency"]["llm_ms"],
                    },
                    (
                        f"[LLM ] frame={rid} image={frame_name_by_id.get(rid, r.get('source_image', 'unknown'))} "
                        f"done={llm.get('done')} error={bool(llm.get('error'))} "
                        f"text={llm.get('description') or ''}"
                    ),
                )

        results = pipeline.get_results()
    finally:
        pipeline.close()

    for r in results:
        r["source_image"] = frame_name_by_id.get(int(r["frame_id"]), "unknown")

    out_json = here / "result.json"
    out_png = here / "annotated.png"
    if results and results[0].get("image_base64"):
        out_png.write_bytes(base64.b64decode(results[0]["image_base64"]))
        print(f"saved {out_png.name}")

    payload = []
    for path, res in zip(image_paths, results, strict=True):
        item = dict(res)
        item["source_image"] = path.name
        payload.append(item)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
