"""Local, rule-based visual screening for line-like surface anomalies."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np


class QualityInspector:
    """Find persistent, high-contrast line candidates in a configured image ROI.

    This is a screening heuristic for visible surface marks, not a crack classifier.
    Measurements are image pixels unless the optional pixel/mm scale is calibrated.
    """

    def __init__(self, settings: Mapping[str, Any], event_csv_path: str | Path):
        self.enabled = bool(settings.get("enabled", True))
        roi = settings.get("roi_normalized", [0.05, 0.05, 0.9, 0.9])
        if len(roi) != 4:
            raise ValueError("quality inspection roi_normalized must contain x, y, width, height")
        self.roi = tuple(float(value) for value in roi)
        if (
            any(not math.isfinite(value) for value in self.roi)
            or self.roi[0] < 0 or self.roi[1] < 0
            or self.roi[2] <= 0 or self.roi[3] <= 0
            or self.roi[0] + self.roi[2] > 1 or self.roi[1] + self.roi[3] > 1
        ):
            raise ValueError("quality inspection ROI must fit within normalized image bounds [0, 1]")
        self.contrast_threshold = int(settings.get("contrast_threshold", 35))
        self.min_length_px = float(settings.get("min_length_px", 16))
        self.max_width_px = float(settings.get("max_width_px", 12))
        self.min_aspect_ratio = float(settings.get("min_aspect_ratio", 2.2))
        self.min_area_px = int(settings.get("min_area_px", 8))
        self.warning_length_px = float(settings.get("warning_length_px", 30))
        self.critical_length_px = float(settings.get("critical_length_px", 90))
        self.critical_width_px = float(settings.get("critical_width_px", 8))
        self.confirmation_frames = max(1, int(settings.get("confirmation_frames", 3)))
        self.max_candidates = max(1, int(settings.get("max_candidates", 20)))
        self.polarity = str(settings.get("polarity", "dark")).lower()
        if self.polarity not in {"dark", "bright"}:
            raise ValueError("quality inspection polarity must be 'dark' or 'bright'")
        if not 0 <= self.contrast_threshold <= 255 or self.min_length_px <= 0 or self.max_width_px <= 0:
            raise ValueError("quality inspection contrast, minimum length, and maximum width are out of range")
        if self.critical_length_px < self.warning_length_px:
            raise ValueError("critical_length_px must be greater than or equal to warning_length_px")
        self.pixels_per_mm = float(settings.get("pixels_per_mm", 0.0))
        if not math.isfinite(self.pixels_per_mm) or self.pixels_per_mm < 0:
            raise ValueError("pixels_per_mm must be zero (uncalibrated) or a positive calibrated scale")
        self.event_csv_path = Path(event_csv_path)
        self.event_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.event_csv_path.exists():
            with self.event_csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=[
                    "Timestamp", "Inspection_ID", "Severity", "Length_Px", "Width_Px",
                    "BoundingBox_X", "BoundingBox_Y", "BoundingBox_Width", "BoundingBox_Height",
                    "Contrast", "ROI", "Recommended_Action",
                ]).writeheader()
        self._next_id = 1
        self._candidates: Dict[str, Dict[str, Any]] = {}
        self._events: deque[Dict[str, Any]] = deque(maxlen=100)
        self._latest: Dict[str, Any] = {
            "enabled": self.enabled,
            "status": "WAITING" if self.enabled else "DISABLED",
            "severity": "none",
            "confirmed_candidate_count": 0,
            "pending_candidate_count": 0,
            "candidates": [],
            "measurement_unit": "pixels",
            "assessment_scope": "Rule-based screening for persistent visible line-like contrast; not certified crack detection.",
            "recommended_action": "Position the camera on the surface and verify the configured ROI, focus, and lighting.",
        }

    @staticmethod
    def _severity(length_px: float, width_px: float, settings: "QualityInspector") -> str:
        if length_px >= settings.critical_length_px or width_px >= settings.critical_width_px:
            return "critical"
        if length_px >= settings.warning_length_px:
            return "warning"
        return "minor"

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"minor": 0, "warning": 1, "critical": 2}.get(severity, -1)

    def _find_candidates(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        rx, ry, rw, rh = self.roi
        x0 = max(0, min(width - 1, int(rx * width)))
        y0 = max(0, min(height - 1, int(ry * height)))
        x1 = max(x0 + 1, min(width, int((rx + rw) * width)))
        y1 = max(y0 + 1, min(height, int((ry + rh) * height)))
        region = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        kernel_size = max(9, int(round(min(region.shape[:2]) * 0.035)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        background = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE if self.polarity == "dark" else cv2.MORPH_OPEN, kernel
        )
        response = cv2.subtract(background, gray) if self.polarity == "dark" else cv2.subtract(gray, background)
        _, mask = cv2.threshold(response, self.contrast_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = []
        for label in range(1, count):
            bx, by, bw, bh, area = (int(value) for value in stats[label])
            if area < self.min_area_px:
                continue
            points = cv2.findNonZero((labels[by:by + bh, bx:bx + bw] == label).astype(np.uint8))
            if points is None or len(points) < 2:
                continue
            points = points.reshape(-1, 2).astype(np.float32)
            center = points.mean(axis=0)
            centered = points - center
            covariance = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major = eigenvectors[:, int(np.argmax(eigenvalues))]
            minor = np.array([-major[1], major[0]], dtype=np.float32)
            longitudinal = centered @ major
            transverse = centered @ minor
            length = float(longitudinal.max() - longitudinal.min() + 1.0)
            line_width = float(transverse.max() - transverse.min() + 1.0)
            aspect_ratio = length / max(1.0, line_width)
            if length < self.min_length_px or line_width > self.max_width_px or aspect_ratio < self.min_aspect_ratio:
                continue
            contrast = float(response[by:by + bh, bx:bx + bw][labels[by:by + bh, bx:bx + bw] == label].mean())
            candidates.append({
                "bbox": [x0 + bx, y0 + by, bw, bh],
                "center": [x0 + float(center[0]), y0 + float(center[1])],
                "length_px": round(length, 1),
                "width_px": round(line_width, 1),
                "contrast": round(contrast, 1),
                "area_px": area,
            })
        candidates.sort(key=lambda item: (item["length_px"], item["contrast"]), reverse=True)
        return candidates[:self.max_candidates], (x0, y0, x1 - x0, y1 - y0)

    def _log_event(self, candidate: Dict[str, Any], roi_px: Tuple[int, int, int, int]) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        x, y, width, height = candidate["bbox"]
        action = candidate["recommended_action"]
        row = {
            "Timestamp": timestamp,
            "Inspection_ID": candidate["inspection_id"],
            "Severity": candidate["severity"].upper(),
            "Length_Px": candidate["length_px"],
            "Width_Px": candidate["width_px"],
            "BoundingBox_X": x,
            "BoundingBox_Y": y,
            "BoundingBox_Width": width,
            "BoundingBox_Height": height,
            "Contrast": candidate["contrast"],
            "ROI": json.dumps(list(roi_px)),
            "Recommended_Action": action,
        }
        with self.event_csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
        event = {
            "timestamp": timestamp,
            "rule": "visual_surface_anomaly",
            "severity": candidate["severity"],
            "message": f"{candidate['inspection_id']} persistent line-like surface candidate: {candidate['severity']} review.",
            "details": {
                "length_px": candidate["length_px"],
                "width_px": candidate["width_px"],
                "contrast": candidate["contrast"],
                "recommended_action": action,
                "assessment_scope": self._latest["assessment_scope"],
            },
        }
        self._events.append(event)

    def update(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.enabled:
            self._latest = {**self._latest, "status": "DISABLED", "candidates": []}
            return self._latest
        if frame is None or frame.size == 0:
            raise ValueError("quality inspection requires a non-empty webcam frame")
        detections, roi_px = self._find_candidates(frame)
        unmatched = set(self._candidates)
        observed = []
        for detection in detections:
            cx, cy = detection["center"]
            best_id = None
            best_distance = float("inf")
            for candidate_id in unmatched:
                old = self._candidates[candidate_id]
                ox, oy = old["center"]
                distance = math.hypot(cx - ox, cy - oy)
                limit = max(12.0, min(40.0, detection["length_px"] * 0.5))
                if distance <= limit and distance < best_distance:
                    best_id, best_distance = candidate_id, distance
            if best_id is None:
                best_id = f"Q-{self._next_id:04d}"
                self._next_id += 1
                track = {
                    "inspection_id": best_id, "hits": 1, "missed": 0,
                    "last_logged_severity": None,
                }
            else:
                unmatched.remove(best_id)
                track = self._candidates[best_id]
                track["hits"] += 1
                track["missed"] = 0
            severity = self._severity(detection["length_px"], detection["width_px"], self)
            action = {
                "critical": "Hold the item for qualified quality review; do not infer failure or release it based on this screening.",
                "warning": "Inspect this region using the approved quality procedure before accepting the item.",
                "minor": "Record and visually verify this candidate under the approved inspection procedure.",
            }[severity]
            candidate = {
                **track,
                **detection,
                "severity": severity,
                "recommended_action": action,
                "confirmed": track["hits"] >= self.confirmation_frames,
            }
            track.update(detection)
            track["center"] = detection["center"]
            track["severity"] = severity
            track["recommended_action"] = action
            if self.pixels_per_mm > 0:
                candidate["length_mm"] = round(detection["length_px"] / self.pixels_per_mm, 2)
                candidate["width_mm"] = round(detection["width_px"] / self.pixels_per_mm, 2)
            if candidate["confirmed"] and (
                track["last_logged_severity"] is None
                or self._severity_rank(severity) > self._severity_rank(track["last_logged_severity"])
            ):
                self._log_event(candidate, roi_px)
                track["last_logged_severity"] = severity
            self._candidates[best_id] = track
            observed.append(candidate)
        for candidate_id in unmatched:
            track = self._candidates[candidate_id]
            track["missed"] += 1
            if track["hits"] >= self.confirmation_frames and track["missed"] <= 2:
                observed.append({**track, "confirmed": True})
            if track["missed"] > 2:
                self._candidates.pop(candidate_id, None)
        confirmed = [candidate for candidate in observed if candidate["confirmed"]]
        priority = {"critical": 0, "warning": 1, "minor": 2}
        confirmed.sort(key=lambda item: (priority[item["severity"]], -item["length_px"]))
        unit = "mm (calibrated)" if self.pixels_per_mm > 0 else "pixels"
        state = confirmed[0]["severity"].upper() if confirmed else ("CONFIRMING" if observed else "NO CANDIDATE")
        self._latest = {
            "enabled": True,
            "status": state,
            "severity": confirmed[0]["severity"] if confirmed else "none",
            "confirmed_candidate_count": len(confirmed),
            "pending_candidate_count": len(observed) - len(confirmed),
            "candidates": [{key: value for key, value in candidate.items() if key != "center"} for candidate in confirmed],
            "roi_px": list(roi_px),
            "measurement_unit": unit,
            "assessment_scope": "Rule-based screening for persistent visible line-like contrast; not certified crack detection.",
            "recommended_action": confirmed[0]["recommended_action"] if confirmed else (
                "Verify camera focus, lighting, and ROI; no confirmed line-like candidate in the configured region."
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        return self._latest

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        for candidate in self._latest.get("candidates", []):
            x, y, width, height = candidate["bbox"]
            color = (0, 0, 255) if candidate["severity"] == "critical" else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            label = f"{candidate['inspection_id']} {candidate['severity'].upper()} {candidate['length_px']:.0f}px"
            cv2.putText(frame, label, (max(0, x), max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def status(self) -> Dict[str, Any]:
        return dict(self._latest, events=list(self._events))
