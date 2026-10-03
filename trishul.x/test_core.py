import sys
import unittest
from pathlib import Path
import numpy as np
import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from app import Handler, Tracker, iou, VisionApp
from industrial import IndustrialConfig, IndustrialMonitor


class CoreTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,10,10]), 1/3)

    def test_tracker_recovers_same_id(self):
        tracker=Tracker(); tracker.fps=10
        det=lambda x:{"bbox":[x,0,20,40],"class_id":0,"class_name":"person","confidence":.8}
        first=tracker.update([det(10)], timestamp=10)[0]["id"]
        track=tracker.update([det(12)], timestamp=10.1)[0]
        self.assertEqual(first, track["id"])
        self.assertEqual(track["state"], "TRACKING")

    def test_new_detections_are_not_marked_missed_or_duplicated(self):
        tracker=Tracker()
        detections=[
            {"bbox":[10,10,30,60],"class_id":0,"class_name":"person","confidence":.9},
            {"bbox":[100,10,30,60],"class_id":0,"class_name":"person","confidence":.9},
        ]
        tracks=tracker.update(detections, timestamp=5)
        self.assertEqual(len(tracks), 2)
        self.assertEqual({track["state"] for track in tracks}, {"NEW"})
        self.assertEqual({track["missed"] for track in tracks}, {0})
        again=tracker.update(detections, timestamp=5.1)
        self.assertEqual({track["id"] for track in tracks}, {track["id"] for track in again})
        self.assertEqual(len({track["id"] for track in again}), 2)

    def test_weak_detection_does_not_create_track(self):
        tracker=Tracker()
        weak={"bbox":[10,10,30,60],"class_id":0,"class_name":"person","confidence":.1}
        self.assertEqual(tracker.update([weak], timestamp=1), [])

    def test_missing_track_can_be_reidentified_from_local_appearance(self):
        tracker=Tracker()
        frame=np.zeros((240,320,3),dtype=np.uint8)
        frame[25:210,25:95]=(0,0,220)
        first={"bbox":[25,25,70,185],"class_id":0,"class_name":"person","confidence":.9}
        original=tracker.update([first],frame,timestamp=10)[0]
        self.assertIsNotNone(original["appearance"])
        tracker.update([],timestamp=13)
        reentry_frame=np.zeros_like(frame)
        reentry_frame[25:210,180:250]=(0,0,220)
        reentry={"bbox":[180,25,70,185],"class_id":0,"class_name":"person","confidence":.9}
        reacquired=tracker.update([reentry],reentry_frame,timestamp=14)
        restored=next(track for track in reacquired if track["id"]==original["id"])
        self.assertEqual(restored["state"],"TRACKING")
        self.assertEqual(restored["identity_event"],"RE-IDENTIFIED")

    def test_industrial_zone_event(self):
        cfg=IndustrialConfig.from_mapping({"zones":[{"name":"A","polygon":[[0,0],[2,0],[2,2],[0,2]]}], "rules_enabled":["zone_entry"]})
        engine=IndustrialMonitor(cfg)
        events=engine.update([{"id":"c1","class_name":"component","x_m":1,"y_m":1}])
        self.assertTrue(any(e.rule == "zone_entry" for e in events))

    def test_industrial_hazard_alert_stays_active_until_track_leaves(self):
        cfg=IndustrialConfig.from_mapping({
            "zones":[{"name":"Restricted","polygon":[[0,0],[5,0],[5,5],[0,5]],"hazard":True,"recommended_action":"Stop work and notify the controller."}],
            "rules_enabled":["hazard_zone_entry","zone_exit"],
        })
        engine=IndustrialMonitor(cfg)
        events=engine.update([{"id":"Person-001","class_name":"person","x":2,"y":2}],100.0)
        self.assertTrue(any(event.rule=="hazard_zone_entry" and event.severity=="critical" for event in events))
        self.assertEqual(len(engine.active_alerts()),1)
        engine.update([],101.0)
        self.assertEqual(engine.active_alerts(),[])

    def test_industrial_app_path_emits_pixel_zone_events(self):
        app=VisionApp(mode="industrial")
        track={"id":1,"class_name":"person","bbox":[20,20,30,80],"state":"TRACKING"}
        app._update_industrial([track],100.0)
        self.assertTrue(any(event.get("rule")=="zone_entry" for event in app.events))
        app._update_industrial([],101.0)
        self.assertTrue(any(event.get("rule")=="zone_exit" for event in app.events))

    def test_public_safety_summary_distinguishes_empty_warning_and_missing(self):
        app=VisionApp()
        empty=app.public_safety_summary()
        self.assertEqual(empty["status"],"NO ACTIVE TRACKS")
        self.assertFalse(empty["human_verification_required"])

        app.tracks=[{
            "id":1,"class_name":"person","bbox":[0,0,50,100],"confidence":0.3,
            "state":"TRACKING","velocity_px_s":[0.0,0.0],"stationary_since":time.monotonic(),
            "last_seen_wall":time.time(),
        }]
        warning=app.public_safety_summary()
        self.assertEqual(warning["status"],"WARNING")
        self.assertTrue(warning["human_verification_required"])

        app.tracks[0]["state"]="MISSING"
        missing=app.public_safety_summary()
        self.assertEqual(missing["status"],"MISSING")
        self.assertEqual(missing["counts"]["MISSING"],1)

    def test_assistance_request_is_operator_confirmed_and_contains_no_face_data(self):
        app=VisionApp()
        app.tracks=[{
            "id":7,"bbox":[10,20,30,60],"confidence":0.8,"safety_status":"WARNING",
            "last_seen_wall":time.time(),
        }]
        with self.assertRaises(ValueError):
            app.request_assistance("Person-007","possible_distress",False)
        record=app.request_assistance("Person-007","possible_distress",True)
        self.assertEqual(record["handoff_status"],"LOCAL_OPERATOR_REVIEW_REQUIRED")
        self.assertEqual(record["image_location_px"],{"center_x":25.0,"center_y":50.0})
        self.assertNotIn("image",record)
        self.assertNotIn("face",record)
        self.assertEqual(len(app.assistance_requests),1)

    def test_http_dashboard_assets_and_industrial_mode(self):
        app=VisionApp()
        Handler.app=app
        server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        base=f"http://127.0.0.1:{server.server_port}"
        try:
            for path, marker in (("/",b"PUBLIC / INDUSTRIAL SAFETY"),("/dashboard/app.js",b"renderPublicSafety"),("/dashboard/styles.css",b":root"),("/dashboard/public-safety.css",b"public-safety-bar")):
                with urlopen(base+path,timeout=5) as response:
                    self.assertEqual(response.status,200)
                    self.assertIn(marker,response.read())
            request=Request(base+"/api/mode",data=json.dumps({"mode":"industrial"}).encode(),headers={"Content-Type":"application/json"},method="POST")
            with urlopen(request,timeout=5) as response:
                payload=json.loads(response.read())
            self.assertEqual(payload["mode"],"industrial")
            self.assertNotEqual(payload["model_status"],"NO DETECTOR")
            with urlopen(base+"/api/status",timeout=5) as response:
                self.assertEqual(json.loads(response.read())["industrial_monitor"],"active")
            with urlopen(base+"/api/public-safety",timeout=5) as response:
                self.assertEqual(json.loads(response.read())["status"],"NO ACTIVE TRACKS")
            with urlopen(base+"/api/industrial/status",timeout=5) as response:
                self.assertTrue(json.loads(response.read())["enabled"])
            app.tracks=[{"id":9,"bbox":[2,3,10,20],"confidence":0.8,"safety_status":"WARNING","last_seen_wall":time.time()}]
            request=Request(base+"/api/assistance",data=json.dumps({"person_id":"Person-009","reason":"operator_observed_concern","operator_confirmed":True}).encode(),headers={"Content-Type":"application/json"},method="POST")
            with urlopen(request,timeout=5) as response:
                assistance=json.loads(response.read())
            self.assertEqual(assistance["request"]["handoff_status"],"LOCAL_OPERATOR_REVIEW_REQUIRED")
            self.assertIn("No face data",assistance["message"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_camera_absence_is_explicit(self):
        app=VisionApp()
        status=app.status()
        self.assertTrue(status["offline"])
        if not status["camera_online"]:
            self.assertTrue(status["camera_error"] or status["detector"] == "Unavailable (install opencv-python)")


if __name__ == "__main__": unittest.main()
