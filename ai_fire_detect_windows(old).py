import base64
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import requests
from pymavlink import mavutil
from ultralytics import YOLO

# ============================================================
# AI FIRE / SMOKE DETECTION
# + RTSP / USB CAMERA INPUT
# + MAVLINK GPS
# + DASHBOARD TELEMETRY
# + FIRE ALERT IMAGE
#
# Recommended DJI Air 2S source:
#     rtsp://127.0.0.1:8554/air2s
#
# VIDEO_SOURCE options:
#     rtsp   -> MediaMTX / DJI gateway RTSP stream
#     camera -> Local USB / laptop camera
#     file   -> Local video file
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# -------------------- Video input ----------------------------

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "rtsp").strip().lower()
RTSP_URL = os.getenv("RTSP_URL", "rtsp://127.0.0.1:8554/air2s").strip()
RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
VIDEO_FILE = os.getenv("VIDEO_FILE", "").strip()

VIDEO_RECONNECT_DELAY = float(os.getenv("VIDEO_RECONNECT_DELAY", "2.0"))
MAX_FRAME_FAILURES = int(os.getenv("MAX_FRAME_FAILURES", "10"))
FRAME_WAIT_TIMEOUT = float(os.getenv("FRAME_WAIT_TIMEOUT", "5.0"))

DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", "800"))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", "600"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "true").strip().lower() in {
    "1", "true", "yes", "on"
}

# These options are read by OpenCV's FFmpeg backend when the RTSP
# connection is opened. TCP is more reliable for Wi-Fi testing.
if VIDEO_SOURCE == "rtsp":
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{RTSP_TRANSPORT}"
        "|fflags;nobuffer"
        "|flags;low_delay"
        "|max_delay;500000"
        "|stimeout;5000000"
    )

# -------------------- AI model -------------------------------

MODEL_PATH = os.getenv(
    "FIRE_MODEL_PATH",
    str(BASE_DIR / "fire_smoke.pt")
)

CONFIDENCE_LIMIT = float(os.getenv("CONFIDENCE_LIMIT", "0.30"))
ALERT_HOLD_TIME = float(os.getenv("ALERT_HOLD_TIME", "2.0"))
ALERT_COOLDOWN = float(os.getenv("ALERT_COOLDOWN", "20.0"))
TELEMETRY_SEND_INTERVAL = float(
    os.getenv("TELEMETRY_SEND_INTERVAL", "0.5")
)

# -------------------- Dashboard ------------------------------

DASHBOARD_BASE_URL = os.getenv(
    "DASHBOARD_BASE_URL",
    "http://127.0.0.1:5000"
).rstrip("/")

DASHBOARD_ALERT_URL = f"{DASHBOARD_BASE_URL}/api/fire-alert"
DASHBOARD_TELEMETRY_URL = f"{DASHBOARD_BASE_URL}/api/telemetry"

# -------------------- MAVLink --------------------------------

DRONE_ID = os.getenv("DRONE_ID", "DRONE_001")

# Keep this configurable because the correct port depends on your
# RosettaDrone telemetry settings.
MAVLINK_CONNECTION_STRING = os.getenv(
    "MAVLINK_CONNECTION_STRING",
    "udp:127.0.0.1:3000"
)

LAST_GPS = {
    "latitude": 0.0,
    "longitude": 0.0,
    "altitude": 0.0,
    "gps_ok": False,
}


def video_source_description() -> str:
    if VIDEO_SOURCE == "rtsp":
        return RTSP_URL
    if VIDEO_SOURCE == "camera":
        return f"camera:{CAMERA_INDEX}"
    if VIDEO_SOURCE == "file":
        return VIDEO_FILE
    return VIDEO_SOURCE


class ResilientVideoReader:
    """
    Continuously reads the selected source in a background thread.

    Keeping capture in a separate thread prevents RTSP frames from building
    up while YOLO is processing. The AI loop always receives the latest frame.
    """

    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._connected = False
        self._last_error = "Video reader has not started"
        self._last_frame_time = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="video-reader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        capture = self._capture
        if capture is not None:
            capture.release()

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        with self._lock:
            self._connected = False

    def get_latest(
        self,
        last_frame_id: int,
    ) -> Tuple[Optional[np.ndarray], int, bool, str, float]:
        """
        Return a copy of the newest frame when it has not already been read.
        """
        with self._lock:
            connected = self._connected
            error = self._last_error
            frame_time = self._last_frame_time

            if self._latest_frame is None or self._frame_id == last_frame_id:
                return None, self._frame_id, connected, error, frame_time

            return (
                self._latest_frame.copy(),
                self._frame_id,
                connected,
                error,
                frame_time,
            )

    def _set_status(self, connected: bool, error: str = "") -> None:
        with self._lock:
            self._connected = connected
            if error:
                self._last_error = error

    def _open_capture(self) -> cv2.VideoCapture:
        if VIDEO_SOURCE == "rtsp":
            print(f"Opening RTSP stream: {RTSP_URL}")
            capture = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        elif VIDEO_SOURCE == "camera":
            print(f"Opening camera index: {CAMERA_INDEX}")
            if os.name == "nt":
                capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            else:
                capture = cv2.VideoCapture(CAMERA_INDEX)

        elif VIDEO_SOURCE == "file":
            if not VIDEO_FILE:
                raise ValueError(
                    "VIDEO_FILE is empty while VIDEO_SOURCE=file"
                )

            print(f"Opening video file: {VIDEO_FILE}")
            capture = cv2.VideoCapture(VIDEO_FILE)

        else:
            raise ValueError(
                "Unsupported VIDEO_SOURCE. Use rtsp, camera, or file."
            )

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                capture = self._open_capture()
                self._capture = capture

                if not capture.isOpened():
                    message = (
                        f"Unable to open video source: "
                        f"{video_source_description()}"
                    )
                    print(message)
                    self._set_status(False, message)
                    capture.release()
                    self._sleep_before_reconnect()
                    continue

                print(
                    "Video source connected:",
                    video_source_description(),
                )
                self._set_status(True, "")
                consecutive_failures = 0

                while not self._stop_event.is_set():
                    success, frame = capture.read()

                    if not success or frame is None:
                        consecutive_failures += 1

                        if consecutive_failures == 1:
                            print("Video frame was not received.")

                        if consecutive_failures >= MAX_FRAME_FAILURES:
                            message = (
                                "Video stream lost. Reconnecting..."
                            )
                            print(message)
                            self._set_status(False, message)
                            break

                        time.sleep(0.03)
                        continue

                    consecutive_failures = 0

                    with self._lock:
                        self._latest_frame = frame
                        self._frame_id += 1
                        self._connected = True
                        self._last_error = ""
                        self._last_frame_time = time.time()

                capture.release()
                self._capture = None

            except Exception as error:
                message = f"Video reader error: {error}"
                print(message)
                self._set_status(False, message)

            self._sleep_before_reconnect()

    def _sleep_before_reconnect(self) -> None:
        end_time = time.time() + VIDEO_RECONNECT_DELAY

        while (
            not self._stop_event.is_set()
            and time.time() < end_time
        ):
            time.sleep(0.1)


def make_status_frame(message: str) -> np.ndarray:
    frame = np.zeros(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        frame,
        "DJI AIR 2S AI INPUT",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        message[:85],
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        video_source_description()[:95],
        (30, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )

    cv2.putText(
        frame,
        "Waiting for stream. Press Q to quit.",
        (30, DISPLAY_HEIGHT - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return frame


def connect_to_drone():
    print("Connecting to drone/SITL:", MAVLINK_CONNECTION_STRING)

    try:
        master = mavutil.mavlink_connection(
            MAVLINK_CONNECTION_STRING
        )

        print("Waiting for MAVLink heartbeat...")
        master.wait_heartbeat(timeout=15)

        print("Heartbeat received")
        print("System ID:", master.target_system)
        print("Component ID:", master.target_component)
        return master

    except Exception as error:
        print("Drone/SITL connection failed:", error)
        print(
            "AI detection will continue, but GPS will remain "
            "0.0 until MAVLink connects."
        )
        return None


def read_latest_drone_gps(master):
    global LAST_GPS

    if master is None:
        return LAST_GPS

    try:
        while True:
            message = master.recv_match(
                type="GLOBAL_POSITION_INT",
                blocking=False,
            )

            if message is None:
                break

            LAST_GPS = {
                "latitude": message.lat / 1e7,
                "longitude": message.lon / 1e7,
                "altitude": message.relative_alt / 1000.0,
                "gps_ok": True,
            }

    except Exception as error:
        print("GPS read error:", error)

    return LAST_GPS


def get_drone_gps(master):
    return read_latest_drone_gps(master)


def frame_to_base64(
    frame: np.ndarray,
    quality: int = 80,
) -> Optional[str]:
    encode_parameters = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        int(quality),
    ]

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        encode_parameters,
    )

    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")


HTTP_SESSION = requests.Session()


def post_json(
    url: str,
    payload: dict,
    timeout: Union[int, float] = 2,
):
    try:
        response = HTTP_SESSION.post(
            url,
            json=payload,
            timeout=timeout,
        )
        return response.status_code, response.text

    except requests.RequestException as error:
        return None, str(error)


def send_live_telemetry(
    gps: dict,
    label: str,
    confidence: float,
    fire_detected: bool,
    fire_confirmed: bool,
):
    payload = {
        "drone_id": DRONE_ID,
        "latitude": gps["latitude"],
        "longitude": gps["longitude"],
        "altitude": gps["altitude"],
        "gps_ok": gps["gps_ok"],
        "latest_label": label if label else "--",
        "latest_confidence": round(float(confidence), 3),
        "fire_detected": bool(fire_detected),
        "fire_confirmed": bool(fire_confirmed),
        "mavlink_connection": MAVLINK_CONNECTION_STRING,
        "video_source": video_source_description(),
        "timestamp": datetime.now().isoformat(),
        "source": "ai_fire_detect_windows",
    }

    status, text = post_json(
        DASHBOARD_TELEMETRY_URL,
        payload,
        timeout=1,
    )

    if status is None:
        # Do not print on every frame when the dashboard is offline.
        return False, text

    return True, text


def send_alert_to_dashboard(
    label: str,
    confidence: float,
    frame: np.ndarray,
    gps: dict,
) -> None:
    image_base64 = frame_to_base64(frame, quality=75)

    payload = {
        "drone_id": DRONE_ID,
        "event_type": f"{label}_detected",
        "label": label,
        "confidence": round(float(confidence), 3),
        "latitude": gps["latitude"],
        "longitude": gps["longitude"],
        "altitude": gps["altitude"],
        "gps_ok": gps["gps_ok"],
        "video_source": video_source_description(),
        "timestamp": datetime.now().isoformat(),
        "status": "new",
        "image": image_base64,
    }

    print("\n==============================")
    print("CONFIRMED ALERT PAYLOAD")
    print({
        "drone_id": payload["drone_id"],
        "event_type": payload["event_type"],
        "label": payload["label"],
        "confidence": payload["confidence"],
        "latitude": payload["latitude"],
        "longitude": payload["longitude"],
        "altitude": payload["altitude"],
        "gps_ok": payload["gps_ok"],
        "timestamp": payload["timestamp"],
        "status": payload["status"],
        "image_included": image_base64 is not None,
    })
    print("==============================\n")

    status, text = post_json(
        DASHBOARD_ALERT_URL,
        payload,
        timeout=5,
    )

    if status in (200, 201):
        print("Dashboard alert sent successfully")
    else:
        print(
            "Dashboard alert send failed:",
            status,
            text,
        )


def main() -> None:
    if VIDEO_SOURCE not in {"rtsp", "camera", "file"}:
        raise ValueError(
            "VIDEO_SOURCE must be rtsp, camera, or file"
        )

    if VIDEO_SOURCE == "file" and not VIDEO_FILE:
        raise ValueError(
            "Set VIDEO_FILE when VIDEO_SOURCE=file"
        )

    if not Path(MODEL_PATH).exists():
        print("ERROR: AI model file not found:", MODEL_PATH)
        print(
            "Place fire_smoke.pt in the same folder as this "
            "script or set FIRE_MODEL_PATH."
        )
        return

    print("Loading AI model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully:", MODEL_PATH)

    print("Selected video source:", VIDEO_SOURCE)
    print("Video input:", video_source_description())

    master = connect_to_drone()

    video_reader = ResilientVideoReader()
    video_reader.start()

    fire_start_time: Optional[float] = None
    last_alert_time = 0.0
    last_telemetry_time = 0.0
    last_dashboard_error_print = 0.0
    last_frame_id = -1
    last_frame_received_at = time.time()

    print("AI Fire/Smoke Detection started.")
    print("Press Q or Escape to quit.")

    try:
        while True:
            (
                frame,
                frame_id,
                video_connected,
                video_error,
                frame_timestamp,
            ) = video_reader.get_latest(last_frame_id)

            if frame is None:
                if time.time() - last_frame_received_at > FRAME_WAIT_TIMEOUT:
                    fire_start_time = None

                if SHOW_WINDOW:
                    status_message = (
                        "Connected, waiting for next frame..."
                        if video_connected
                        else video_error or "Connecting to video..."
                    )

                    cv2.imshow(
                        "AI Fire Detection - Windows",
                        make_status_frame(status_message),
                    )

                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.03)

                continue

            last_frame_id = frame_id
            last_frame_received_at = frame_timestamp

            frame = cv2.resize(
                frame,
                (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            )
            display_frame = frame.copy()

            gps = get_drone_gps(master)

            try:
                results = model(frame, verbose=False)
            except Exception as error:
                print("AI inference error:", error)
                time.sleep(0.1)
                continue

            fire_detected = False
            best_label = ""
            best_confidence = 0.0
            best_box = None

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id].lower()

                    is_target = (
                        "fire" in class_name
                        or "smoke" in class_name
                    )

                    if (
                        confidence >= CONFIDENCE_LIMIT
                        and is_target
                        and confidence > best_confidence
                    ):
                        fire_detected = True
                        best_label = class_name
                        best_confidence = confidence
                        best_box = box

            if fire_detected and best_box is not None:
                x1, y1, x2, y2 = best_box.xyxy[0]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"{best_label.upper()} "
                    f"{best_confidence:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            fire_confirmed = False

            if fire_detected:
                if fire_start_time is None:
                    fire_start_time = time.time()

                elapsed = time.time() - fire_start_time

                if elapsed >= ALERT_HOLD_TIME:
                    fire_confirmed = True

                    cv2.putText(
                        display_frame,
                        "CONFIRMED FIRE/SMOKE ALERT",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

                    current_time = time.time()

                    if (
                        current_time - last_alert_time
                        >= ALERT_COOLDOWN
                    ):
                        print(
                            "CONFIRMED ALERT:",
                            best_label,
                            f"confidence={best_confidence:.2f}",
                        )

                        send_alert_to_dashboard(
                            best_label,
                            best_confidence,
                            display_frame,
                            gps,
                        )
                        last_alert_time = current_time

                else:
                    cv2.putText(
                        display_frame,
                        f"Checking... {elapsed:.1f}s",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3,
                    )
            else:
                fire_start_time = None

                cv2.putText(
                    display_frame,
                    "No Fire/Smoke",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )

            now = time.time()

            if (
                now - last_telemetry_time
                >= TELEMETRY_SEND_INTERVAL
            ):
                ok, error = send_live_telemetry(
                    gps,
                    best_label,
                    best_confidence,
                    fire_detected,
                    fire_confirmed,
                )

                if (
                    not ok
                    and now - last_dashboard_error_print > 5
                ):
                    print(
                        "Telemetry API not reachable:",
                        error,
                    )
                    last_dashboard_error_print = now

                last_telemetry_time = now

            gps_status = (
                "GPS OK"
                if gps["gps_ok"]
                else "GPS WAIT"
            )

            cv2.putText(
                display_frame,
                (
                    f"{gps_status}: "
                    f"{gps['latitude']:.7f}, "
                    f"{gps['longitude']:.7f}, "
                    f"Alt: {gps['altitude']:.2f}m"
                ),
                (30, DISPLAY_HEIGHT - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                (
                    f"Threshold: {CONFIDENCE_LIMIT}, "
                    f"Hold: {ALERT_HOLD_TIME}s, "
                    f"Input: {VIDEO_SOURCE.upper()}"
                ),
                (30, DISPLAY_HEIGHT - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            if SHOW_WINDOW:
                cv2.imshow(
                    "AI Fire Detection - Windows",
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("Stopping by keyboard interrupt...")

    finally:
        video_reader.stop()
        HTTP_SESSION.close()
        cv2.destroyAllWindows()
        print("AI Fire/Smoke Detection stopped.")


if __name__ == "__main__":
    main()
