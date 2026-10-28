from flask import Flask, request, jsonify, redirect, send_file
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import sqlite3
import os
import threading
import time

try:
    from pymavlink import mavutil
except Exception:
    mavutil = None

# ============================================================
# DRONE FIRE DETECTION DASHBOARD SERVER
# Flask + SQLite + API + Advanced UI + Telemetry + Command Handlers
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("FIRE_ALERT_DB", BASE_DIR / "fire_alerts.db"))
UI_FILE = Path(os.getenv("DASHBOARD_UI_FILE", BASE_DIR / "drone_dashboard_integrated.html"))

# Mission Planner SITL connection used by your project.
MAVLINK_CONNECTION_STRING = os.getenv("MAVLINK_CONNECTION_STRING", "udp:127.0.0.1:3000")

# Keep enabled for SITL testing. For real drone safety, set ENABLE_DRONE_COMMANDS=0.
ENABLE_DRONE_COMMANDS = os.getenv("ENABLE_DRONE_COMMANDS", "1") == "1"

app = Flask(__name__)
CORS(app)

telemetry_lock = threading.Lock()
telemetry_state = {
    "drone_id": "DRONE_001",
    "latitude": 0.0,
    "longitude": 0.0,
    "altitude": 0.0,
    "gps_ok": False,
    "latest_label": "--",
    "latest_confidence": 0.0,
    "fire_detected": False,
    "fire_confirmed": False,
    "mavlink_connection": MAVLINK_CONNECTION_STRING,
    "timestamp": None,
    "source": "dashboard_server"
}

command_log = []
command_lock = threading.Lock()
mav_master = None
mav_lock = threading.Lock()


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fire_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drone_id TEXT,
            event_type TEXT,
            label TEXT,
            confidence REAL,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            timestamp TEXT,
            status TEXT,
            image_base64 TEXT
        )
    """)

    conn.commit()
    conn.close()


def normalize_status_class(status):
    if status == "verified":
        return "verified"
    if status == "false_alert":
        return "false-alert"
    if status == "closed":
        return "closed"
    return "new"


def row_to_alert(row):
    return {
        "id": row["id"],
        "drone_id": row["drone_id"],
        "event_type": row["event_type"],
        "label": row["label"],
        "confidence": row["confidence"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "altitude": row["altitude"],
        "timestamp": row["timestamp"],
        "status": row["status"],
        "status_class": normalize_status_class(row["status"]),
        "image": row["image_base64"]
    }


def save_alert_to_db(alert):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO fire_alerts (
            drone_id, event_type, label, confidence, latitude, longitude,
            altitude, timestamp, status, image_base64
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert["drone_id"],
        alert["event_type"],
        alert["label"],
        alert["confidence"],
        alert["latitude"],
        alert["longitude"],
        alert["altitude"],
        alert["timestamp"],
        alert["status"],
        alert["image"]
    ))

    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return alert_id


def get_all_alerts():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM fire_alerts ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [row_to_alert(row) for row in rows]


def update_alert_status(alert_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE fire_alerts SET status = ? WHERE id = ?", (status, alert_id))
    changed = cursor.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def delete_alert_from_db(alert_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fire_alerts WHERE id = ?", (alert_id,))
    changed = cursor.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def delete_all_alerts_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM fire_alerts")
    cursor.execute("DELETE FROM sqlite_sequence WHERE name='fire_alerts'")
    conn.commit()
    conn.close()


# ============================================================
# TELEMETRY + COMMAND HELPERS
# ============================================================

def add_command_log(command, status, message):
    entry = {
        "time": datetime.now().isoformat(),
        "command": command,
        "status": status,
        "message": message
    }
    with command_lock:
        command_log.insert(0, entry)
        del command_log[100:]
    print("[COMMAND]", entry)
    return entry


def update_telemetry_state(data):
    global telemetry_state
    with telemetry_lock:
        telemetry_state.update({
            "drone_id": data.get("drone_id", telemetry_state.get("drone_id", "DRONE_001")),
            "latitude": float(data.get("latitude", telemetry_state.get("latitude", 0.0))),
            "longitude": float(data.get("longitude", telemetry_state.get("longitude", 0.0))),
            "altitude": float(data.get("altitude", telemetry_state.get("altitude", 0.0))),
            "gps_ok": bool(data.get("gps_ok", telemetry_state.get("gps_ok", False))),
            "latest_label": data.get("latest_label", telemetry_state.get("latest_label", "--")),
            "latest_confidence": float(data.get("latest_confidence", telemetry_state.get("latest_confidence", 0.0))),
            "fire_detected": bool(data.get("fire_detected", telemetry_state.get("fire_detected", False))),
            "fire_confirmed": bool(data.get("fire_confirmed", telemetry_state.get("fire_confirmed", False))),
            "mavlink_connection": data.get("mavlink_connection", MAVLINK_CONNECTION_STRING),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "source": data.get("source", "ai_fire_detect_windows")
        })


def get_mavlink_master():
    global mav_master

    if not ENABLE_DRONE_COMMANDS:
        return None, "Drone command sending is disabled. Set ENABLE_DRONE_COMMANDS=1 to enable."

    if mavutil is None:
        return None, "pymavlink is not installed. Run: pip install pymavlink"

    with mav_lock:
        if mav_master is not None:
            return mav_master, "connected"

        try:
            print("Connecting command MAVLink:", MAVLINK_CONNECTION_STRING)
            mav_master = mavutil.mavlink_connection(MAVLINK_CONNECTION_STRING)
            mav_master.wait_heartbeat(timeout=10)
            print("Command MAVLink heartbeat received")
            return mav_master, "connected"
        except Exception as e:
            mav_master = None
            return None, f"MAVLink command connection failed: {e}"


def set_mode(master, mode_name):
    mode_mapping = master.mode_mapping()
    if mode_name not in mode_mapping:
        raise RuntimeError(f"Mode {mode_name} is not available. Available: {list(mode_mapping.keys())}")

    mode_id = mode_mapping[mode_name]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )


def arm_vehicle(master):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0
    )


def send_takeoff(master, altitude=10.0):
    set_mode(master, "GUIDED")
    time.sleep(0.5)
    arm_vehicle(master)
    time.sleep(0.5)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0,
        float(altitude)
    )


def send_drone_command(command, params=None):
    params = params or {}
    master, message = get_mavlink_master()

    if master is None:
        return False, message

    try:
        if command == "takeoff":
            send_takeoff(master, float(params.get("altitude", 10.0)))
            return True, "Takeoff command sent"

        if command == "land":
            set_mode(master, "LAND")
            return True, "LAND mode sent"

        if command == "rtl":
            set_mode(master, "RTL")
            return True, "RTL mode sent"

        if command == "loiter":
            set_mode(master, "LOITER")
            return True, "LOITER mode sent"

        if command == "emergency_stop":
            # SITL/emergency only. For real drone, use proper safety policy.
            master.mav.command_long_send(
                master.target_system,
                master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0, 21196, 0, 0, 0, 0, 0
            )
            return True, "Emergency disarm command sent"

        return False, f"Unknown command: {command}"

    except Exception as e:
        return False, f"Command failed: {e}"


# ============================================================
# PAGE ROUTES
# ============================================================

@app.route("/")
def dashboard_page():
    if UI_FILE.exists():
        return send_file(UI_FILE)

    return f"""
    <h1>Dashboard UI file not found</h1>
    <p>Expected file: {UI_FILE}</p>
    <p>API is still running. Try /api/alerts or /api/telemetry</p>
    """, 404


# ============================================================
# API ROUTES
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "database": str(DB_PATH),
        "ui_file": str(UI_FILE),
        "ui_file_exists": UI_FILE.exists(),
        "alerts_count": len(get_all_alerts()),
        "commands_enabled": ENABLE_DRONE_COMMANDS,
        "mavlink_connection": MAVLINK_CONNECTION_STRING
    })


@app.route("/api/telemetry", methods=["GET"])
def api_get_telemetry():
    with telemetry_lock:
        return jsonify(dict(telemetry_state))


@app.route("/api/telemetry", methods=["POST"])
def api_post_telemetry():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    update_telemetry_state(data)
    with telemetry_lock:
        return jsonify({"message": "Telemetry updated", "telemetry": dict(telemetry_state)}), 200


@app.route("/api/drone-command", methods=["POST"])
def api_drone_command():
    data = request.get_json(silent=True) or {}
    command = data.get("command")
    params = data.get("params", {})

    allowed = {"takeoff", "land", "rtl", "loiter", "emergency_stop"}
    if command not in allowed:
        return jsonify({"error": "Invalid command", "allowed": sorted(allowed)}), 400

    ok, message = send_drone_command(command, params)
    entry = add_command_log(command, "sent" if ok else "failed", message)
    return jsonify({"ok": ok, "message": message, "log": entry}), 200 if ok else 500


@app.route("/api/command-log", methods=["GET"])
def api_command_log():
    with command_lock:
        return jsonify(list(command_log))


@app.route("/api/fire-alert", methods=["POST"])
def receive_fire_alert():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    alert = {
        "drone_id": data.get("drone_id", "UNKNOWN"),
        "event_type": data.get("event_type", "unknown_event"),
        "label": data.get("label", "unknown"),
        "confidence": data.get("confidence", 0),
        "latitude": data.get("latitude", 0),
        "longitude": data.get("longitude", 0),
        "altitude": data.get("altitude", 0),
        "timestamp": data.get("timestamp", datetime.now().isoformat()),
        "status": data.get("status", "new"),
        "image": data.get("image", None)
    }

    alert_id = save_alert_to_db(alert)

    # Also update live telemetry from confirmed alert.
    update_telemetry_state({
        "drone_id": alert["drone_id"],
        "latitude": alert["latitude"],
        "longitude": alert["longitude"],
        "altitude": alert["altitude"],
        "gps_ok": True,
        "latest_label": alert["label"],
        "latest_confidence": alert["confidence"],
        "fire_detected": True,
        "fire_confirmed": True,
        "timestamp": alert["timestamp"],
        "source": "fire_alert"
    })

    print("\n========== NEW FIRE ALERT SAVED TO SQLITE ==========")
    print({
        "id": alert_id,
        "drone_id": alert["drone_id"],
        "event_type": alert["event_type"],
        "label": alert["label"],
        "confidence": alert["confidence"],
        "latitude": alert["latitude"],
        "longitude": alert["longitude"],
        "altitude": alert["altitude"],
        "timestamp": alert["timestamp"],
        "status": alert["status"],
        "image_received": alert["image"] is not None
    })
    print("===================================================\n")

    return jsonify({
        "message": "Fire alert saved to SQLite successfully",
        "id": alert_id,
        "alert": {
            "id": alert_id,
            "drone_id": alert["drone_id"],
            "event_type": alert["event_type"],
            "label": alert["label"],
            "confidence": alert["confidence"],
            "latitude": alert["latitude"],
            "longitude": alert["longitude"],
            "altitude": alert["altitude"],
            "timestamp": alert["timestamp"],
            "status": alert["status"],
            "image_received": alert["image"] is not None
        }
    }), 201


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    return jsonify(get_all_alerts())


@app.route("/api/alerts/<int:alert_id>/status", methods=["POST"])
def api_update_status(alert_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")

    allowed = {"new", "verified", "false_alert", "closed"}
    if status not in allowed:
        return jsonify({"error": "Invalid status", "allowed": sorted(allowed)}), 400

    changed = update_alert_status(alert_id, status)
    if not changed:
        return jsonify({"error": "Alert not found"}), 404

    return jsonify({"message": "Status updated", "id": alert_id, "status": status})


@app.route("/api/alerts/<int:alert_id>", methods=["DELETE"])
def api_delete_alert(alert_id):
    changed = delete_alert_from_db(alert_id)
    if not changed:
        return jsonify({"error": "Alert not found"}), 404

    return jsonify({"message": "Alert deleted", "id": alert_id})


@app.route("/api/alerts", methods=["DELETE"])
def api_delete_all_alerts():
    delete_all_alerts_from_db()
    return jsonify({"message": "All alerts deleted"})


# Simple browser action routes for backward compatibility.
@app.route("/verify/<int:alert_id>")
def verify_alert(alert_id):
    update_alert_status(alert_id, "verified")
    return redirect("/")


@app.route("/false/<int:alert_id>")
def false_alert(alert_id):
    update_alert_status(alert_id, "false_alert")
    return redirect("/")


@app.route("/close/<int:alert_id>")
def close_alert(alert_id):
    update_alert_status(alert_id, "closed")
    return redirect("/")


@app.route("/delete/<int:alert_id>")
def delete_alert(alert_id):
    delete_alert_from_db(alert_id)
    return redirect("/")


@app.route("/delete-all")
def delete_all_alerts():
    delete_all_alerts_from_db()
    return redirect("/")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    init_db()

    print("Starting Drone Fire Detection Dashboard...")
    print("Database:", DB_PATH)
    print("UI file:", UI_FILE)
    print("MAVLink:", MAVLINK_CONNECTION_STRING)
    print("Commands enabled:", ENABLE_DRONE_COMMANDS)
    print("Open browser: http://127.0.0.1:5000")

    app.run(host="0.0.0.0", port=5000, debug=True)
