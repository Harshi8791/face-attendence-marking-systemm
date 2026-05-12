import sys
import os
import cv2
import numpy as np
import threading
import time
import sqlite3
import pickle
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration & Constants
#  All paths are absolute and relative to the script location —
#  so the system works correctly regardless of CWD.
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "attendance_data")
FACES_DIR    = os.path.join(DATA_DIR, "faces")
RECORDS_DIR  = os.path.join(DATA_DIR, "records")
DB_FILE      = os.path.join(DATA_DIR, "attendance.db")
CACHE_FILE   = os.path.join(DATA_DIR, "encodings.pkl")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
LOG_FILE     = os.path.join(DATA_DIR, "attendance_system.log")

# Legacy paths (for one-time migration only)
LEGACY_CFG_FILE = os.path.join(DATA_DIR, "config.json")
LEGACY_CSV      = os.path.join(DATA_DIR, "attendance.csv")
ATTENDANCE_CSV  = os.path.join(RECORDS_DIR, "attendance.csv")

# Ensure required directories exist
for _d in (DATA_DIR, FACES_DIR, RECORDS_DIR):
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging Setup  (file + console)
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("AttendanceSystem")

# ─────────────────────────────────────────────────────────────────────────────
#  Optional face_recognition dependency
# ─────────────────────────────────────────────────────────────────────────────
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    logger.warning("'face_recognition' not found — face recognition disabled.")
    logger.warning("Install with:  pip install face_recognition")

# ─────────────────────────────────────────────────────────────────────────────
#  Config Manager  (reads / writes settings.json)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_SETTINGS = {
    "camera_index"        : 0,
    "recognition_tolerance": 0.50,
    "late_threshold"      : "09:00",   # HH:MM  — records after this are "Late"
    "kiosk_cooldown_sec"  : 60,        # seconds before re-checking same person
}

class ConfigManager:
    def __init__(self):
        self._settings = dict(_DEFAULT_SETTINGS)
        self._load()

    def _load(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._settings.update(saved)
                logger.info("Settings loaded from %s", SETTINGS_FILE)
            except Exception as e:
                logger.warning("Could not read settings.json: %s — using defaults.", e)

    def save(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except Exception as e:
            logger.error("Could not save settings: %s", e)

    def get(self, key):
        return self._settings.get(key, _DEFAULT_SETTINGS.get(key))

    def set(self, key, value):
        self._settings[key] = value
        self.save()

    def show(self):
        print("\n--- ⚙  SETTINGS ---")
        for k, v in self._settings.items():
            print(f"  {k:<30} : {v}")

# ─────────────────────────────────────────────────────────────────────────────
#  Database Manager
# ─────────────────────────────────────────────────────────────────────────────
class DatabaseManager:
    """
    Thread-safe SQLite wrapper.

    Key guarantees:
      • A `threading.Lock` wraps every cursor operation — safe for
        multi-threaded use (camera thread + main thread).
      • All CSV rebuilds use  ORDER BY datetime(timestamp) DESC
        so the NEWEST record is always the FIRST row in the CSV file.
      • The attendance table includes a `status` column (On Time / Late).
    """

    CSV_HEADER = ["Name", "Date", "Time", "Timestamp", "Status"]

    def __init__(self, db_path, config: ConfigManager):
        self.db_path = db_path
        self.config  = config
        self._lock   = threading.Lock()
        # check_same_thread=False + explicit lock = correct multi-thread access
        self.conn   = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._sync_csv_on_startup()

    # ── Schema ────────────────────────────────────────────────────────────────
    def _create_tables(self):
        with self._lock:
            self.cursor.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT    UNIQUE NOT NULL,
                    photo_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS attendance (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   INTEGER,
                    name      TEXT,
                    timestamp TIMESTAMP,
                    date      TEXT,
                    time      TEXT,
                    status    TEXT DEFAULT "On Time",
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
            ''')
            # Add status column to existing DBs (idempotent)
            try:
                self.cursor.execute("ALTER TABLE attendance ADD COLUMN status TEXT DEFAULT 'On Time'")
            except sqlite3.OperationalError:
                pass  # column already exists
            self.conn.commit()

    # ── CSV helpers ───────────────────────────────────────────────────────────
    def _compute_status(self, time_str: str) -> str:
        """Return 'On Time' or 'Late' based on the configured late_threshold."""
        try:
            threshold = datetime.strptime(self.config.get("late_threshold"), "%H:%M").time()
            recorded  = datetime.strptime(time_str, "%H:%M:%S").time()
            return "On Time" if recorded <= threshold else "Late"
        except Exception:
            return "On Time"

    def _query_all_records(self):
        """Fetch all attendance rows ordered newest-first (reliable datetime cast)."""
        self.cursor.execute(
            "SELECT name, date, time, timestamp, status "
            "FROM attendance "
            "ORDER BY datetime(timestamp) DESC"
        )
        return self.cursor.fetchall()

    def _write_csv(self, records):
        """Overwrite ATTENDANCE_CSV with header + records (newest first)."""
        with open(ATTENDANCE_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)
            writer.writerows(records)

    def _sync_csv_on_startup(self):
        """Full CSV rebuild on every startup — keeps CSV always in sync with DB."""
        try:
            with self._lock:
                records = self._query_all_records()
            self._write_csv(records)
            if records:
                logger.info("CSV synced on startup: %d records → %s", len(records), ATTENDANCE_CSV)
            else:
                logger.info("CSV initialised (empty) at %s", ATTENDANCE_CSV)
        except Exception as e:
            logger.error("CSV startup sync failed: %s", e)

    def _rebuild_csv(self):
        """Full CSV rebuild (called after every write operation)."""
        try:
            records = self._query_all_records()
            self._write_csv(records)
        except Exception as e:
            logger.error("CSV rebuild failed: %s", e)

    # ── User CRUD ─────────────────────────────────────────────────────────────
    def add_user(self, name, photo_path):
        try:
            with self._lock:
                self.cursor.execute(
                    "INSERT INTO users (name, photo_path) VALUES (?, ?)", (name, photo_path)
                )
                self.conn.commit()
            logger.info("User added: %s", name)
            return True, "User added successfully"
        except sqlite3.IntegrityError:
            return False, "User already exists"

    def get_users(self):
        with self._lock:
            self.cursor.execute("SELECT name, photo_path, id FROM users")
            return self.cursor.fetchall()

    def delete_user(self, name):
        try:
            with self._lock:
                self.cursor.execute("SELECT photo_path FROM users WHERE name = ?", (name,))
                result = self.cursor.fetchone()
                photo_path = result[0] if result else None
                self.cursor.execute("DELETE FROM users WHERE name = ?", (name,))
                self.cursor.execute("DELETE FROM attendance WHERE name = ?", (name,))
                self.conn.commit()
            self._rebuild_csv()
            if photo_path and os.path.exists(photo_path):
                os.remove(photo_path)
            logger.info("User deleted: %s", name)
            return True, "User deleted successfully"
        except Exception as e:
            logger.error("delete_user error: %s", e)
            return False, f"Error: {e}"

    def edit_user(self, old_name, new_name):
        try:
            with self._lock:
                self.cursor.execute("SELECT photo_path FROM users WHERE name = ?", (old_name,))
                result = self.cursor.fetchone()
                if not result:
                    return False, "User not found"
                old_photo_path = result[0]

                self.cursor.execute("SELECT id FROM users WHERE name = ?", (new_name,))
                if self.cursor.fetchone():
                    return False, "New name already exists"

                new_photo_path = old_photo_path
                if old_photo_path and os.path.exists(old_photo_path):
                    ext           = os.path.splitext(old_photo_path)[1]
                    old_filename  = os.path.basename(old_photo_path)
                    ts_part       = old_filename.split("_", 1)[1] if "_" in old_filename else f"renamed{ext}"
                    new_filename  = f"{new_name}_{ts_part}"
                    new_photo_path = os.path.join(os.path.dirname(old_photo_path), new_filename)
                    os.rename(old_photo_path, new_photo_path)

                self.cursor.execute(
                    "UPDATE users SET name = ?, photo_path = ? WHERE name = ?",
                    (new_name, new_photo_path, old_name),
                )
                self.cursor.execute(
                    "UPDATE attendance SET name = ? WHERE name = ?", (new_name, old_name)
                )
                self.conn.commit()
            self._rebuild_csv()
            logger.info("User renamed: %s → %s", old_name, new_name)
            return True, "User renamed successfully"
        except Exception as e:
            logger.error("edit_user error: %s", e)
            return False, f"Error: {e}"

    # ── Attendance CRUD ───────────────────────────────────────────────────────
    def log_attendance(self, name):
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            self.cursor.execute(
                "SELECT id FROM attendance WHERE name = ? AND date = ?", (name, today)
            )
            if self.cursor.fetchone():
                return False, "Already marked today"

            now       = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            time_str  = now.strftime("%H:%M:%S")
            status    = self._compute_status(time_str)

            self.cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            res     = self.cursor.fetchone()
            user_id = res[0] if res else None

            self.cursor.execute(
                "INSERT INTO attendance (user_id, name, timestamp, date, time, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name, timestamp, today, time_str, status),
            )
            self.conn.commit()

        # Rebuild CSV outside the lock (file I/O doesn't need the DB lock)
        self._rebuild_csv()
        logger.info("Attendance logged: %s at %s [%s]", name, time_str, status)
        return True, f"Marked {name} at {time_str} [{status}]"

    def get_attendance(self, date=None, name=None):
        query  = "SELECT name, date, time, timestamp, status FROM attendance"
        params, conditions = [], []
        if date:
            conditions.append("date = ?");  params.append(date)
        if name:
            conditions.append("name LIKE ?"); params.append(f"%{name}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY datetime(timestamp) DESC"
        with self._lock:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()

    def get_attendance_with_id(self, date=None, name=None):
        query  = "SELECT id, name, date, time, timestamp, status FROM attendance"
        params, conditions = [], []
        if date:
            conditions.append("date = ?");  params.append(date)
        if name:
            conditions.append("name LIKE ?"); params.append(f"%{name}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY datetime(timestamp) DESC"
        with self._lock:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()

    def delete_attendance_record(self, record_id):
        try:
            with self._lock:
                self.cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
                self.conn.commit()
            self._rebuild_csv()
            logger.info("Attendance record %d deleted.", record_id)
            
            return True, "Record deleted successfully"
        except Exception as e:
            logger.error("delete_attendance_record error: %s", e)
            return False, f"Error: {e}"

    def edit_attendance_record(self, record_id, new_name=None, new_date=None, new_time=None):
        try:
            with self._lock:
                self.cursor.execute(
                    "SELECT name, date, time FROM attendance WHERE id = ?", (record_id,)
                )
                result = self.cursor.fetchone()
                if not result:
                    return False, "Record not found"
                cur_name, cur_date, cur_time = result
                final_name = new_name or cur_name
                final_date = new_date or cur_date
                final_time = new_time or cur_time
                final_ts   = f"{final_date} {final_time}"
                status     = self._compute_status(final_time)
                self.cursor.execute(
                    "UPDATE attendance SET name=?, date=?, time=?, timestamp=?, status=? WHERE id=?",
                    (final_name, final_date, final_time, final_ts, status, record_id),
                )
                self.conn.commit()
            self._rebuild_csv()
            logger.info("Attendance record %d updated.", record_id)
            return True, "Record updated successfully"
        except Exception as e:
            logger.error("edit_attendance_record error: %s", e)
            return False, f"Error: {e}"

    # ── Export ────────────────────────────────────────────────────────────────
    def export_to_csv(self, filename=None, start_date=None, end_date=None):
        """
        Export attendance to CSV.

        Parameters
        ----------
        filename   : optional output filename (auto-generated if None).
                     Always saved inside RECORDS_DIR for consistent location.
        start_date : optional filter  "YYYY-MM-DD"
        end_date   : optional filter  "YYYY-MM-DD"
        """
        if not filename:
            filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Ensure filename is inside RECORDS_DIR (strip any path the user typed)
        filename = os.path.join(RECORDS_DIR, os.path.basename(filename))

        query  = "SELECT name, date, time, timestamp, status FROM attendance"
        params, conditions = [], []
        if start_date:
            conditions.append("date >= ?"); params.append(start_date)
        if end_date:
            conditions.append("date <= ?"); params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY datetime(timestamp) DESC"

        try:
            with self._lock:
                self.cursor.execute(query, params)
                records = self.cursor.fetchall()
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADER)
                writer.writerows(records)
            logger.info("Exported %d records to %s", len(records), filename)
            return True, f"Exported {len(records)} records to:\n  {filename}"
        except Exception as e:
            logger.error("export_to_csv error: %s", e)
            return False, f"Export failed: {e}"

    # ── Statistics ────────────────────────────────────────────────────────────
    def get_statistics(self):
        """Return per-person statistics: (name, total_days, on_time, late, last_seen)."""
        with self._lock:
            self.cursor.execute("""
                SELECT
                    name,
                    COUNT(*)                                        AS total_days,
                    SUM(CASE WHEN status = 'On Time' THEN 1 ELSE 0 END) AS on_time,
                    SUM(CASE WHEN status = 'Late'    THEN 1 ELSE 0 END) AS late,
                    MAX(date)                                       AS last_seen
                FROM attendance
                GROUP BY name
                ORDER BY total_days DESC
            """)
            return self.cursor.fetchall()

    def get_total_working_days(self):
        """Unique dates in the attendance table (used to compute attendance %)."""
        with self._lock:
            self.cursor.execute("SELECT COUNT(DISTINCT date) FROM attendance")
            result = self.cursor.fetchone()
            return result[0] if result else 0


# ─────────────────────────────────────────────────────────────────────────────
#  Face Engine  (with disk cache)
# ─────────────────────────────────────────────────────────────────────────────
class FaceEngine:
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        self.db     = db_manager
        self.config = config
        self.known_encodings = []
        self.known_names     = []
        self.cache  = self._load_cache()
        self.reload_faces()

    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error("Face cache save failed: %s", e)

    def reload_faces(self):
        if not FACE_REC_AVAILABLE:
            return
        users = self.db.get_users()
        self.known_encodings = []
        self.known_names     = []
        cache_updated        = False

        logger.info("Loading face database (%d user(s))…", len(users))
        for name, photo_path, _ in users:
            if not photo_path or not os.path.exists(photo_path):
                logger.warning("Photo missing for %s — skipped.", name)
                continue

            file_mtime  = os.path.getmtime(photo_path)
            cached_data = self.cache.get(name)

            if cached_data and cached_data.get("mtime") == file_mtime:
                self.known_encodings.append(cached_data["encoding"])
                self.known_names.append(name)
            else:
                try:
                    logger.info("  Computing encoding for %s …", name)
                    image     = face_recognition.load_image_file(photo_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_names.append(name)
                        self.cache[name] = {"encoding": encodings[0], "mtime": file_mtime}
                        cache_updated = True
                    else:
                        logger.warning("  No face detected in %s — skipped.", photo_path)
                except Exception as e:
                    logger.error("  Error processing %s: %s", photo_path, e)

        if cache_updated:
            self._save_cache()

        logger.info("✓ Face database ready: %d face(s) loaded.", len(self.known_names))

    def identify(self, frame):
        tolerance = self.config.get("recognition_tolerance")
        if not FACE_REC_AVAILABLE or not self.known_encodings:
            return []

        small_frame    = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small      = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        results = []
        for face_enc, face_loc in zip(face_encodings, face_locations):
            name       = "Unknown"
            confidence = 0.0
            distances  = face_recognition.face_distance(self.known_encodings, face_enc)
            if len(distances) > 0:
                best = int(np.argmin(distances))
                if distances[best] < tolerance:
                    name       = self.known_names[best]
                    confidence = 1.0 - distances[best]

            top, right, bottom, left = face_loc
            results.append({
                "name"      : name,
                "confidence": confidence,
                "box"       : (left * 4, top * 4, right * 4, bottom * 4),
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  Camera Thread
# ─────────────────────────────────────────────────────────────────────────────
class CameraStream:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera (index {src}). "
                               "Check your camera_index in settings.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.capture.set(cv2.CAP_PROP_FPS,            30)

        self._queue   = Queue(maxsize=2)
        self._stopped = False
        self._thread  = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while not self._stopped:
            if not self.capture.isOpened():
                self._stopped = True
                break
            ret, frame = self.capture.read()
            if not ret:
                self._stopped = True
                break
            # Discard stale frame to keep queue fresh
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)

    def read(self):
        try:
            return self._queue.get(timeout=0.5)
        except Empty:
            return None

    @property
    def stopped(self):
        return self._stopped

    def stop(self):
        self._stopped = True
        self.capture.release()


# ─────────────────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedAttendanceSystem:
    def __init__(self):
        self.config = ConfigManager()
        self.db     = DatabaseManager(DB_FILE, self.config)
        self._migrate_legacy_data()
        self.engine = FaceEngine(self.db, self.config)

    # ── Legacy Migration ──────────────────────────────────────────────────────
    def _migrate_legacy_data(self):
        # 1. Migrate users from config.json
        if os.path.exists(LEGACY_CFG_FILE):
            logger.info("Found legacy config.json — migrating users…")
            try:
                with open(LEGACY_CFG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                count = 0
                for name, data in config.items():
                    added, _ = self.db.add_user(name, data.get("photo_path", ""))
                    if added:
                        count += 1
                logger.info("Migrated %d user(s) from config.json.", count)
                os.rename(LEGACY_CFG_FILE, LEGACY_CFG_FILE + ".migrated")
            except Exception as e:
                logger.error("User migration failed: %s", e)

        # 2. Migrate attendance from legacy CSV
        if os.path.exists(LEGACY_CSV):
            logger.info("Found legacy attendance.csv — migrating records…")
            try:
                with open(LEGACY_CSV, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    # Normalise headers to lowercase for safe lookup
                    count = 0
                    for row in reader:
                        r = {k.lower(): v for k, v in row.items()}
                        with self.db._lock:
                            self.db.cursor.execute(
                                "INSERT INTO attendance (name, timestamp, date, time, status) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (
                                    r.get("name"),
                                    r.get("timestamp"),
                                    r.get("date"),
                                    r.get("time"),
                                    r.get("status", "On Time"),
                                ),
                            )
                            count += 1
                    self.db.conn.commit()
                self.db._rebuild_csv()
                logger.info("Migrated %d attendance record(s) from legacy CSV.", count)
                os.rename(LEGACY_CSV, LEGACY_CSV + ".migrated")
            except Exception as e:
                logger.error("Attendance migration failed: %s", e)

    # ── Add Person ────────────────────────────────────────────────────────────
    def add_person(self):
        print("\n--- 👤 ADD NEW PERSON ---")
        name = input("Enter Name: ").strip()
        if not name:
            return

        path = input("Enter Photo Path: ").strip().strip('"')
        if not os.path.exists(path):
            print("❌ File not found.")
            return

        ext         = os.path.splitext(path)[1]
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path    = os.path.join(FACES_DIR, f"{name}_{ts}{ext}")

        try:
            import shutil
            shutil.copy2(path, new_path)
            success, msg = self.db.add_user(name, new_path)
            if success:
                print(f"✓ {msg}")
                self.engine.reload_faces()
            else:
                print(f"❌ {msg}")
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error("add_person error: %s", e)

    # ── View Records ──────────────────────────────────────────────────────────
    def view_records(self):
        print("\n--- 📊 ATTENDANCE RECORDS ---")
        filter_date = input("Filter by date (YYYY-MM-DD) or press Enter for all: ").strip() or None
        filter_name = input("Filter by name or press Enter for all: ").strip() or None
        records = self.db.get_attendance(date=filter_date, name=filter_name)

        if not records:
            print("❌ No records found.")
            return

        print(f"\n{'#':<5} {'Name':<22} {'Date':<12} {'Time':<10} {'Status':<10}")
        print("─" * 62)
        for idx, (name, date, time_val, _, status) in enumerate(records[:100], 1):
            status_icon = "✅" if status == "On Time" else "⏰"
            print(f"{idx:<5} {name:<22} {date:<12} {time_val:<10} {status_icon} {status}")
        if len(records) > 100:
            print(f"  … and {len(records) - 100} more records.")

    # ── List People ───────────────────────────────────────────────────────────
    def list_people(self):
        print("\n--- 👥 REGISTERED PEOPLE ---")
        users = self.db.get_users()
        if not users:
            print("❌ No people registered yet.")
            return
        print(f"{'#':<4} {'Name':<25} {'Photo Path'}")
        print("─" * 70)
        for idx, (name, photo_path, _) in enumerate(users, 1):
            exists = "✓" if photo_path and os.path.exists(photo_path) else "✗"
            print(f"{idx:<4} {name:<25} [{exists}] {photo_path}")
        print(f"\nTotal: {len(users)} registered people")

    # ── Manage People ──────────────────────────────────────────────────────────
    def manage_people(self):
        self.list_people()
        print("\n--- 🔧 MANAGE PEOPLE ---")
        print("1. Edit Person Name")
        print("2. Delete Person")
        print("3. Back")
        choice = input("\nSelect > ").strip()

        if choice == "1":
            old_name = input("Current name: ").strip()
            new_name = input("New name: ").strip()
            if old_name and new_name:
                success, msg = self.db.edit_user(old_name, new_name)
                print(f"{'✓' if success else '❌'} {msg}")
                if success:
                    if old_name in self.engine.cache:
                        self.engine.cache[new_name] = self.engine.cache.pop(old_name)
                        self.engine._save_cache()
                    self.engine.reload_faces()
            else:
                print("❌ Names cannot be empty.")

        elif choice == "2":
            name = input("Name to delete: ").strip()
            if name:
                confirm = input(f"⚠  Delete '{name}' and ALL their records? (yes/no): ").strip().lower()
                if confirm == "yes":
                    success, msg = self.db.delete_user(name)
                    print(f"{'✓' if success else '❌'} {msg}")
                    if success:
                        self.engine.cache.pop(name, None)
                        self.engine._save_cache()
                        self.engine.reload_faces()
                else:
                    print("❌ Cancelled.")

    # ── Manage Records ────────────────────────────────────────────────────────
    def manage_records(self):
        print("\n--- 📝 MANAGE ATTENDANCE RECORDS ---")
        records = self.db.get_attendance_with_id()
        if not records:
            print("❌ No records found.")
            return

        print(f"\n{'#':<5} {'Name':<22} {'Date':<12} {'Time':<10} {'Status'}")
        print("─" * 65)
        for idx, (rid, name, date, time_val, _, status) in enumerate(records[:50], 1):
            print(f"{idx:<5} {name:<22} {date:<12} {time_val:<10} {status}")

        total = len(records)
        shown = min(total, 50)
        print(f"\nShowing {shown} of {total} records (newest first)")

        print("\n1. Edit Record\n2. Delete Record\n3. Back")
        choice = input("\nSelect > ").strip()

        if choice == "1":
            try:
                num = int(input("Record # to edit: "))
                if 1 <= num <= shown:
                    rid, cur_name, cur_date, cur_time = records[num - 1][:4]
                    print(f"\nCurrent: {cur_name} | {cur_date} | {cur_time}")
                    print("Leave blank to keep current.")
                    new_name = input(f"New name [{cur_name}]: ").strip()
                    new_date = input(f"New date (YYYY-MM-DD) [{cur_date}]: ").strip()
                    new_time = input(f"New time (HH:MM:SS) [{cur_time}]: ").strip()
                    success, msg = self.db.edit_attendance_record(
                        rid,
                        new_name or None,
                        new_date or None,
                        new_time or None,
                    )
                    print(f"{'✓' if success else '❌'} {msg}")
                else:
                    print("❌ Invalid record number.")
            except ValueError:
                print("❌ Invalid input.")

        elif choice == "2":
            try:
                num = int(input("Record # to delete: "))
                if 1 <= num <= shown:
                    rid, r_name, r_date = records[num - 1][:3]
                    confirm = input(f"⚠  Delete record for {r_name} on {r_date}? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        success, msg = self.db.delete_attendance_record(rid)
                        print(f"{'✓' if success else '❌'} {msg}")
                    else:
                        print("❌ Cancelled.")
                else:
                    print("❌ Invalid record number.")
            except ValueError:
                print("❌ Invalid input.")

    # ── Export ────────────────────────────────────────────────────────────────
    def export_data(self):
        print("\n--- 📤 EXPORT DATA ---")
        filename    = input("Filename (Enter for auto-generated): ").strip()
        start_date  = input("Start date (YYYY-MM-DD) or Enter to skip: ").strip() or None
        end_date    = input("End date   (YYYY-MM-DD) or Enter to skip: ").strip() or None
        success, msg = self.db.export_to_csv(
            filename or None, start_date=start_date, end_date=end_date
        )
        print(f"{'✓' if success else '❌'} {msg}")

    # ── Statistics ────────────────────────────────────────────────────────────
    def show_statistics(self):
        print("\n--- 📈 ATTENDANCE STATISTICS ---")
        stats       = self.db.get_statistics()
        total_days  = self.db.get_total_working_days()

        if not stats:
            print("❌ No attendance data yet.")
            return

        print(f"{'#':<4} {'Name':<22} {'Days':>6} {'On Time':>9} {'Late':>7} {'%':>7} {'Last Seen'}")
        print("─" * 70)
        for idx, (name, days, on_time, late, last_seen) in enumerate(stats, 1):
            pct = f"{(days / total_days * 100):.1f}" if total_days else "N/A"
            print(f"{idx:<4} {name:<22} {days:>6} {on_time:>9} {late:>7} {pct:>7}% {last_seen}")

        print(f"\nTotal recorded working days: {total_days}")

    # ── Settings ──────────────────────────────────────────────────────────────
    def manage_settings(self):
        self.config.show()
        print("\n1. Change late threshold (current:", self.config.get("late_threshold"), ")")
        print("2. Change face recognition tolerance (current:", self.config.get("recognition_tolerance"), ")")
        print("3. Change camera index (current:", self.config.get("camera_index"), ")")
        print("4. Change kiosk cooldown seconds (current:", self.config.get("kiosk_cooldown_sec"), ")")
        print("5. Back")
        choice = input("\nSelect > ").strip()

        if choice == "1":
            val = input("New late threshold (HH:MM, e.g. 09:00): ").strip()
            if val:
                self.config.set("late_threshold", val)
                print("✓ Updated.")
        elif choice == "2":
            try:
                val = float(input("New tolerance (0.0 – 1.0, lower = stricter): "))
                if 0.0 <= val <= 1.0:
                    self.config.set("recognition_tolerance", val)
                    print("✓ Updated.")
                else:
                    print("❌ Must be between 0.0 and 1.0")
            except ValueError:
                print("❌ Invalid input.")
        elif choice == "3":
            try:
                val = int(input("Camera index (0 = built-in, 1 = external): "))
                self.config.set("camera_index", val)
                print("✓ Updated.")
            except ValueError:
                print("❌ Invalid input.")
        elif choice == "4":
            try:
                val = int(input("Cooldown seconds (e.g. 60): "))
                self.config.set("kiosk_cooldown_sec", val)
                print("✓ Updated.")
            except ValueError:
                print("❌ Invalid input.")

    # ── Kiosk Mode ────────────────────────────────────────────────────────────
    def run_kiosk_mode(self):
        if not self.engine.known_encodings:
            print("❌ No faces registered. Please add a person first (option 2).")
            return

        cam_index = self.config.get("camera_index")
        cooldown  = self.config.get("kiosk_cooldown_sec")
        
        print("  🚀  ADVANCED ATTENDANCE SYSTEM  v3.0")
        print("Press 'q' to quit.\n")

        try:
            cam = CameraStream(src=cam_index)
        except RuntimeError as e:
            print(f"❌ {e}")
            return

        GREEN = (0, 220, 80)
        RED   = (0, 60, 230)
        BLUE  = (230, 130, 0)
        FONT  = cv2.FONT_HERSHEY_SIMPLEX

        # Per-person cooldown: name → epoch time of last mark attempt
        last_attempt: dict[str, float] = {}

        try:
            while True:
                frame = cam.read()
                if frame is None:
                    if cam.stopped:
                        print("⚠  Camera disconnected.")
                        break
                    continue

                display = frame.copy()
                results = self.engine.identify(frame)

                for res in results:
                    name = res["name"]
                    conf = res["confidence"]
                    l, t, r, b = res["box"]

                    color = GREEN if name != "Unknown" else RED

                    # Draw bounding box
                    cv2.rectangle(display, (l, t), (r, b), color, 2)
                    label = f"{name}  {int(conf * 100)}%" if name != "Unknown" else "Unknown"
                    cv2.rectangle(display, (l, t - 32), (r, t), color, cv2.FILLED)
                    cv2.putText(display, label, (l + 5, t - 8), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                    # Mark attendance with cooldown guard
                    if name != "Unknown":
                        now_ts = time.time()
                        if now_ts - last_attempt.get(name, 0) >= cooldown:
                            last_attempt[name] = now_ts
                            success, msg = self.db.log_attendance(name)
                            if success:
                                print(f"  ✅ {msg}")
                                # Green flash
                                cv2.rectangle(display, (0, 0),
                                              (display.shape[1], display.shape[0]),
                                              GREEN, 12)
                                # Status overlay on face box
                                status_txt = "On Time" if self.db._compute_status(
                                    datetime.now().strftime("%H:%M:%S")) == "On Time" else "Late"
                                cv2.putText(display, status_txt, (l, b + 20),
                                            FONT, 0.5, GREEN, 1, cv2.LINE_AA)
                            else:
                                # Already marked — show quietly
                                cv2.putText(display, "Already Marked", (l, b + 20),
                                            FONT, 0.45, BLUE, 1, cv2.LINE_AA)

                # HUD – date/time overlay
                hud = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                cv2.putText(display, hud, (10, display.shape[0] - 12),
                            FONT, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

                cv2.imshow("🎓 Advanced Attendance System", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cam.stop()
            cv2.destroyAllWindows()
            logger.info("Kiosk mode exited.")

    # ── Main Menu ─────────────────────────────────────────────────────────────
    def main_menu(self):
        while True:
            print("\n" + "═" * 60)
            print("  🚀  ADVANCED ATTENDANCE SYSTEM  v3.0")
            print("═" * 60)
            print("  1.  Start Attendance System (Camera)")
            print("  2.  Add New Person")
            print("  3.  View Records")
            print("  4.  Registered People")
            print("  5.  Manage People  (Edit / Delete)")
            print("  6.  Manage Records (Edit / Delete)")
            print("  7.  Export Data to CSV")
            print("  8.  Attendance Statistics")
            print("  9.  Settings")
            print("  10. Exit")
            print("─" * 60)

            choice = input("  Select > ").strip()

            if   choice == "1":  self.run_kiosk_mode()
            elif choice == "2":  self.add_person()
            elif choice == "3":  self.view_records()
            elif choice == "4":  self.list_people()
            elif choice == "5":  self.manage_people()
            elif choice == "6":  self.manage_records()
            elif choice == "7":  self.export_data()
            elif choice == "8":  self.show_statistics()
            elif choice == "9":  self.manage_settings()
            elif choice == "10":
                print("\n  👋 Thank you for using the Advanced Attendance System!")
                logger.info("Application exited by user.")
                sys.exit(0)
            else:
                print("  ❌ Invalid choice. Please select 1–10.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AdvancedAttendanceSystem()
    app.main_menu()