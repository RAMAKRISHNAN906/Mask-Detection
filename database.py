"""
Database layer for detection logs and user authentication using SQLite.
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password):
    """Hash a password with SHA-256."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            total_faces INTEGER DEFAULT 0,
            mask_count INTEGER DEFAULT 0,
            no_mask_count INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.0,
            result TEXT NOT NULL,
            image_path TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT DEFAULT '',
            role TEXT DEFAULT 'user',
            avatar_color TEXT DEFAULT '#00f0ff',
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    # Migrate: drop legacy admin_users table if it exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='admin_users'")
    if cursor.fetchone():
        cursor.execute("DROP TABLE admin_users")
    # Insert default admin if no users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, role, avatar_color, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "admin",
            "admin@maskguard.ai",
            _hash_password("admin123"),
            "Administrator",
            "admin",
            "#a855f7",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))
    conn.commit()
    conn.close()


def log_detection(source, total_faces, mask_count, no_mask_count, confidence, result, image_path=None):
    """Log a detection event."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO detection_logs
        (timestamp, source, total_faces, mask_count, no_mask_count, confidence, result, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source,
        total_faces,
        mask_count,
        no_mask_count,
        round(confidence, 2),
        result,
        image_path
    ))
    conn.commit()
    conn.close()


def get_all_logs(limit=100):
    """Retrieve recent detection logs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM detection_logs ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_statistics():
    """Get aggregate detection statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM detection_logs")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COALESCE(SUM(mask_count), 0) as masks FROM detection_logs")
    total_masks = cursor.fetchone()["masks"]

    cursor.execute("SELECT COALESCE(SUM(no_mask_count), 0) as no_masks FROM detection_logs")
    total_no_masks = cursor.fetchone()["no_masks"]

    cursor.execute("SELECT COALESCE(AVG(confidence), 0) as avg_conf FROM detection_logs")
    avg_confidence = cursor.fetchone()["avg_conf"]

    cursor.execute("SELECT COALESCE(SUM(total_faces), 0) as faces FROM detection_logs")
    total_faces = cursor.fetchone()["faces"]

    # Daily counts for the last 7 days
    cursor.execute("""
        SELECT
            DATE(timestamp) as date,
            SUM(mask_count) as masks,
            SUM(no_mask_count) as no_masks
        FROM detection_logs
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp) DESC
        LIMIT 7
    """)
    daily = [dict(row) for row in cursor.fetchall()]
    daily.reverse()

    # Source breakdown
    cursor.execute("""
        SELECT source, COUNT(*) as count
        FROM detection_logs
        GROUP BY source
    """)
    sources = {row["source"]: row["count"] for row in cursor.fetchall()}

    conn.close()

    return {
        "total_detections": total,
        "total_faces": total_faces,
        "total_masks": total_masks,
        "total_no_masks": total_no_masks,
        "avg_confidence": round(avg_confidence, 2),
        "daily": daily,
        "sources": sources
    }


def verify_admin(username, password):
    """Verify admin credentials (legacy wrapper)."""
    user = authenticate_user(username, password)
    return user is not None


# ── User Authentication ────────────────────────────────────────

def register_user(username, email, password, full_name=""):
    """Register a new user. Returns (success, message)."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check for duplicates
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username already taken"

    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cursor.fetchone():
        conn.close()
        return False, "Email already registered"

    # Pick a random avatar colour
    import random
    colors = ["#00f0ff", "#a855f7", "#f43f5e", "#10b981", "#f59e0b", "#6366f1", "#ec4899"]
    avatar_color = random.choice(colors)

    cursor.execute("""
        INSERT INTO users (username, email, password_hash, full_name, role, avatar_color, created_at)
        VALUES (?, ?, ?, ?, 'user', ?, ?)
    """, (
        username,
        email,
        _hash_password(password),
        full_name,
        avatar_color,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()
    return True, "Account created successfully"


def authenticate_user(username, password):
    """Authenticate user by username or email. Returns user dict or None."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE (username = ? OR email = ?) AND password_hash = ?",
        (username, username, _hash_password(password))
    )
    row = cursor.fetchone()
    if row:
        user = dict(row)
        # Update last_login
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user["id"])
        )
        conn.commit()
        conn.close()
        return user
    conn.close()
    return None


def get_user_by_id(user_id):
    """Get user by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_user_profile(user_id, full_name=None, email=None):
    """Update user profile fields. Returns (success, message)."""
    conn = get_connection()
    cursor = conn.cursor()

    if email:
        cursor.execute("SELECT id FROM users WHERE email = ? AND id != ?", (email, user_id))
        if cursor.fetchone():
            conn.close()
            return False, "Email already in use by another account"

    updates, params = [], []
    if full_name is not None:
        updates.append("full_name = ?")
        params.append(full_name)
    if email:
        updates.append("email = ?")
        params.append(email)

    if not updates:
        conn.close()
        return False, "Nothing to update"

    params.append(user_id)
    cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    conn.close()
    return True, "Profile updated"


def change_user_password(user_id, current_password, new_password):
    """Change password after verifying current one. Returns (success, message)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM users WHERE id = ? AND password_hash = ?",
        (user_id, _hash_password(current_password))
    )
    if not cursor.fetchone():
        conn.close()
        return False, "Current password is incorrect"

    cursor.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (_hash_password(new_password), user_id)
    )
    conn.commit()
    conn.close()
    return True, "Password changed successfully"


def get_all_users():
    """Get all users (admin feature)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, full_name, role, avatar_color, created_at, last_login FROM users ORDER BY id")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def clear_logs():
    """Clear all detection logs."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detection_logs")
    conn.commit()
    conn.close()
