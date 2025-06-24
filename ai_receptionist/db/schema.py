# ai_receptionist/db/schema.py

from .db_config import get_connection

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # Rooms
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rooms (
        room_id INTEGER PRIMARY KEY AUTOINCREMENT,
        room_number TEXT UNIQUE,
        category TEXT,
        is_available INTEGER DEFAULT 1
    )
    """)

    # Bookings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bookings (
        booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
        guest_name TEXT,
        room_number TEXT,
        check_in DATE,
        check_out DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Events
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        room_number TEXT,
        start_time DATETIME,
        end_time DATETIME
    )
    """)

    conn.commit()
    conn.close()
