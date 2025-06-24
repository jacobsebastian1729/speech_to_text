# ai_receptionist/db/operations.py

from .db_config import get_connection

def add_room(room_number, category):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO rooms (room_number, category) VALUES (?, ?)", (room_number, category))
    conn.commit()
    conn.close()

def book_room(guest_name, room_number, check_in, check_out):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO bookings (guest_name, room_number, check_in, check_out)
        VALUES (?, ?, ?, ?)
    """, (guest_name, room_number, check_in, check_out))
    # Mark room unavailable
    cursor.execute("UPDATE rooms SET is_available = 0 WHERE room_number = ?", (room_number,))
    conn.commit()
    conn.close()

def get_available_rooms(category=None):
    conn = get_connection()
    cursor = conn.cursor()
    if category:
        cursor.execute("SELECT * FROM rooms WHERE is_available = 1 AND category = ?", (category,))
    else:
        cursor.execute("SELECT * FROM rooms WHERE is_available = 1")
    rooms = cursor.fetchall()
    conn.close()
    return rooms

def add_event(name, room_number, start_time, end_time):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO events (name, room_number, start_time, end_time)
        VALUES (?, ?, ?, ?)
    """, (name, room_number, start_time, end_time))
    conn.commit()
    conn.close()

def get_all_bookings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM bookings")
    data = cursor.fetchall()
    conn.close()
    return data
