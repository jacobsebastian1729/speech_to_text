# ai_receptionist/main.py

from .schema import create_tables
from .operations import add_room, book_room, get_available_rooms, add_event, get_all_bookings

def main():
    print("🔧 Creating tables...")
    create_tables()

    print("📦 Adding sample rooms...")
    add_room("101", "Deluxe")
    add_room("102", "Oceanfront")
    add_room("103", "Penthouse")

    print("🛏️ Booking room 101 for John...")
    book_room("John Doe", "101", "2025-06-15", "2025-06-17")

    print("📅 Scheduling event in room 102...")
    add_event("Wedding", "102", "2025-06-16 14:00", "2025-06-16 18:00")

    print("✅ Available Rooms:")
    for room in get_available_rooms():
        print(room)

    print("📄 All Bookings:")
    for booking in get_all_bookings():
        print(booking)

if __name__ == "__main__":
    main()
