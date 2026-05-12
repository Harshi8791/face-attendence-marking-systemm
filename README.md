# Advanced Attendance System v3.0

An industrial-grade, real-time facial recognition attendance system built with Python, OpenCV, and SQLite. This system features a robust thread-safe database, fast face encoding caching, and an interactive command-line interface (CLI) for managing people, records, and settings.

## 🚀 Key Features

*   **Real-time Facial Recognition**: Uses `face_recognition` and OpenCV to detect and identify registered faces in real-time.
*   **Thread-Safe Database**: Powered by SQLite with a `threading.Lock` to ensure data integrity during simultaneous camera reads and database writes.
*   **Kiosk Mode**: An automated attendance taking mode displaying an interactive HUD with green/red bounding boxes, confidences, and cooldown guards to prevent spamming entries.
*   **Dynamic Status Logging**: Automatically calculates if a person is "On Time" or "Late" based on a configurable time threshold.
*   **Fast Face Caching**: Speeds up system startup by securely caching previously computed face encodings using `pickle`.
*   **Comprehensive CLI Menu**: Manage registered users, view records, edit details, adjust settings, and view attendance statistics straight from the terminal.
*   **Data Export & Sync**: Keeps an always up-to-date `attendance.csv` file synced with the SQLite database, and supports manual filtered exports.
*   **Legacy Data Migration**: Automatically migrates old `config.json` and legacy CSV setups to the new database format.

## 🛠 Prerequisites

Make sure you have Python 3.8+ installed. You also need a working webcam (built-in or USB).

This project relies on the `face_recognition` library which requires `dlib`. It is recommended to install CMake and Visual Studio C++ Build Tools (on Windows) before installing the requirements.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/attendance-system.git
   cd attendance-system
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy face_recognition
   ```
   *(Note: `sqlite3`, `json`, `csv`, `threading`, and `pickle` are part of the standard Python library)*

## 🎮 Usage

Run the main script to launch the interactive CLI:

```bash
python attendance_system.py
```

### Main Menu Options:
1. **Start Attendance System**: Launches Kiosk Mode. Press `q` to exit the camera view.
2. **Add New Person**: Register a new user by providing a name and a path to their photo.
3. **View Records**: Display attendance logs with optional date and name filtering.
4. **Registered People**: View all enrolled users.
5. **Manage People**: Edit user names or completely delete a user and their records.
6. **Manage Records**: Modify or delete specific attendance entries.
7. **Export Data to CSV**: Generate a CSV report for specific date ranges.
8. **Attendance Statistics**: View total days, on-time percentage, and late counts per user.
9. **Settings**: Adjust the late threshold time, recognition tolerance, camera index, and cooldown seconds.
10. **Exit**: Gracefully shut down the application.

## 📂 Project Structure

```text
├── attendance_system.py      # Main application script
├── README.md                 # Project documentation
└── attendance_data/          # Auto-generated data directory
    ├── attendance.db         # SQLite database
    ├── encodings.pkl         # Cached face encodings
    ├── settings.json         # User settings
    ├── attendance_system.log # System execution logs
    ├── faces/                # Directory storing registered user photos
    └── records/              # Directory storing synced/exported CSVs
```

## ⚙️ Configuration

The system automatically generates a `settings.json` file in the `attendance_data/` folder. You can edit this directly or use **Option 9** in the CLI.

*   `camera_index`: Set to `0` for default laptop camera, `1` for external USB camera.
*   `recognition_tolerance`: Float between `0.0` and `1.0`. Lower is stricter (default `0.50`).
*   `late_threshold`: "HH:MM" format. Time after which attendance is marked as "Late".
*   `kiosk_cooldown_sec`: Minimum seconds between logging the same person.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have ideas for improvements or find any bugs.

## 📝 License

This project is licensed under the MIT License.
