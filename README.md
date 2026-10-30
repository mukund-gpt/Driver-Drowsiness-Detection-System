# Driver Drowsiness Monitoring System

A real-time driver monitoring application that uses a webcam and MediaPipe face and hand landmarks to detect possible drowsiness. It displays a live dashboard, plays an alarm, records events to a CSV file, and can send an emergency email with a webcam photo.

## Features

- Detects prolonged eye closure using the Eye Aspect Ratio (EAR).
- Detects prolonged yawning using the Mouth Aspect Ratio (MAR).
- Estimates head pose (yaw, pitch, and roll) from facial landmarks.
- Displays monitoring status, EAR and MAR measurements, session time, and event counts over the webcam feed.
- Plays `beep.mp3` continuously while an alert is active, if the audio device supports playback.
- Recognizes a thumbs-up gesture to dismiss an active alert.
- Captures a photo and attempts to send an emergency email when an alert is triggered.
- Logs session events and measurements to `logs/driver_events.csv`.

## Requirements

- Python 3.11 (the commands below use the Windows Python launcher, `py`).
- A working webcam.
- Speakers or headphones for the alarm (optional).
- An internet connection and Gmail SMTP access if you want to use emergency email.

The required Python packages and versions are listed in [requirements.txt](./requirements.txt). No separate dataset or model download is needed; face and hand landmark processing is provided by MediaPipe.

## Run on Windows

1. Open PowerShell in the project folder.
2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, open Command Prompt and activate it with:

   ```bat
   .venv\Scripts\activate.bat
   ```

3. Install the dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Connect or enable your webcam and, if desired, speakers. Keep `beep.mp3` in the project folder.
5. Start the application from the project folder:

   ```powershell
   python main.py
   ```

6. The webcam dashboard opens. Press **Q** while the dashboard window is focused to stop monitoring. A session summary is printed in the terminal.

## Run on macOS or Linux

Use Python 3.11 and run these commands from the project folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Grant the terminal or Python process permission to use the camera if prompted. Press **Q** in the dashboard window to quit.

## Detection and alert behavior

- The webcam is opened using camera index `0` (`CAMERA_INDEX` in `main.py`).
- Eye closure is considered prolonged when EAR is below `0.23` for at least 5 seconds.
- A yawn is detected when MAR is above `0.60` for at least 1.5 seconds.
- An alert is triggered by prolonged eye closure or prolonged yawning.
- A thumbs-up gesture detected for 5 consecutive frames dismisses the active alert.
- `EMERGENCY_DELAY_SECONDS` is currently `0`, so the emergency notification is triggered as soon as an alert becomes active. Change this value in `main.py` if you want a delay.

These are fixed thresholds in the source code, not command-line options. Detection quality depends on camera placement, lighting, and the driver's position; this project is a prototype and should not be treated as a safety-critical or medical device.

## Emergency email setup

Email is sent using Gmail SMTP over SSL. Before relying on it:

1. Set the intended recipient in the `EmergencyEmail` configuration in `main.py`.
2. Configure the sender address and Gmail app password in `emergency.py`.
3. Keep credentials private. Do not commit real credentials or share a copy of `emergency.py` containing them.

The current `EmergencyEmail` constructor has non-empty sender defaults, so setting `EMERGENCY_SENDER_EMAIL` or `EMERGENCY_SENDER_APP_PASSWORD` in the environment alone will not override those defaults. Replace the defaults or update the constructor configuration before expecting environment-variable settings to be used. The application attempts delivery in a background thread; if credentials, a recipient, or network access are unavailable, it reports the failure in the terminal. Emergency photos are saved temporarily under `emergency_photos` and removed after the email attempt.

## Output files

- `logs/driver_events.csv` — timestamps, event names, details, and available EAR/MAR/head-pose measurements.
- `emergency_photos/` — temporary photos captured for emergency notifications; the email sender attempts to delete each photo after delivery is attempted.

Both directories are created automatically when the application runs.

## Project files

- `main.py` — initializes the webcam, MediaPipe face/hand tracking, detection and alert logic, dashboard, audio alarm, and CSV event logging.
- `emergency.py` — captures a webcam frame and sends it as an email attachment.
- `beep.mp3` — alarm sound played during an active alert.
- `requirements.txt` — pinned Python dependencies.
