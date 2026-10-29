
import os
import time
import threading
import datetime
import smtplib
import cv2

from email.message import EmailMessage


class EmergencyEmail:
    def __init__(
        self,
        recipients=None,
        photo_dir="emergency_photos",
        smtp_host="smtp.gmail.com",
        smtp_port=465,
        sender_email="mukund.jupyter@gmail.com",
        sender_app_password="lcljvkxwglwvsnma",
        email_rounds=3,
        email_interval=30,
    ):
        self.recipients = recipients or [
            email.strip()
            for email in os.getenv("EMERGENCY_RECIPIENTS", "").split(",")
            if email.strip()
        ]

        self.photo_dir = photo_dir
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = (
            sender_email or os.getenv("EMERGENCY_SENDER_EMAIL")
        )
        self.sender_app_password = (
            sender_app_password
            or os.getenv("EMERGENCY_SENDER_APP_PASSWORD")
        )
        self.email_rounds = max(1, int(email_rounds))
        self.email_interval = max(0, int(email_interval))

        os.makedirs(self.photo_dir, exist_ok=True)

    def capture_photo(self, frame):
        """Save the current webcam frame."""
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            raise ValueError("The webcam frame is empty.")

        timestamp = datetime.datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        photo_path = os.path.join(
            self.photo_dir,
            f"driver_emergency_{timestamp}.jpg",
        )

        if not cv2.imwrite(photo_path, frame):
            raise IOError("Failed to save the emergency photograph.")

        return photo_path

    def _send_emails(self, photo_path, alert_reason, event_time):
        """Send one email per recipient and delete the photo afterward."""

        try:
            if not self.sender_email or not self.sender_app_password:
                print("Emergency email credentials are missing.")
                return

            if not self.recipients:
                print("No emergency recipients are configured.")
                return

            with open(photo_path, "rb") as file:
                photo_data = file.read()

            for recipient in self.recipients:
                message = EmailMessage()
                message["From"] = self.sender_email
                message["To"] = recipient
                message["Subject"] = "EMERGENCY: Driver Alert"

                message.set_content(
                    "EMERGENCY DRIVER MONITORING ALERT\n\n"
                    "The driver monitoring alarm remained active for "
                    "the configured emergency delay.\n\n"
                    f"Alert reason: {alert_reason}\n"
                    f"Incident time: {event_time}\n\n"
                    "The driver's photograph is attached. Please follow "
                    "your emergency response procedure.\n"
                )

                message.add_attachment(
                    photo_data,
                    maintype="image",
                    subtype="jpeg",
                    filename=os.path.basename(photo_path),
                )

                try:
                    with smtplib.SMTP_SSL(
                        self.smtp_host,
                        self.smtp_port,
                        timeout=20,
                    ) as smtp:
                        smtp.login(
                            self.sender_email,
                            self.sender_app_password,
                        )
                        smtp.send_message(message)

                    print(f"Emergency email sent successfully to {recipient}")
                    return  # Stop after the first successful email.

                except (smtplib.SMTPException, OSError) as error:
                    print(f"Failed to email {recipient}: {error}")

        except OSError as error:
            print(f"Could not read emergency photograph: {error}")

        finally:
            # Delete the photo whether sending succeeds or fails.
            try:
                if os.path.exists(photo_path):
                    os.remove(photo_path)
                    print(f"Emergency photo deleted: {photo_path}")
            except OSError as error:
                print(f"Could not delete emergency photo: {error}")
    def trigger(self, frame, alert_reason="Drowsiness detected"):
        """Capture the photo and start email delivery in the background."""
        photo_path = self.capture_photo(frame)

        event_time = datetime.datetime.now().isoformat(
            timespec="seconds"
        )

        threading.Thread(
            target=self._send_emails,
            args=(photo_path, alert_reason, event_time),
            daemon=True,
        ).start()

        return photo_path
