# 👶 Baby Monitoring System using OpenCV, MediaPipe, Twilio, and Pygame

This Python-based baby monitoring system detects key baby behaviors and conditions using computer vision and sends alerts in real-time. It uses webcam video input to detect if the baby:

- 👀 Wakes up (based on Eye Aspect Ratio)
- 😢 Cries (based on mouth opening)
- 🚫 Moves out of a defined safe zone
- 🕵️‍♀️ Becomes invisible to the camera

If any of these events occur, the system:
- Plays a **beep** or **lullaby**
- Sends **SMS alerts** to parents via **Twilio API**

---

## 📁 Project Structure

- `FINAL_PROJECT_CODE.py`: Main script for monitoring.
- `cryaudio.mp3`: Audio file for the lullaby (You must place this at the correct `SONG_PATH`).

---

## 🛠️ Requirements

Install dependencies with pip:

```bash
pip install opencv-python mediapipe pygame twilio scipy
