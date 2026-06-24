# Real-Time Squat Counter using MediaPipe Pose

A real-time squat repetition counter built with **Python**, **OpenCV**, and **MediaPipe Tasks Vision API**.

The application uses webcam input to detect human body landmarks, calculates the knee angle during movement, and automatically counts completed squat repetitions.

---

## Features

* Real-time pose estimation
* Live skeleton visualization
* Knee angle calculation
* Automatic squat repetition counting
* State-based movement tracking (UP / DOWN)
* Webcam support

---

## Demo

### Squat Detection Logic

The application monitors the knee angle:

* **DOWN position** → Knee angle < 80°
* **UP position** → Knee angle > 150°

A repetition is counted when:

```text
UP → DOWN → UP
```

transition is completed.

---

## Technologies Used

* Python 3.x
* OpenCV
* NumPy
* MediaPipe Tasks Vision API

---

## Project Structure

```text
realtime-squat-counter/
│
├── main.py
├── README.md
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/realtime-squat-counter.git
cd realtime-squat-counter
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

### 4. Download MediaPipe Pose Model

Download:

```text
pose_landmarker_full.task
```

and place it in the project root directory.

---

## Running the Application

```bash
python main.py
```

Press:

```text
Q
```

to quit the application.

---

## How It Works

### 1. Pose Detection

MediaPipe Pose Landmarker detects 33 body landmarks from the webcam stream.

### 2. Landmark Extraction

The following landmarks are used:

| Landmark   | Index |
| ---------- | ----- |
| Left Hip   | 23    |
| Left Knee  | 25    |
| Left Ankle | 27    |

### 3. Angle Calculation

Two vectors are created:

```text
Hip -> Knee
Ankle -> Knee
```

The angle between these vectors is calculated using the cosine formula:

```python
cos_theta = np.dot(v1, v2) / (||v1|| * ||v2||)
theta = arccos(cos_theta)
```

The resulting angle is converted from radians to degrees.

### 4. State Machine

Current state can be:

```text
UP
DOWN
```

Logic:

```text
If angle < 80°
    State = DOWN

If angle > 150° and previous state == DOWN
    State = UP
    repetitions += 1
```

This prevents false counting caused by noisy detections.

---

## Current Limitations

* Uses only the left leg for angle calculation.
* Supports a single person.
* Thresholds are manually selected.
* No form quality assessment.

---

## Future Improvements

* Right/left leg averaging
* Form correctness analysis
* Rep quality scoring
* Exercise classification
* Push-up counter
* Plank timer
* Lunge detection
* GUI dashboard
* Performance optimization

---

## Example Output

```text
Degree: 163.21 | State: UP | Repeats: 7
```

---

## Author

Emin Çağan Apaydın

---

## License

MIT

---

Developed as a Computer Vision project using MediaPipe Pose estimation and OpenCV.
