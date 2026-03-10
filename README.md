
# 🏀 Basketball Auto Zooming

An automatic basketball video zooming system that detects and classifies scene objects (players, referees, ball, audience) and dynamically adjusts the camera view using weighted positioning and PID smoothing to simulate professional sports broadcasting cameras.

> Built with YOLO-based detection, spatial weighting, and PID control to produce smooth, human-like camera motion.

---

## 🎥 Demo

![demo](Compare.gif)

👉 Full demo video: (replace with your YouTube / Drive link)

---

## 🚀 Overview

Basketball broadcast footage often contains wide-angle shots where key gameplay details are difficult to observe.  
This project implements an intelligent **auto-zoom camera system** that:

- Detects humans using YOLO-based detection  
- Classifies detected objects into gameplay-relevant roles  
- Computes a weighted focus center  
- Dynamically adjusts zoom region  
- Uses PID control to simulate smooth human camera motion  

The system mimics how professional broadcast cameras track gameplay naturally.

---

## ✨ Features

- 🎯 YOLO-based human detection  
- 🧠 Role classification (player / referee / audience / ball)  
- ⚖️ Weighted spatial attention mechanism  
- 🎥 Dynamic zoom framing  
- 🎛 PID-based motion smoothing  
- ⚡ Fully offline and lightweight pipeline  

---

## 🧠 Methodology

### 1️⃣ Detection (YOLO-based)

Each frame is processed using a YOLO-based detector to identify human figures and objects.  
Bounding boxes are extracted for all detected entities.

---

### 2️⃣ Scene Classification

Detected objects are categorized into:

- **Players** — main gameplay focus  
- **Referees** — secondary gameplay relevance  
- **Ball** — highest dynamic importance  
- **Audience** — ignored or minimally weighted  

Classification is performed using spatial rules and contextual heuristics such as:

- Court region constraints  
- Bounding box scale  
- Motion behavior across frames  

---

### 3️⃣ Weighted Focus Estimation

Each category contributes differently to the final camera center:

Example weighting scheme:

| Object Type | Weight |
|-------------|--------|
| Ball        | High   |
| Player      | High   |
| Referee     | Medium |
| Audience    | Low    |

The final focus center is computed as:
$C = \frac{\sum w_i \cdot p_i}{\sum w_i}$

Where:

- $\(p_i\)$: object center  
- $\(w_i\)$: object weight  

This ensures gameplay-relevant regions dominate camera focus.

---

### 4️⃣ Dynamic Zoom Region Generation

The zoom window is computed using:

- Focus center  
- Spatial distribution of players  
- Minimum framing constraints  

This avoids excessive cropping while maintaining clarity.

---

### 5️⃣ PID-Based Camera Motion (Key Contribution)

To simulate human camera operation, the system applies a PID controller:

$\[
u(t)=K_p e(t)+K_i \int e(t)\,dt+K_d \frac{de(t)}{dt}
\]$

Where:

- $\(e(t)\)$: difference between target center and current frame center  

Effects:

- **P-term** → responsiveness  
- **I-term** → drift correction  
- **D-term** → jitter suppression  

This produces smooth, natural camera motion instead of robotic jumps.

---

## 🛠 Tech Stack

- Python  
- OpenCV  
- NumPy  
- YOLO (object detection)  
- Control theory (PID)

---

## 📂 Project Structure

## 📂 Project Structure



```text
BasketballAutoZooming/
│
├── main.py          # Entry point
├── zoom.py          # Zooming logic
├── detect.py        # Detection logic
├── utils.py         # Helper functions
├── data/            # (ignored) input/output videos
└── README.md
```


## 📊 Example Use Cases

* Sports broadcasting automation
* Training footage analysis
* Highlight generation
* AI-assisted video editing workflows

---

## 🔮 Future Improvements

* Real-time streaming support
* Ball detection integration
* Multi-camera switching
* Deep learning-based tracking

---



## 📄 License

This project is for educational and research purposes.

