# 🚦 End-to-End Workflow

### Capture → Detect → Track → Alert → Record → Explain → Explore

A classroom demonstration that combines **computer vision, local AI, and evidence-based incident analysis** using a Windows PC, an Android phone, and Hot Wheels cars.

---

## 1. Prepare the Environment

Open a terminal in the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download the local vision-language model:

```powershell
ollama pull qwen3-vl:2b-instruct
```

Check that Ollama is available:

```powershell
ollama list
```

> Keep the Ollama Windows application running in the background. If it is not running, execute `ollama serve` in a separate terminal and leave that terminal open.

---

## 2. Connect the Camera

On the Windows PC, open:

**Settings → Bluetooth & devices → Mobile devices → Manage devices → Use as connected camera**

Connect the Android phone through **Link to Windows** and accept camera permissions.

Find its camera number:

```powershell
.\.venv\Scripts\python.exe check_cameras.py
```

- Press **N** to preview the next camera.
- Note the number showing the phone’s view.
- Press **Q** to close the checker.

> The examples below use camera index `1`. Replace it with your phone’s actual camera index.

---

## 3. Start Crossing Monitoring

Arrange the miniature road and railway crossing, position the phone, and run:

```powershell
.\.venv\Scripts\python.exe -m crossing_monitor.video --camera 1 --monitor
```

### Select the restricted area

1. Click the crossing’s **four corners**, moving around its boundary in order.
2. Press **Enter** to confirm.
3. Press **R** before confirming if the boundary needs redrawing.

### Demonstrate an intrusion

| Action | Expected behaviour |
|---|---|
| Move a car outside the crossing | Detection boxes and tracking IDs appear |
| Enter while the crossing is Open | Occupancy is displayed |
| Press **C** to close the crossing | Closed-crossing monitoring begins |
| Move a car into the restricted zone | An **ENTRY** intrusion starts |
| Keep it inside for at least 3 seconds | The event becomes **PERSISTENT** |
| Keep it inside for at least 6 seconds | The event becomes **PROLONGED** |
| Move it outside | The incident closes |
| Press **Q** | Monitoring stops and remaining records are saved |

**Keep the camera stationary after selecting the zone.** If its position or orientation changes, restart monitoring and select the zone again.

### What happens internally?

```text
Phone Camera
     ↓
OpenCV captures frames
     ↓
YOLO detects vehicles
     ↓
ByteTrack maintains temporary object IDs
     ↓
Zone overlap + simulated crossing status
     ↓
Per-track intrusion timers
     ↓
Visual warning + escalation + saved evidence
```

---

## 4. Review Saved Incidents

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the **Local URL** printed in the terminal, usually:

```text
http://localhost:8501
```

Select **Incident review** in the sidebar.

You can:

- Filter incidents by escalation level, record status, source, and duration.
- Inspect timestamps, tracking IDs, durations, and end reasons.
- View screenshots from observed escalation stages.
- Download filtered incident records as CSV.
- Generate or review Ollama analysis.

Monitoring can run in another terminal. Click **Refresh records** after recording new incidents.

### Saved outputs

```text
data/runtime/
├── incidents.db
├── evidence/
│   ├── <incident_id>_entry.jpg
│   ├── <incident_id>_persistent.jpg
│   └── <incident_id>_prolonged.jpg
└── reports/
    ├── incidents.csv
    └── <incident_id>_analysis.json
```

> Each intrusion receives a unique incident ID. Tracking IDs are temporary, so incident totals represent **tracking episodes**, not unique physical cars.

---

## 5. Generate an AI Incident Report

In **Incident review**:

1. Select a closed incident.
2. Open the **Ollama report** tab.
3. Click **Generate analysis**.
4. Review the scene description, visible observations, and uncertainties.

The application sends a saved screenshot and recorded facts to:

```text
qwen3-vl:2b-instruct
```

### Report structure

| Section | Information source |
|---|---|
| Incident duration, level, and timestamps | SQLite records |
| Factual incident summary | Python-generated text |
| Scene description and visual observations | Ollama |
| Uncertainties | Ollama |
| Generation details | Model-response metadata |

**Measured facts remain separate from AI interpretation.** A valid JSON response still requires review for visual accuracy.

To generate a report from the terminal instead:

```powershell
.\.venv\Scripts\python.exe -m crossing_monitor.ollama_analysis --model qwen3-vl:2b-instruct --timeout 600
```

---

## 6. Ask Questions About Incident History

Select **Ask incident history** in the dashboard.

Try:

```text
How many intrusions happened today?
```

```text
Tell me about yesterday's incidents.
```

```text
Which incident lasted longest?
```

```text
Which incident lasted more than 8 seconds?
```

```text
Show prolonged incidents.
```

### How the answer is produced

```text
Your Question
     ↓
Python parses common requests
or Ollama interprets other wording
     ↓
Validated search filters
     ↓
Read-only SQLite queries
     ↓
Written answer with incident IDs and durations
     ↓
Optional supporting records
```

The written answer appears first. Expand:

- **Interpreted search** to verify the applied filters.
- **Supporting records** to inspect or download matching rows.

“More than 8 seconds” applies **`duration > 8`**.  
“At least 8 seconds” applies **`duration >= 8`**.

Dates use the PC’s local timezone and the incident’s recorded start time. Ask each question independently.

---

## 7. Inspect or Export Records from the Terminal

View recent incidents:

```powershell
.\.venv\Scripts\python.exe -m crossing_monitor.storage --limit 10
```

Export the incident summary:

```powershell
.\.venv\Scripts\python.exe -m crossing_monitor.storage --export data/runtime/reports/incidents.csv
```

For a repeatable demonstration using recorded footage:

```powershell
.\.venv\Scripts\python.exe -m crossing_monitor.video --video "data/videos/sample.mp4" --monitor
```

---

# 💡 Novelty & Design Contribution

The project’s contribution is the integration of **local perception, temporal reasoning, traceable evidence, and grounded AI interaction** into one demonstrable workflow.

| Design contribution | What makes it useful |
|---|---|
| **Temporal intrusion escalation** | Distinguishes a brief entry from prolonged occupancy using independent timers for tracked vehicles |
| **Vision and language working together** | YOLO and ByteTrack provide structured observations; Ollama adds readable visual explanations |
| **Alerts independent of language generation** | Rule-based warnings trigger directly, while AI reports are generated separately |
| **Evidence-linked incident records** | Unique incident IDs connect timestamps, escalation states, screenshots, and analysis reports |
| **Grounded natural-language history search** | Questions produce database-calculated answers with supporting IDs instead of model-generated totals |
| **Local edge-processing architecture** | The PC performs detection and analysis close to the camera source, without requiring a cloud AI API |
| **Reproducible miniature testbed** | A phone and toy crossing make it possible to repeat intrusion scenarios in a classroom |

### The central idea

> **Move beyond detecting an object: track its intrusion over time, preserve the evidence, explain the recorded event, and make its history searchable.**

These are **system-design contributions**, rather than claims of a new detection algorithm. Research novelty should be established through a literature review and measured comparisons.

---

<div align="center">

**📷 Capture locally → 🚦 Respond automatically → 🗂️ Preserve evidence → 🧠 Explain and explore**

*A miniature crossing demonstrating an end-to-end computer-vision monitoring workflow.*

</div>