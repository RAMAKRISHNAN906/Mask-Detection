import os
import textwrap
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


TITLE = "Face Mask Detection using Image Processing and Deep Learning"
SUBTITLE = "Final Year Project Documentation"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "outputs", "final_year_project_face_mask_detection_80_pages.pdf")
WRAP = 102
MAX_LINES = 56

SECTION_PLAN = [
    ("Abstract", 2),
    ("Table of Contents", 2),
    ("Introduction", 6),
    ("Problem Statement", 3),
    ("Objectives", 3),
    ("System Analysis", 5),
    ("Existing System", 4),
    ("Proposed System", 5),
    ("System Design", 5),
    ("Architecture Diagram", 4),
    ("Flowchart", 3),
    ("UML Diagrams", 4),
    ("Implementation", 10),
    ("Modules Description", 6),
    ("Technologies Used", 4),
    ("Results and Discussion", 5),
    ("Output Screens", 3),
    ("Advantages", 2),
    ("Limitations", 2),
    ("Conclusion", 1),
    ("Future Scope", 1),
]

TOPICS = {
    "Introduction": [
        "public health motivation",
        "image processing basics",
        "deep learning for classification",
        "real time monitoring need",
        "project scope and boundaries",
        "expected outcomes",
    ],
    "Problem Statement": [
        "manual checking inconsistency",
        "crowded environment difficulty",
        "human fatigue",
        "slow traditional verification",
        "lack of digital record",
        "need for automation",
    ],
    "Objectives": [
        "detect faces from image webcam video",
        "classify mask and no mask",
        "maintain detection logs",
        "deliver user friendly interface",
        "keep inference fast",
        "enable report generation",
    ],
    "System Analysis": [
        "functional requirements",
        "non functional requirements",
        "hardware and software feasibility",
        "risk identification",
        "security and validation",
        "stakeholder needs",
    ],
    "Existing System": [
        "manual entry point checks",
        "rule based legacy tools",
        "no integrated dashboard",
        "lack of alert evidence",
        "limited scalability",
        "higher error rate",
    ],
    "Proposed System": [
        "face detector plus classifier pipeline",
        "flask route based architecture",
        "sqlite logging and analytics",
        "three input modes",
        "admin dashboard and pdf report",
        "modular extensible design",
    ],
    "System Design": [
        "layered module layout",
        "input validation",
        "model loading flow",
        "database schema",
        "api route mapping",
        "error handling strategy",
    ],
    "Technologies Used": [
        "python and flask",
        "opencv for cv pipeline",
        "pytorch and torchvision",
        "sqlite persistence",
        "html css js frontend",
        "reportlab for pdf",
    ],
    "Results and Discussion": [
        "experimental setup",
        "accuracy observations",
        "confidence trends",
        "webcam responsiveness",
        "video throughput tradeoff",
        "error case analysis",
    ],
    "Output Screens": [
        "home page",
        "webcam detection page",
        "image upload result",
        "video upload result",
        "dashboard statistics",
        "login and profile pages",
    ],
    "Advantages": [
        "works on image webcam video",
        "simple deployment",
        "real time alert support",
        "modular codebase",
        "admin analytics",
        "project friendly complexity",
    ],
    "Limitations": [
        "lighting sensitivity",
        "occlusion handling limits",
        "dataset dependence",
        "small scale database",
        "no person tracking",
        "single language ui",
    ],
    "Conclusion": [
        "successful integration of cv and dl",
        "practical usability demonstrated",
        "modular implementation achieved",
        "academic goals completed",
    ],
    "Future Scope": [
        "multi class ppe detection",
        "edge deployment",
        "cloud analytics",
        "notification integration",
        "domain adapted training",
        "tracking and reidentification",
    ],
}

IMPL_SPECS = [
    ("train_model.py", "def get_data_transforms", "Data Preprocessing"),
    ("train_model.py", "def build_model", "Model Construction"),
    ("train_model.py", "def train", "Training Pipeline"),
    ("app.py", "def load_face_detector", "Face Detector Loading"),
    ("app.py", "def detect_faces", "Face Localization"),
    ("app.py", "def predict_mask", "Mask Prediction"),
    ("app.py", "def process_frame", "End to End Frame Processing"),
    ("app.py", "@app.route(\"/api/detect/frame\"", "Webcam Detection API"),
    ("app.py", "@app.route(\"/api/detect/image\"", "Image Detection API"),
    ("app.py", "@app.route(\"/api/detect/video\"", "Video Detection API"),
]

MODULE_SPECS = [
    ("config.py", "MODEL_PATH =", "Configuration Module"),
    ("database.py", "def init_db", "Database Initialization"),
    ("database.py", "def authenticate_user", "Authentication Module"),
    ("database.py", "def get_statistics", "Analytics Module"),
    ("app.py", "@app.route(\"/api/report/pdf\")", "PDF Report Module"),
    (os.path.join("static", "js", "webcam.js"), "async function detectLoop", "Frontend Webcam Module"),
]


def a(text):
    return text.encode("ascii", "ignore").decode("ascii")


def w(text):
    text = a(text).strip()
    if not text:
        return [""]
    return textwrap.wrap(text, WRAP, break_long_words=False, break_on_hyphens=False) or [""]


def ranges():
    out = {}
    p = 1
    for name, count in SECTION_PLAN:
        out[name] = (p, p + count - 1)
        p += count
    return out


def read_lines(path):
    full = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    if not os.path.exists(full):
        return []
    with open(full, "r", encoding="utf-8", errors="ignore") as f:
        return [a(x.rstrip("\n")) for x in f.readlines()]


def snippet(path, anchor, max_lines=24):
    lines = read_lines(path)
    if not lines:
        return [f"# File not found: {path}"]
    start = -1
    for i, line in enumerate(lines):
        if anchor in line:
            start = i
            break
    if start < 0:
        return [f"# Anchor not found: {anchor}", f"# Source: {path}"]
    if lines[start].lstrip().startswith("def "):
        j = start - 1
        while j >= 0 and lines[j].lstrip().startswith("@"):
            j -= 1
        start = j + 1
    return lines[start : min(start + max_lines, len(lines))]


def format_code(lines):
    out = []
    for i, line in enumerate(lines, 1):
        pref = f"{i:02d} | "
        line = line.replace("\t", "    ").rstrip()
        parts = textwrap.wrap(
            line if line else "",
            max(20, WRAP - len(pref)),
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        out.append(pref + parts[0])
        for p in parts[1:]:
            out.append(" " * len(pref) + p)
    return out


def paragraph(section, topic, part, seed):
    s1 = f"In {section.lower()} part {part}, the focus is {topic}."
    s2 = "The implemented solution uses OpenCV for face localization and a MobileNetV2 based classifier for mask status prediction."
    s3 = "This keeps the system practical for final year development, easy to explain during viva, and suitable for real demonstrations."
    s4 = "Logging, reporting, and modular route design improve usability beyond a basic machine learning prototype."
    variants = [s2, s3, s4]
    return " ".join([s1] + variants[seed % 3 :] + variants[: seed % 3])


def build_abstract():
    p1 = []
    for txt in [
        "This project presents a face mask detection system using image processing and deep learning.",
        "The core pipeline detects faces and classifies each face as mask or no mask using a transfer learning model.",
        "The application supports image upload, webcam stream, and video upload for practical use in different scenarios.",
        "A Flask backend serves API routes, while SQLite stores logs for monitoring and analysis.",
        "The complete implementation is simple, modular, and suitable for final year documentation and demonstration.",
    ]:
        p1 += w(txt) + [""]
    p1 += w("Keywords: face mask detection, OpenCV, PyTorch, MobileNetV2, Flask, image processing, deep learning")

    p2 = []
    for txt in [
        "Training is performed with data augmentation and transfer learning to improve classification robustness.",
        "Inference converts detected face regions to 224 by 224 normalized tensors before model prediction.",
        "The system produces output files, confidence values, dashboard statistics, and downloadable report documents.",
        "This report is intentionally text only and uses simple language for direct academic submission.",
    ]:
        p2 += w(txt) + [""]
    return [
        {"section": "Abstract", "title": "Project Summary - Part 1", "lines": p1},
        {"section": "Abstract", "title": "Project Summary - Part 2", "lines": p2},
    ]


def build_toc():
    r = ranges()
    entries = [
        f"Abstract ............................................................ {r['Abstract'][0]}-{r['Abstract'][1]}",
        f"Table of Contents ................................................... {r['Table of Contents'][0]}-{r['Table of Contents'][1]}",
        "",
        "MAIN CONTENT",
    ]
    for name, _ in SECTION_PLAN:
        if name in ("Abstract", "Table of Contents"):
            continue
        s, e = r[name]
        page = f"{s}" if s == e else f"{s}-{e}"
        entries.append(f"{name:<35} ....................................... {page}")

    p1 = w("This table of contents follows the exact chapter order requested for the final year project report.") + [""]
    for line in entries[:14]:
        p1 += w(line)
    p2 = []
    for line in entries[14:]:
        p2 += w(line)
    p2 += [""] + w("All chapters are text only. Architecture, flowchart, and UML sections are represented using simple plain text diagrams.")
    return [
        {"section": "Table of Contents", "title": "Contents - Part 1", "lines": p1},
        {"section": "Table of Contents", "title": "Contents - Part 2", "lines": p2},
    ]


def cyc(items, start, count):
    return [items[(start + i) % len(items)] for i in range(count)]


def build_generic(section, count):
    topics = TOPICS[section]
    pages = []
    for part in range(1, count + 1):
        lines = []
        lines += w(
            f"This page is part {part} of {count} in {section}. The content is written in simple text format for direct inclusion in final year documentation."
        )
        lines += [""]
        for i, topic in enumerate(cyc(topics, part - 1, 5)):
            lines += w(paragraph(section, topic, part, i))
            lines += [""]
        lines += w("Key points for this page:")
        for kp in cyc(topics, part + 1, 6):
            lines += w(f"- {kp}")
        lines += [""] + w("The sequence of ideas across this chapter supports implementation, testing, and project viva explanation.")
        pages.append({"section": section, "title": f"{section} - Part {part}", "lines": lines})
    return pages


def build_architecture():
    raws = [
        [
            "Text Architecture Diagram - Overall View",
            "",
            "+-------------------+      +------------------------+      +----------------------+",
            "| Input Layer       | ---> | Processing Layer       | ---> | Output Layer         |",
            "| image/webcam/video|      | face detect + classify |      | preview + logs + pdf |",
            "+-------------------+      +------------------------+      +----------------------+",
            "",
            "Input receives files or live frame data.",
            "Processing stage runs face detection and mask classification.",
            "Output stage shows annotated result and stores log details.",
        ],
        [
            "Text Architecture Diagram - Backend Components",
            "",
            "Browser UI -> Flask Routes -> Detection Engine -> Database and File Storage",
            "",
            "Key routes:",
            "- /api/detect/frame",
            "- /api/detect/image",
            "- /api/detect/video",
            "- /api/dashboard/*",
            "- /api/report/pdf",
        ],
        [
            "Text Architecture Diagram - Inference Path",
            "",
            "Frame -> detect_faces() -> predict_mask() -> draw_detections() -> JSON response",
            "",
            "Each detected face is cropped, resized, normalized, and passed to model.",
            "Confidence score and label are returned per face object.",
        ],
        [
            "Text Architecture Diagram - Deployment View",
            "",
            "User Browser <-> Flask App <-> model files + sqlite db + output folder",
            "",
            "Client scripts handle webcam capture and dashboard rendering.",
            "Server modules handle prediction, logging, and report generation.",
            "This setup is lightweight and suitable for college lab deployment.",
        ],
    ]
    pages = []
    for i, raw in enumerate(raws, 1):
        lines = w(f"Architecture chapter page {i} of 4. Diagram is represented in plain text as requested.") + [""]
        for r in raw:
            if len(r) > WRAP:
                lines += w(r)
            else:
                lines.append(a(r))
        lines += [""] + w("The modular architecture makes testing and explanation easier for final year project evaluation.")
        pages.append({"section": "Architecture Diagram", "title": f"Architecture Diagram - Part {i}", "lines": lines})
    return pages


def build_flow():
    raws = [
        [
            "Flowchart Page 1 - High Level",
            "",
            "Start -> Select source -> Validate input -> Detect faces -> Predict mask status",
            "-> Draw output -> Save logs -> Show result -> End",
        ],
        [
            "Flowchart Page 2 - Webcam API",
            "",
            "Receive base64 frame -> Decode -> Validate -> Process frame",
            "If no mask found then save alert screenshot",
            "Return JSON with detections, counts, and fps",
        ],
        [
            "Flowchart Page 3 - Video Mode",
            "",
            "Open video -> Loop frames -> Process every third frame -> Write output",
            "Update totals -> Close resources -> Save summary -> Return download path",
        ],
    ]
    pages = []
    for i, raw in enumerate(raws, 1):
        lines = w(f"Flowchart chapter page {i} of 3. Steps are written in text format to keep the report simple.") + [""]
        for r in raw:
            lines += [a(r)] if len(r) <= WRAP else w(r)
        lines += [""] + w("These flow steps map directly to implemented route logic and helper functions in the project code.")
        pages.append({"section": "Flowchart", "title": f"Flowchart - Part {i}", "lines": lines})
    return pages


def build_uml():
    raws = [
        [
            "Use Case Diagram (Text)",
            "Actors: User, Admin, System",
            "User: register, login, upload image/video, use webcam, view output",
            "Admin: view dashboard, inspect logs, clear logs, generate report",
            "System: detect faces, classify mask status, save detection entries",
        ],
        [
            "Class Diagram (Text)",
            "User class: id, username, email, role, password_hash",
            "DetectionLog class: timestamp, source, total_faces, mask_count, result",
            "DetectionEngine class: load_model, detect_faces, predict_mask, process_frame",
            "Relationship: one admin can review many detection logs",
        ],
        [
            "Sequence Diagram (Text)",
            "User -> UI -> /api/detect/image -> DetectionEngine -> Database -> UI -> User",
            "System returns annotated output and statistics with confidence values",
        ],
        [
            "Activity Diagram (Text)",
            "Start -> Input -> Validate -> Detect face -> Predict label -> Save output -> Log -> End",
            "Alternative: if no face, return no face result and keep process stable",
        ],
    ]
    pages = []
    for i, raw in enumerate(raws, 1):
        lines = w(f"UML chapter page {i} of 4. {raw[0]} is represented as plain text.") + [""]
        for r in raw:
            lines += [a(r)] if len(r) <= WRAP else w(r)
        lines += [""] + w("Text based UML is used to satisfy no design formatting requirement while still documenting software structure.")
        pages.append({"section": "UML Diagrams", "title": f"UML Diagrams - Part {i}", "lines": lines})
    return pages


def build_code(section, specs):
    pages = []
    total = len(specs)
    for i, (path, anchor, title) in enumerate(specs, 1):
        code = snippet(path, anchor, 24)
        lines = []
        lines += w(f"{section} coding page {i} of {total}. Topic: {title}.")
        lines += [""]
        lines += w("This page includes an extracted code segment from your current project workspace for authentic documentation.")
        lines += [""]
        lines += w("Implementation notes:")
        lines += w("- Listing is trimmed to the most relevant lines.")
        lines += w("- Line numbers are added for viva discussion.")
        lines += w("- The segment is linked to the chapter explanation.")
        lines += [""] + w("Code listing:")
        lines += format_code(code)
        lines += [""] + w("This module contributes to end to end mask detection workflow and report generation.")
        pages.append({"section": section, "title": f"{title} ({i}/{total})", "lines": lines})
    return pages


def build_pages():
    all_pages = []
    for section, count in SECTION_PLAN:
        if section == "Abstract":
            pages = build_abstract()
        elif section == "Table of Contents":
            pages = build_toc()
        elif section == "Architecture Diagram":
            pages = build_architecture()
        elif section == "Flowchart":
            pages = build_flow()
        elif section == "UML Diagrams":
            pages = build_uml()
        elif section == "Implementation":
            pages = build_code("Implementation", IMPL_SPECS)
        elif section == "Modules Description":
            pages = build_code("Modules Description", MODULE_SPECS)
        else:
            pages = build_generic(section, count)
        if len(pages) != count:
            raise ValueError(f"Section {section} expected {count}, got {len(pages)}")
        all_pages.extend(pages)
    if len(all_pages) != 80:
        raise ValueError(f"Expected 80 pages, got {len(all_pages)}")
    return all_pages


def draw_pdf(pages):
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    c = canvas.Canvas(OUT_PATH, pagesize=A4)
    wpage, hpage = A4
    mx = 34
    hy = hpage - 34
    by = hpage - 74
    fy = 24
    total = len(pages)

    for pno, page in enumerate(pages, 1):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(mx, hy, TITLE)
        c.setFont("Helvetica", 9)
        c.drawString(mx, hy - 14, f"{a(page['section'])} | {a(page['title'])}")
        c.drawRightString(wpage - mx, hy - 14, f"Page {pno} of {total}")
        c.line(mx, hy - 18, wpage - mx, hy - 18)

        text = c.beginText(mx, by)
        text.setFont("Courier", 8.8)
        text.setLeading(12)
        body = [a(str(x)) for x in page["lines"]]
        body = body[:MAX_LINES] if len(body) <= MAX_LINES else body[: MAX_LINES - 1] + ["[Content trimmed for formatting]"]
        for line in body:
            text.textLine(line)
        c.drawText(text)

        c.setFont("Helvetica", 8)
        c.drawString(mx, fy, SUBTITLE)
        c.drawRightString(wpage - mx, fy, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.showPage()
    c.save()


def main():
    pages = build_pages()
    draw_pdf(pages)
    print(f"Generated: {OUT_PATH}")
    print(f"Total pages: {len(pages)}")


if __name__ == "__main__":
    main()
