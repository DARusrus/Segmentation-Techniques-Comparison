<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4E79A7,33:F28E2B,66:59A14F,100:E15759&height=200&section=header&text=Four%20Paradigms,%20One%20Scene&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=A%20Forensic%20Comparison%20of%20Segmentation%20Ontologies&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=18&duration=3000&pause=800&color=7AB8D9&center=true&vCenter=true&multiline=true&width=700&height=80&lines=Semantic+%C2%B7+Instance+%C2%B7+Panoptic+%C2%B7+Promptable;SegFormer+%C2%B7+YOLOv8x-seg+%C2%B7+Mask2Former+%C2%B7+SAM+2;125+COCO+Images+%C2%B7+4+Models+%C2%B7+6+Deep+Metrics)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-T4%20GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

<br/>

![License](https://img.shields.io/badge/License-MIT-59A14F?style=flat-square)
![COCO](https://img.shields.io/badge/Dataset-COCO%202017-4E79A7?style=flat-square)
![Models](https://img.shields.io/badge/Models-4%20Pretrained-F28E2B?style=flat-square)
![Images](https://img.shields.io/badge/Evaluated%20On-125%20Images-E15759?style=flat-square)
![Strata](https://img.shields.io/badge/Scene%20Strata-5%20Types-59A14F?style=flat-square)

</div>

---

<div align="center">

## ❝ Choosing your segmentation paradigm is choosing your question — and the wrong question costs you everything. ❞

</div>

---

## 🧭 Overview

<div align="center">

| Paradigm | Core Question | Model | Speed | Vocabulary |
|:---:|:---|:---:|:---:|:---:|
| 🔵 **Semantic** | *What class is at every pixel?* | SegFormer-B2 | `72 ms` | Closed (150) |
| 🟠 **Instance** | *Where is each individual object?* | YOLOv8x-seg | `65 ms` | Closed (80) |
| 🟢 **Panoptic** | *What AND which — full scene?* | Mask2Former | `164 ms` | Closed (133) |
| 🔴 **Promptable** | *Whatever I point at — right now?* | SAM 2 Auto | `5539 ms` | **Open** |

</div>

<br/>

This project is not another "run four models and show a table." It is a **forensic dissection** of what each segmentation paradigm fundamentally *is* — what question it answers, where it catastrophically fails, and exactly when you should reach for it over the others. Every claim is backed by metrics computed over **125 stratified COCO 2017 images** across five scene archetypes, rendered through **6 publication-quality analyses** and deployed as an **interactive Gradio application**.

---

## ✨ Features

<div align="center">

<table width="100%">
  <thead>
    <tr>
      <th colspan="2" align="center">✦ &nbsp; WHAT MAKES THIS DIFFERENT &nbsp; ✦</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="28%" align="center"><strong>🎯 Stratified Sampling</strong></td>
      <td>25 images × 5 scene archetypes — <code>crowd · animal · indoor · vehicle · mixed</code><br/>Ensures every metric has diagnostic meaning, not just average behaviour</td>
    </tr>
    <tr>
      <td align="center"><strong>📐 Boundary Sharpness</strong></td>
      <td>Trimap-based edge F-measure via Sobel magnitude — measures pixel-level mask-to-edge alignment<br/>Goes far beyond standard IoU to expose where each model truly snaps to real boundaries</td>
    </tr>
    <tr>
      <td align="center"><strong>🧮 Paradigm Divergence</strong></td>
      <td>Pairwise pixel-level IoU agreement between all 4 paradigms rendered as a 4×4 heatmap<br/>Quantifies how fundamentally differently each model interprets the same scene</td>
    </tr>
    <tr>
      <td align="center"><strong>🌐 Pareto Frontier</strong></td>
      <td>Speed vs composite quality with per-stratum satellite sub-points connected by spokes<br/>Reveals that paradigm rankings <em>shift</em> by scene — YOLO rises on vehicles, SAM collapses on crowds</td>
    </tr>
    <tr>
      <td align="center"><strong>🗺️ Decision Matrix</strong></td>
      <td>Data-driven 6×4 suitability heatmap for real-world use cases scored 1–5 from actual measured metrics<br/>Autonomous driving · Medical imaging · Robotics · Surveillance · E-commerce · Content moderation</td>
    </tr>
    <tr>
      <td align="center"><strong>⚡ Live Gradio App</strong></td>
      <td>3-tab interactive demo — upload any image for live 4-model inference, browse 20 pre-computed failure<br/>cases, or consult the full decision atlas. Public URL via <code>share=True</code>, zero infrastructure</td>
    </tr>
  </tbody>
</table>

</div>

---

## 🏗️ System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#0d1f2d',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#4E79A7',
  'lineColor': '#59A14F',
  'secondaryColor': '#1a0d05',
  'tertiaryColor': '#0a130a',
  'edgeLabelBackground': '#0d1f2d',
  'clusterBkg': '#0a0f14'
}}}%%

flowchart TB
    subgraph INPUT["📦 Input Layer"]
        COCO[("COCO 2017\nval2017\n5,000 images")]
        STRAT["Stratified Sampler\n25 × 5 strata\n= 125 images"]
        COCO --> STRAT
    end

    subgraph STRATA["🗂️ Scene Archetypes"]
        direction LR
        S1["👥 Crowd"]
        S2["🐾 Animal"]
        S3["🪑 Indoor"]
        S4["🚗 Vehicle"]
        S5["🔀 Mixed"]
    end

    STRAT --> S1 & S2 & S3 & S4 & S5

    subgraph MODELS["🤖 Model Layer  —  All Pretrained · Zero Training"]
        direction TB
        SEM["🔵 SegFormer-B2\nnvidia/segformer-b2\nADE20K 150 classes\n72 ms / image"]
        INST["🟠 YOLOv8x-seg\nultralytics\nCOCO 80 classes\n65 ms / image"]
        PAN["🟢 Mask2Former\nfacebook/mask2former\nswin-large-coco-panoptic\n164 ms / image"]
        SAM["🔴 SAM 2\nfacebook/sam2-hiera-base+\nClass-agnostic\n5539 ms / image"]
    end

    S1 & S2 & S3 & S4 & S5 --> SEM & INST & PAN & SAM

    subgraph RESULTS["💾 RESULTS Store"]
        RES[("RESULTS dict\n125 × 4 model outputs\nlabel maps · masks\nRGB overlays · timing")]
    end

    SEM & INST & PAN & SAM --> RES

    subgraph METRICS["📐 Metric Engine"]
        M1["Boundary Sharpness\nSobel + trimap band"]
        M2["Count Faithfulness\n|pred − GT|"]
        M3["Thing/Stuff Coverage\nPixel fraction"]
        M4["Semantic Entropy\nShannon H bits"]
        M5["Paradigm Divergence\nPairwise IoU"]
        M6["Composite Quality\nWeighted score"]
    end

    RES --> M1 & M2 & M3 & M4 & M5 & M6

    subgraph OUTPUTS["📊 Outputs"]
        direction LR
        P1["Boundary\nViolin Plots"]
        P2["Count\nScatter Grid"]
        P3["Coverage\nBubble Chart"]
        P4["Entropy\nBox Plots"]
        P5["Divergence\nHeatmap"]
        P6["Radar\nCharts"]
        P7["Failure\nGallery"]
        P8["Pareto\nFrontier"]
        P9["Decision\nMatrix"]
        P10["Insight\nPoster"]
    end

    M1 --> P1
    M2 --> P2
    M3 --> P3
    M4 --> P4
    M5 --> P5
    M1 & M2 & M3 & M4 & M5 --> P6
    M1 & M2 --> P7
    M6 --> P8
    M6 --> P9
    P1 & P2 & P3 & P4 & P5 & P6 --> P10

    subgraph DEPLOY["🚀 Deployment"]
        GRAD["Gradio App\n3 Tabs\nLive inference\nStress gallery\nDecision atlas"]
    end

    OUTPUTS --> DEPLOY
```

---

## 🔄 Pipeline & Data Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#1f1005',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#F28E2B',
  'lineColor': '#E15759',
  'secondaryColor': '#0a0d0f',
  'tertiaryColor': '#0a130a',
  'activationBkgColor': '#1f1005',
  'activationBorderColor': '#F28E2B',
  'labelBoxBkgColor': '#0d1f2d',
  'labelBoxBorderColor': '#4E79A7',
  'loopTextColor': '#ffffff',
  'noteBkgColor': '#0a130a',
  'noteTextColor': '#ffffff',
  'noteBorderColor': '#59A14F'
}}}%%

sequenceDiagram
    actor User
    participant S1 as 📦 Snippet 1<br/>Foundation
    participant S2 as ⚙️ Snippet 2<br/>Inference
    participant S3 as 📐 Snippet 3<br/>Metrics
    participant S4 as 🔬 Snippet 4<br/>Forensics
    participant S5 as 🚀 Snippet 5<br/>Gradio
    participant GPU as 🖥️ T4 GPU
    participant FS as 💾 /kaggle/working

    User->>S1: Execute kernel
    S1->>S1: Install packages & configure matplotlib
    S1->>S1: Load COCO instances_val2017.json
    S1->>S1: Classify 5000 images → 5 strata
    S1->>S1: Sample 25 × 5 = 125 images
    S1->>GPU: Load SegFormer-B2 (110 MB)
    S1->>GPU: Load YOLOv8x-seg (137 MB)
    S1->>GPU: Load Mask2Former (866 MB)
    S1->>GPU: Load SAM 2 (323 MB)
    S1-->>User: ✓ Models warm · RESULTS={} ready

    User->>S2: Execute kernel
    loop 125 images × 4 models
        S2->>GPU: run_segformer(pil, hw) → 72 ms
        S2->>GPU: run_yolo(pil, hw) → 65 ms
        S2->>GPU: run_mask2former(pil, hw) → 164 ms
        S2->>GPU: run_sam2(pil, hw) → 5539 ms
        S2->>S2: Pack → RESULTS[iid]
    end
    S2->>FS: timing_chart.png
    S2->>FS: gallery_crowd/animal/indoor/vehicle/mixed.png
    S2-->>User: ✓ 125 images · health table printed

    User->>S3: Execute kernel
    S3->>S3: Pre-compute METRICS[iid] for all 125
    note over S3: Boundary sharpness via Sobel + trimap<br/>Count error vs COCO GT<br/>Thing/stuff coverage<br/>Shannon entropy<br/>Pairwise IoU divergence
    S3->>FS: boundary_sharpness.png
    S3->>FS: count_faithfulness.png
    S3->>FS: thing_stuff_coverage.png
    S3->>FS: semantic_entropy.png
    S3->>FS: divergence_heatmap.png
    S3->>FS: radar_profiles.png
    S3-->>User: ✓ 6 analysis plots saved

    User->>S4: Execute kernel
    S4->>S4: Rank images by paradigm-specific failure scores
    S4->>S4: Select worst-3 per paradigm → failure gallery
    S4->>S4: Select worst-1 per paradigm → edge-case quartet
    S4->>S4: Compute per-stratum Pareto sub-points
    S4->>S4: Weight matrix → decision heatmap (1–5 scores)
    S4->>S4: Assemble 6-panel insight poster
    S4->>FS: failure_gallery.png
    S4->>FS: edge_case_quartet.png
    S4->>FS: pareto_frontier.png
    S4->>FS: decision_matrix.png
    S4->>FS: insight_poster.png
    S4-->>User: ✓ 5 forensic outputs saved

    User->>S5: Execute kernel
    S5->>S5: Build Gradio app (3 tabs)
    S5->>S5: Pre-cache 20 worst-case images
    S5-->>User: Public URL (gradio.live · 72 h)

    User->>S5: Upload image → Tab 1
    S5->>GPU: run_segformer + run_yolo + run_mask2former + run_sam2
    GPU-->>S5: 4× outputs + timing
    S5->>S5: compute_live_metrics()
    S5->>S5: make_comparison_figure() with metric cards
    S5-->>User: 5-panel figure + recommendation badge

    User->>S5: Select failure case → Tab 2
    S5->>S5: Lookup RESULTS[iid] (zero latency)
    S5-->>User: 5-panel + root cause HTML

    User->>S5: Tab 3 — Decision Atlas
    S5-->>User: Static charts + readable decision table
```

---

## 📊 Results & Key Insights

<div align="center">

### ⏱️ Inference Speed (125-image mean on T4)

| Model | Paradigm | Mean | Std | Relative |
|:---:|:---:|---:|---:|:---:|
| YOLOv8x-seg | Instance | `65.2 ms` | ±6.8 | `1.0×` baseline |
| SegFormer-B2 | Semantic | `72.5 ms` | ±4.8 | `1.1×` |
| Mask2Former | Panoptic | `164.2 ms` | ±4.9 | `2.5×` |
| SAM 2 Auto | Promptable | `5539.3 ms` | ±446 | **`85×`** |

</div>

<br/>

<div align="center">

### 🧩 Mean Output Complexity per Scene Stratum

| Stratum | YOLO Instances | Semantic Classes | Panoptic Segs | SAM Masks |
|:---:|:---:|:---:|:---:|:---:|
| 👥 Crowd | 14.6 | 11.2 | 16.0 | **90.0** |
| 🚗 Vehicle | 8.9 | 11.8 | 12.4 | 82.6 |
| 🪑 Indoor | 8.6 | **14.7** | 12.4 | 71.2 |
| 🔀 Mixed | 4.4 | 10.6 | 6.7 | 57.3 |
| 🐾 Animal | 6.2 | 8.8 | 9.5 | 46.7 |

</div>

<br/>

<div align="center">

### 💡 The 6 Forensic Findings

</div>

```
  FINDING 1 — Semantic & Promptable share < 50% pixel agreement globally
  ───────────────────────────────────────────────────────────────────────
  The lowest pairwise IoU in the divergence matrix. They answer
  fundamentally different questions and their outputs are nearly orthogonal.

  FINDING 2 — SAM 2 over-fragments crowds by 6.2× vs. YOLO
  ───────────────────────────────────────────────────────────────────────
  Crowd stratum: SAM mean = 90 masks vs. YOLO mean = 14.6 instances.
  Class-agnostic design breaks every texture into its own segment.

  FINDING 3 — Panoptic is irreplaceable when stuff > 30% of pixels
  ───────────────────────────────────────────────────────────────────────
  Below that threshold, running instance + semantic separately gives
  equivalent coverage at lower compute cost.

  FINDING 4 — SAM 2 produces the sharpest boundaries of all paradigms
  ───────────────────────────────────────────────────────────────────────
  Highest boundary sharpness score across all strata. The 85× speed
  penalty buys you precision — a deliberate engineering trade-off.

  FINDING 5 — Panoptic is the most stable paradigm across scene types
  ───────────────────────────────────────────────────────────────────────
  Smallest quality variance across strata in the Pareto analysis.
  Semantic and SAM shift dramatically by scene; Mask2Former does not.

  FINDING 6 — YOLO is Pareto-optimal for surveillance and e-commerce
  ───────────────────────────────────────────────────────────────────────
  Best count faithfulness × lowest latency for thing-centric tasks.
  Panoptic wins medical and robotics where stuff labelling matters.
```

---

## 🛠️ Tech Stack

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00BFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Supervision](https://img.shields.io/badge/Supervision-Roboflow-purple?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-T4%20GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

<br/>

| Component | Choice | Reason |
|:---:|:---:|:---|
| Semantic model | `SegFormer-B2` | Best mIoU/speed trade-off on ADE20K; transformer encoder |
| Instance model | `YOLOv8x-seg` | Fastest mask inference; clean Ultralytics API |
| Panoptic model | `Mask2Former Swin-L` | 57.8 PQ on COCO panoptic; single-pass things+stuff |
| Promptable model | `SAM 2 hiera-base+` | 2024 foundation model; sharpest boundaries of any tested |
| Visualization | `Matplotlib` dark theme | Consistent dark aesthetic across all 10+ figures |
| Deployment | `Gradio 4.x` | Public URL via `share=True`; zero infrastructure |

</div>

---

## 📁 Project Structure

```
four-paradigms-one-scene/
│
├── 📓 notebooks/
│   ├── snippet_1_foundation.ipynb       # Installs · Data · Models · Utilities
│   ├── snippet_2_inference.ipynb        # Full inference loop · Timing · Gallery
│   ├── snippet_3_metrics.ipynb          # 6 deep analysis plots
│   ├── snippet_4_forensics.ipynb        # Failure gallery · Pareto · Decision matrix
│   └── snippet_5_gradio.ipynb           # Interactive 3-tab Gradio application
│
├── 📊 outputs/
│   ├── stratum_preview.png              # 5×5 sampling validation grid
│   ├── timing_chart.png                 # Speed bar chart with error bars
│   ├── gallery_{crowd,animal,...}.png   # Per-stratum 5-panel figures (×5)
│   ├── boundary_sharpness.png           # Violin plots per stratum
│   ├── count_faithfulness.png           # Scatter grid vs GT count
│   ├── thing_stuff_coverage.png         # Bubble coverage chart
│   ├── semantic_entropy.png             # Box plots (Shannon entropy)
│   ├── divergence_heatmap.png           # 4×4 pairwise paradigm IoU
│   ├── radar_profiles.png               # 5 radar charts (per stratum)
│   ├── failure_gallery.png              # 4×3 worst-case grid
│   ├── edge_case_quartet.png            # 4×5 stress test panels
│   ├── pareto_frontier.png              # Speed-quality Pareto with satellites
│   ├── decision_matrix.png              # 6×4 use-case heatmap (1–5 stars)
│   └── insight_poster.png              # Shareable 24×16 summary poster
│
├── 📋 data/
│   └── (COCO 2017 val — sourced from awsaf49/coco-2017-dataset on Kaggle)
│
└── README.md
```

---

## 🚀 Installation & Reproduction

### Prerequisites

```bash
# GPU: NVIDIA T4 or better  (16 GB VRAM recommended)
# Python 3.10+
# Kaggle dataset: awsaf49/coco-2017-dataset
```

### Quick Start (Kaggle)

```bash
# 1. Add dataset to your Kaggle notebook
#    awsaf49/coco-2017-dataset  →  /kaggle/input/

# 2. Enable GPU accelerator (T4 × 1)

# 3. Run snippets in order — each is a self-contained cell
#    Snippet 1  →  ~5  min  (model downloads + warm-up)
#    Snippet 2  →  ~25 min  (inference loop)
#    Snippet 3  →  ~5  min  (metric computation + 6 plots)
#    Snippet 4  →  ~3  min  (forensic gallery + 5 plots)
#    Snippet 5  →  ~1  min  (Gradio launch → public URL)
```

### Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/four-paradigms-one-scene
cd four-paradigms-one-scene

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45.0 \
            ultralytics>=8.2.0 \
            supervision>=0.21.0 \
            pycocotools \
            gradio>=4.20.0 \
            scipy \
            timm \
            opencv-python

# Download COCO 2017 val
# Place at: ./data/coco2017/val2017/  and  ./data/coco2017/annotations/
```

---

## 🔭 Future Work

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#1a0d05',
  'primaryTextColor': '#ffffff',
  'primaryBorderColor': '#F28E2B',
  'lineColor': '#4E79A7',
  'secondaryColor': '#0d1f2d',
  'tertiaryColor': '#0a130a'
}}}%%

flowchart LR
    NOW["📍 Current\nv1.0\n4 models\n125 images\n6 metrics"]

    subgraph NEXT["🔜 Next — v1.1"]
        N1["Video paradigms\nVOS · VIS · VPS\nSAM 2 video mode"]
        N2["GT Panoptic metrics\nPQ · SQ · RQ\nwith COCO panoptic GT"]
        N3["Quantisation study\nINT8 / FP16 models\nEdge deployment cost"]
    end

    subgraph FUTURE["🌐 Future — v2.0"]
        F1["3D Segmentation\nPoint cloud comparison\nLiDAR scenes"]
        F2["Open-vocabulary\nFC-CLIP · CAT-Seg\nvs closed-vocab"]
        F3["Medical domain\nnnU-Net · MedSAM\nCT / histology strata"]
        F4["Full PQ benchmark\n5000 val images\npublication-ready table"]
    end

    NOW --> N1 & N2 & N3
    N1 & N2 & N3 --> F1 & F2 & F3 & F4
```

---

## 📄 Citation

```bibtex
@misc{fourparadigms2026,
  title   = {Four Paradigms, One Scene: A Forensic Comparison of Segmentation Ontologies},
  author  = {Ahmed},
  year    = {2026},
  note    = {COCO 2017 · SegFormer · YOLOv8x-seg · Mask2Former · SAM 2 · 125 stratified images},
  url     = {https://github.com/YOUR_USERNAME/four-paradigms-one-scene}
}
```

---

<div align="center">

### Referenced Models

[![SegFormer](https://img.shields.io/badge/SegFormer-nvidia%2Fsegformer--b2--finetuned--ade--512--512-blue?style=flat-square&logo=huggingface)](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)
[![YOLOv8](https://img.shields.io/badge/YOLOv8x--seg-ultralytics-00BFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Mask2Former](https://img.shields.io/badge/Mask2Former-facebook%2Fmask2former--swin--large--coco--panoptic-green?style=flat-square&logo=huggingface)](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic)
[![SAM2](https://img.shields.io/badge/SAM_2-facebook%2Fsam2--hiera--base--plus-red?style=flat-square&logo=huggingface)](https://huggingface.co/facebook/sam2-hiera-base-plus)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4E79A7,33:F28E2B,66:59A14F,100:E15759&height=120&section=footer&text=Four%20Paradigms%20%C2%B7%20One%20Scene&fontSize=18&fontColor=ffffff&fontAlignY=65&animation=fadeIn" width="100%"/>

<br/>

**Built with obsessive attention to what segmentation paradigms actually mean**

`Semantic` answers *what* · `Instance` answers *which* · `Panoptic` answers *both* · `Promptable` answers *anything*

<br/>

![Visitors](https://komarev.com/ghpvc/?username=YOUR_USERNAME&label=Profile%20Views&color=4E79A7&style=flat-square)
&nbsp;
[![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/four-paradigms-one-scene?style=flat-square&color=F28E2B)](https://github.com/YOUR_USERNAME/four-paradigms-one-scene)

</div>
