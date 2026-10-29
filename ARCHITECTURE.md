# Dataiku Model API — Architecture & System Design

This API is the **stateless machine learning perception engine** for face detection and dense vector representation in the Smart EDMS ecosystem. It isolates deep neural network computation (TensorFlow, RetinaFace, VGG-Face) from application logic.

---

## 1. System Overview & The Three-Tier AI Pipeline

In the Smart EDMS architecture, **Dataiku Model API** serves as the **ML Perception Middleware** bridging the application gateway ([Face Recognition py](file:///d:/projects/new%20EDMS%20py/Face%20Recognition%20py)) with cognitive enterprise services (**GovAI LLM**):

```mermaid
graph TD
    subgraph Tier1 ["Tier 1: Gateway & Application Orchestration"]
        FR["Face Recognition py<br/>(Port 5002 - FastAPI)"]
        CDB[("ChromaDB<br/>(faces collection - L2 distance)")]
        PIL["Pillow Engine<br/>(Crops, Bounding Boxes, Labels)"]
        FR <--> CDB
        FR <--> PIL
    end

    subgraph Tier2 ["Tier 2: ML Perception Middleware (This Service)"]
        DK["Dataiku Model API<br/>(Port 5009 / Dataiku DSS :13000)"]
        RF["RetinaFace Detector<br/>(Bounding Box Coordinates)"]
        VGG["VGG-Face Extractor<br/>(4096-dim Vector Embeddings)"]
        DK --> RF
        DK --> VGG
    end

    subgraph Tier3 ["Tier 3: Cognitive & Linguistic Intelligence"]
        TRANS["Translator Service<br/>(Port 5010)"]
        GOVAI["GovAI LLM Platform<br/>(llmapi.govai.ae)"]
        TRANS <--> GOVAI
    end

    FR -->|"1. POST /detect_faces (Base64 Image)<br/>2. POST /get_embeddings (Base64 Crops)"| DK
    DK -->|"Returns Coordinates [x,y,w,h]<br/>& Dense Embedding Vectors"| FR
    FR -->|"3. Transliterate Arabic Names"| TRANS
```

---

## 2. What This API Does vs. What It Does NOT Do

### What This API Does
| Responsibility | Details |
|---|---|
| **Facial Detection** | Uses RetinaFace deep neural network to locate faces and return bounding box coordinates `[x, y, w, h]` and confidence scores. |
| **Dense Vector Embeddings** | Computes high-dimensional feature representations (4096-dimensional vectors) via VGG-Face. |
| **Offline Model Bootstrapping** | Pre-loads local `.h5` model weights (`vgg_face_weights.h5`, `retinaface.h5`) into the DeepFace cache during startup. |
| **Stateless Concurrency** | Processes all requests in-memory using Waitress WSGI with 100 worker threads. |

### What This API Does NOT Do
| Responsibility | Handled By |
|---|---|
| **Store Face Images or Thumbnails** | `Face Recognition py` (Gateway) |
| **Maintain Vector Databases** | `Face Recognition py` (ChromaDB `faces` collection) |
| **Draw Visual Annotations / Bounding Boxes** | `Face Recognition py` (Pillow engine) |
| **Bilingual Name Translation** | `GovAI LLM` (via Translator Service) |
| **Direct Client / EDMS Ingestion** | `Face Recognition py` (FastAPI endpoints) |

---

## 3. Data Flow: Full Pipeline Interaction

```mermaid
sequenceDiagram
    autonumber
    actor User as Smart EDMS / ..\EDMS API
    participant GW as Face Recognition py<br/>(Gateway · :5002)
    participant DK as Dataiku Model API<br/>(Middleware · :5009)
    participant CDB as ChromaDB<br/>(Vector Store)
    participant LLM as GovAI LLM<br/>(llmapi.govai.ae)

    User->>GW: POST /api/analyze_image_stream (Image Bytes)
    Note over GW: Base64-encode image

    GW->>DK: POST /detect_faces {"image_b64": "..."}
    Note over DK: RetinaFace inference
    DK-->>GW: {"faces": [{"x": 100, "y": 50, "w": 80, "h": 100, "confidence": 0.99}]}

    Note over GW: Crop face thumbnails with Pillow
    GW->>DK: POST /get_embeddings {"faces": ["<crop_1>", ...]}
    Note over DK: VGG-Face inference
    DK-->>GW: {"embeddings": [[0.012, -0.048, ...]]}

    GW->>CDB: collection.query(embeddings=[...], n_results=1)
    CDB-->>GW: Match: "john_doe" (L2 distance: 0.28)

    opt If new face enrolled with Arabic name
        GW->>LLM: Translate "جون دو" to English
        LLM-->>GW: "John Doe"
    end

    Note over GW: Draw green bounding boxes & text labels
    GW-->>User: {faces: [...], processed_image: "...", original_image_b64: "..."}
```

---

## 4. Endpoints & Interface Contract

### `POST /detect_faces`
Receives a Base64-encoded image and returns an array of detected face boxes.

* **Request**:
  ```json
  { "image_b64": "<base64_string>" }
  ```
* **Response**:
  ```json
  {
    "faces": [
      { "index": 1, "x": 100, "y": 50, "w": 80, "h": 100, "confidence": 0.99 }
    ]
  }
  ```

---

### `POST /get_embeddings`
Computes mathematical vector representations for an array of pre-cropped face images.

* **Request**:
  ```json
  {
    "faces": [
      "<base64_crop_1>",
      "<base64_crop_2>"
    ]
  }
  ```
* **Response**:
  ```json
  {
    "embeddings": [
      [0.0124, -0.0481, 0.0921, "... (4096 floats) ..."],
      null
    ]
  }
  ```

---

### `GET /health`
Readiness and liveness probe for monitoring tools and load balancers.

* **Response**:
  ```json
  {
    "status": "ok",
    "service": "Dataiku Model API (Stateless)"
  }
  ```

---

## 5. Deployment Topology

1. **Local Windows Server (IIS / Waitress)**:
   - Configured via `web.config` with `HttpPlatformHandler`.
   - Runs `dataiku_model_api.py` served by Waitress with 100 threads on port `5009`.
2. **Enterprise Dataiku DSS API Deployer**:
   - The same endpoint logic is deployed as a managed service on Dataiku DSS (`https://dataiku-model-api.example.com:13000`).
   - The gateway (`Face Recognition py`) automatically unwraps Dataiku Deployer envelopes (`{"response": ..., "timing": ...}`).
