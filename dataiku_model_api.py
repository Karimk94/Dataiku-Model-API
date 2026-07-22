"""
Dataiku Model API
=================
A lightweight Flask API that hosts DeepFace models for face detection and
recognition.  This service stores **only embedding vectors** (arrays of floats)
— never actual face images.  Thumbnails and face images are managed by the
companion Thumbnail API.

Endpoints
---------
POST /detect_faces      — Detect face locations in a base64-encoded image.
POST /recognize_faces   — Recognize pre-cropped face thumbnails.
POST /enroll_face       — Compute and store an embedding for a new face.
POST /delete_face       — Remove all embeddings for a person.
GET  /list_faces        — List all enrolled people and embedding counts.
GET  /health            — Health-check / readiness probe.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
from deepface.commons import folder_utils
from PIL import Image
import numpy as np
import io
import base64
import os
import shutil
import random

from embedding_store import EmbeddingStore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

EMBEDDINGS_FILE = os.environ.get(
    "EMBEDDINGS_FILE",
    os.path.join(os.path.dirname(__file__), "embeddings.json"),
)
embedding_store = EmbeddingStore(storage_path=EMBEDDINGS_FILE)


# ---------------------------------------------------------------------------
# Model initialisation (offline — no internet required)
# ---------------------------------------------------------------------------

def _initialize_offline_models():
    """
    Pre-populate the DeepFace cache with local model weight files, then build
    the VGG-Face model.  This ensures no download is attempted at runtime.
    """
    print("Initializing offline models...")

    deepface_cache_path = folder_utils.get_deepface_home()
    weights_path = os.path.join(deepface_cache_path, "weights")
    os.makedirs(weights_path, exist_ok=True)

    local_model_source_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "models")
    )
    required_models = ["vgg_face_weights.h5", "retinaface.h5"]

    for model_file in required_models:
        source_path = os.path.join(local_model_source_dir, model_file)
        dest_path = os.path.join(weights_path, model_file)

        if not os.path.exists(dest_path):
            print(f"  Model '{model_file}' not in cache — copying from local source...")
            if not os.path.exists(source_path):
                raise RuntimeError(
                    f"CRITICAL: Source model file not found at '{source_path}'."
                )
            shutil.copy(source_path, dest_path)
            print(f"  Copied '{model_file}' to cache.")

    print("  Building VGG-Face model from cache...")
    DeepFace.build_model("VGG-Face")
    print("Models initialised successfully.\n")


_initialize_offline_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64_to_image(b64_string):
    """Decode a base64 string into a PIL Image (RGB)."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _save_temp_image(image, prefix="temp"):
    """Save a PIL Image to a temporary JPEG and return its path."""
    path = os.path.join(
        os.path.dirname(__file__),
        f"{prefix}_{random.randint(1, 99999)}.jpg",
    )
    image.save(path, "JPEG")
    return path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    """
    Detect faces in a base64-encoded image.

    Request JSON::

        { "image_b64": "<base64 string>" }

    Response JSON::

        { "faces": [
            { "index": 1, "x": 100, "y": 50, "w": 80, "h": 100, "confidence": 0.99 },
            ...
        ]}
    """
    data = request.get_json(silent=True)
    if not data or "image_b64" not in data:
        return jsonify({"error": "Missing 'image_b64' in request body."}), 400

    temp_path = None
    try:
        image = _b64_to_image(data["image_b64"])
        temp_path = _save_temp_image(image, prefix="detect")

        detected = DeepFace.extract_faces(
            img_path=temp_path,
            enforce_detection=False,
            detector_backend="retinaface",
        )

        faces = []
        if detected and detected[0]["facial_area"]["w"] > 0:
            for i, face in enumerate(detected):
                area = face["facial_area"]
                faces.append({
                    "index": i + 1,
                    "x": int(area["x"]),
                    "y": int(area["y"]),
                    "w": int(area["w"]),
                    "h": int(area["h"]),
                    "confidence": float(face.get("confidence", 0)),
                })

        return jsonify({"faces": faces})

    except Exception as e:
        app.logger.error(f"Error in /detect_faces: {e}", exc_info=True)
        return jsonify({"error": f"Detection failed: {e}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/recognize_faces", methods=["POST"])
def recognize_faces():
    """
    Recognize pre-cropped face thumbnails against enrolled embeddings.

    Request JSON::

        { "faces": ["<b64_crop_1>", "<b64_crop_2>", ...] }

    Response JSON::

        { "faces": [
            { "index": 1, "name": "ahmed_bahrozyan", "distance": 0.45 },
            { "index": 2, "name": "Unknown", "distance": null },
            ...
        ]}
    """
    data = request.get_json(silent=True)
    if not data or "faces" not in data or not isinstance(data["faces"], list):
        return jsonify({"error": "Request must contain a 'faces' array."}), 400

    results = []
    for i, b64_face in enumerate(data["faces"]):
        try:
            img = _b64_to_image(b64_face)
            img_np = np.array(img)

            embedding_result = DeepFace.represent(
                img_path=img_np,
                model_name="VGG-Face",
                enforce_detection=False,
                detector_backend="skip",
            )

            if embedding_result:
                embedding = embedding_result[0]["embedding"]
                name, distance = embedding_store.find_closest(embedding)
                results.append({"index": i + 1, "name": name, "distance": distance})
            else:
                results.append({"index": i + 1, "name": "Unknown", "distance": None})

        except Exception as e:
            app.logger.error(f"Error recognising face {i + 1}: {e}", exc_info=True)
            results.append({"index": i + 1, "name": "Unknown", "distance": None})

    return jsonify({"faces": results})


@app.route("/enroll_face", methods=["POST"])
def enroll_face():
    """
    Enroll a face: compute its embedding and store it (no image is saved).

    Request JSON::

        { "name": "Ahmed Bahrozyan", "face_b64": "<base64 crop>" }

    Response JSON::

        { "success": true, "message": "Embedding stored for 'ahmed_bahrozyan'." }
    """
    data = request.get_json(silent=True)
    if not data or "name" not in data or "face_b64" not in data:
        return jsonify({"error": "Missing 'name' or 'face_b64'."}), 400

    try:
        name = data["name"].replace(" ", "_").lower()
        img = _b64_to_image(data["face_b64"])
        img_np = np.array(img)

        embedding_result = DeepFace.represent(
            img_path=img_np,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="skip",
        )

        if not embedding_result:
            return jsonify({"error": "Could not compute face embedding."}), 500

        embedding = embedding_result[0]["embedding"]
        embedding_store.add(name, embedding)

        return jsonify({
            "success": True,
            "message": f"Embedding stored for '{name}'.",
        })

    except Exception as e:
        app.logger.error(f"Error in /enroll_face: {e}", exc_info=True)
        return jsonify({"error": f"Enrollment failed: {e}"}), 500


@app.route("/delete_face", methods=["POST"])
def delete_face():
    """
    Delete all embeddings for a person.

    Request JSON::

        { "name": "ahmed_bahrozyan" }
    """
    data = request.get_json(silent=True)
    if not data or "name" not in data:
        return jsonify({"error": "Missing 'name'."}), 400

    name = data["name"].replace(" ", "_").lower()
    removed = embedding_store.remove(name)

    if removed:
        return jsonify({"success": True, "message": f"Removed all embeddings for '{name}'."})
    return jsonify({"success": False, "message": f"'{name}' not found in store."})


@app.route("/list_faces", methods=["GET"])
def list_faces():
    """List all enrolled people and their embedding counts."""
    return jsonify({"faces": embedding_store.list_faces()})


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint."""
    faces = embedding_store.list_faces()
    return jsonify({
        "status": "ok",
        "enrolled_people": len(faces),
        "total_embeddings": sum(faces.values()),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        from waitress import serve as waitress_serve

        port = int(os.environ.get("HTTP_PLATFORM_PORT", 5003))
        print(f"Starting Dataiku Model API on port {port}...")
        waitress_serve(app, host="0.0.0.0", port=port, threads=100)
    except ImportError:
        # Fallback to Flask dev server if waitress is not available
        port = int(os.environ.get("HTTP_PLATFORM_PORT", 5003))
        print(f"Starting Dataiku Model API (dev mode) on port {port}...")
        app.run(host="0.0.0.0", port=port, debug=True)
