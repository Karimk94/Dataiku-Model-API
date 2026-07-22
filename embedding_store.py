import json
import os
import threading
import numpy as np


class EmbeddingStore:
    """
    Thread-safe store for face embedding vectors.

    Stores embeddings as {name: [[float, ...], ...]} in a JSON file on disk.
    Each person can have multiple embedding vectors (one per enrolled photo).
    Recognition compares a query embedding against all stored embeddings using
    euclidean L2 distance — the same metric DeepFace uses with VGG-Face.

    An embedding is a 2,622-dimension float vector (~21 KB). No face images
    are stored — only the numerical representations.
    """

    def __init__(self, storage_path="embeddings.json"):
        self.storage_path = os.path.abspath(storage_path)
        self._lock = threading.Lock()
        self._embeddings = {}  # {name: [embedding_vector, ...]}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load embeddings from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self._embeddings = json.load(f)
                total = sum(len(v) for v in self._embeddings.values())
                print(f"Loaded {total} embedding(s) for {len(self._embeddings)} person(s).")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load embeddings file ({e}). Starting fresh.")
                self._embeddings = {}
        else:
            self._embeddings = {}
            print("No existing embeddings found. Starting with an empty store.")

    def _save(self):
        """Persist embeddings to disk (must be called while holding the lock)."""
        with open(self.storage_path, "w") as f:
            json.dump(self._embeddings, f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, name, embedding):
        """
        Add an embedding vector for *name*.

        Parameters
        ----------
        name : str
            Person identifier (e.g. ``"ahmed_bahrozyan"``).
        embedding : list[float]
            The 2,622-dim embedding vector produced by ``DeepFace.represent()``.
        """
        with self._lock:
            if name not in self._embeddings:
                self._embeddings[name] = []
            self._embeddings[name].append(embedding)
            self._save()

    def remove(self, name):
        """Remove **all** embeddings for *name*. Returns ``True`` if found."""
        with self._lock:
            if name in self._embeddings:
                del self._embeddings[name]
                self._save()
                return True
            return False

    def list_faces(self):
        """Return ``{name: embedding_count}`` for every enrolled person."""
        with self._lock:
            return {name: len(embs) for name, embs in self._embeddings.items()}

    def find_closest(self, embedding, threshold=0.9):
        """
        Find the enrolled person whose embedding is closest to *embedding*.

        Uses euclidean L2 distance (``np.linalg.norm``), consistent with
        DeepFace's ``euclidean_l2`` metric for VGG-Face.

        Parameters
        ----------
        embedding : list[float]
            Query embedding vector.
        threshold : float
            Maximum distance to consider a match (default 0.9, same as the
            original ``face_processor.py`` threshold of ``<= 0.9``).

        Returns
        -------
        tuple[str, float | None]
            ``(name, distance)`` of the best match, or ``("Unknown", None)``
            if no match is found within *threshold*.
        """
        with self._lock:
            if not self._embeddings:
                return "Unknown", None

            query = np.array(embedding, dtype=np.float64)
            best_name = "Unknown"
            best_distance = float("inf")

            for name, stored_embeddings in self._embeddings.items():
                for stored in stored_embeddings:
                    distance = float(np.linalg.norm(query - np.array(stored, dtype=np.float64)))
                    if distance < best_distance:
                        best_distance = distance
                        best_name = name

            if best_distance <= threshold:
                return best_name, round(best_distance, 6)
            return "Unknown", None
