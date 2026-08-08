import pytest
import numpy as np
import cv2
import pickle
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_pro"


@pytest.fixture(scope="session")
def detector():
    from uniface import create_detector
    return create_detector("retinaface")


@pytest.fixture(scope="session")
def recognizer():
    from uniface import create_recognizer
    return create_recognizer("arcface")


@pytest.fixture(scope="session")
def spoofer():
    from uniface.spoofing import create_spoofer
    return create_spoofer()


@pytest.fixture
def sample_face_jorge():
    ruta = DATASET_DIR / "Jorge" / "Jorge_0.jpg"
    img = cv2.imread(str(ruta))
    assert img is not None, f"No se pudo cargar {ruta}"
    return img


@pytest.fixture
def sample_face_lino():
    ruta = DATASET_DIR / "Lino313" / "video_lino_0.jpg"
    img = cv2.imread(str(ruta))
    assert img is not None, f"No se pudo cargar {ruta}"
    return img


@pytest.fixture
def sample_face_valu():
    ruta = DATASET_DIR / "Valu313" / "Valu313_0.jpg"
    img = cv2.imread(str(ruta))
    assert img is not None, f"No se pudo cargar {ruta}"
    return img


@pytest.fixture
def video_lino_path():
    ruta = DATASET_DIR / "video_lino.mp4"
    assert ruta.exists(), f"No se encontro {ruta}"
    return str(ruta)


@pytest.fixture
def synthetic_face():
    img = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
    cv2.circle(img, (90, 90), 15, (40, 40, 40), -1)
    cv2.circle(img, (134, 90), 15, (40, 40, 40), -1)
    cv2.ellipse(img, (112, 140), (25, 10), 0, 0, 180, (60, 60, 60), 2)
    cv2.circle(img, (112, 115), 5, (50, 50, 50), -1)
    return img


@pytest.fixture
def synthetic_blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_noise():
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_blurry():
    img = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (31, 31), 15)


@pytest.fixture
def synthetic_dark():
    return np.full((224, 224, 3), 10, dtype=np.uint8)


@pytest.fixture
def synthetic_bright():
    return np.full((224, 224, 3), 245, dtype=np.uint8)


@pytest.fixture
def synthetic_sharp():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 4):
        cv2.line(img, (i, 0), (i, 224), (255, 255, 255), 1)
        cv2.line(img, (0, i), (224, i), (255, 255, 255), 1)
    return img


@pytest.fixture
def tmp_db_path(tmp_path):
    return tmp_path / "test_db.pkl"


@pytest.fixture
def sample_db(tmp_db_path):
    db = {
        "test_user_1": np.random.randn(512).astype("float32"),
        "test_user_2": np.random.randn(512).astype("float32"),
    }
    for k in db:
        db[k] = db[k] / np.linalg.norm(db[k])
    with open(tmp_db_path, "wb") as f:
        pickle.dump(db, f)
    return db


@pytest.fixture
def fake_landmarks_5pts():
    return np.array([
        [80.0, 90.0],
        [140.0, 90.0],
        [112.0, 120.0],
        [95.0, 150.0],
        [130.0, 150.0],
    ], dtype=np.float32)


@pytest.fixture
def fake_bbox():
    return np.array([50.0, 50.0, 180.0, 200.0], dtype=np.float32)


@pytest.fixture
def frame_with_face():
    cap = cv2.VideoCapture(str(DATASET_DIR / "video_lino.mp4"))
    ret, frame = cap.read()
    cap.release()
    assert ret, "No se pudo leer el video"
    return frame
