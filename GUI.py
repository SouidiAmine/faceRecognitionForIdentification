# face_identification_gui.py
import sys, os, time
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2

# =========================================================
#                  BACKENDS (unchanged)
# =========================================================
class FaceNetBackend:
    def __init__(self, device='cuda'):
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=self.device)
        self.net = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def detect_and_crop(self, rgb_img):
        pil = Image.fromarray(rgb_img)
        face = self.mtcnn(pil)
        if face is None:
            return None
        return face.unsqueeze(0).to(self.device)

    def embed_from_rgb(self, rgb_img):
        with self.torch.no_grad():
            tens = self.detect_and_crop(rgb_img)
            if tens is None:
                return None
            feat = self.net(tens).cpu().numpy().squeeze()
            return feat / (np.linalg.norm(feat) + 1e-12)

    def detect_boxes(self, bgr_img):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return face_cascade.detectMultiScale(gray, 1.3, 5)


class ArcFaceBackend:
    def __init__(self):
        import insightface
        self.insightface = insightface
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(256, 256))

    def embed_from_rgb(self, rgb_img):
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if len(faces) == 0:
            return None
        f = max(faces, key=lambda z: z.bbox[2]*z.bbox[3])
        vec = f.normed_embedding.astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)

    def detect_boxes(self, bgr_img):
        faces = self.app.get(bgr_img)
        return [tuple(map(int, [f.bbox[0], f.bbox[1], f.bbox[2]-f.bbox[0], f.bbox[3]-f.bbox[1]])) for f in faces]


# =========================================================
#                  GALLERY + IDENTIFICATION
# =========================================================
def load_gallery(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    feats = d["feats"].astype(np.float32)
    labels = d["labels"].astype(str)
    paths = d["paths"].astype(str)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return feats, labels, paths


def cosine_topk(query_feat, gallery_feats, gallery_labels, gallery_paths, k=5):
    sims = gallery_feats @ query_feat
    idx = np.argsort(-sims)[:k]
    return [(gallery_labels[i], float(sims[i]), gallery_paths[i]) for i in idx]


class Identifier:
    def __init__(self, backend_name):
        self.backend_name = backend_name.lower()
        if self.backend_name == "facenet":
            self.backend = FaceNetBackend()
            self.g_feats, self.g_labels, self.g_paths = load_gallery("outputs/facenet_gallery.npz")
        elif self.backend_name == "arcface":
            self.backend = ArcFaceBackend()
            self.g_feats, self.g_labels, self.g_paths = load_gallery("outputs/arcface_gallery.npz")
        else:
            raise ValueError("backend_name must be 'facenet' or 'arcface'")

    def identify_rgb(self, rgb_img, topk=5):
        feat = self.backend.embed_from_rgb(rgb_img)
        if feat is None:
            return None, []
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        return feat, cosine_topk(feat, self.g_feats, self.g_labels, self.g_paths, k=topk)

    def detect_boxes(self, bgr_img):
        return self.backend.detect_boxes(bgr_img)


# =========================================================
#                        GUI
# =========================================================
class ImageLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                color: #bbb;
                border: 1px solid #444;
                border-radius: 6px;
            }""")

    def show_bgr(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Identification — ArcFace & FaceNet (Interactive)")
        self.resize(1200, 750)
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #eee; font-family: Segoe UI; }
            QPushButton { background-color: #2d89ef; color: white; border-radius: 8px; padding: 6px 12px; font-weight: 600; }
            QPushButton:hover { background-color: #1e5fbf; }
            QLabel { font-size: 14px; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #1e1e1e; color: #ddd; border: 1px solid #444; padding: 2px 4px; }
            QTextEdit { background-color: #1e1e1e; color: #ccc; border: 1px solid #333; border-radius: 6px; }
        """)

        # --- Controls ---
        self.btn_upload = QtWidgets.QPushButton("📁 Upload Image")
        self.btn_cam = QtWidgets.QPushButton("🎥 Start Webcam")
        self.model_combo = QtWidgets.QComboBox(); self.model_combo.addItems(["ArcFace", "FaceNet"])
        self.topk_spin = QtWidgets.QSpinBox(); self.topk_spin.setRange(1, 20); self.topk_spin.setValue(5)
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setRange(-1.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.45)
        self.status = QtWidgets.QLabel("Ready."); self.status.setStyleSheet("color:#2ecc71; font-weight:600;")

        tl = QtWidgets.QHBoxLayout()
        for w in [self.btn_upload, self.btn_cam, QtWidgets.QLabel("Model:"), self.model_combo,
                  QtWidgets.QLabel("Top-K:"), self.topk_spin,
                  QtWidgets.QLabel("Threshold:"), self.threshold_spin]:
            tl.addWidget(w)
        tl.addStretch(1)
        tl.addWidget(self.status)

        # --- Views ---
        self.view = ImageLabel(); self.view.setMinimumSize(700, 520)
        self.ref_view = ImageLabel(); self.ref_view.setMinimumSize(260, 200)
        self.ref_caption = QtWidgets.QLabel("Best Match (Reference)")
        self.ref_caption.setAlignment(QtCore.Qt.AlignCenter)
        self.ref_caption.setStyleSheet("font-weight:600; color:#4da3ff;")
        self.result_text = QtWidgets.QTextEdit(); self.result_text.setReadOnly(True)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.ref_caption)
        right.addWidget(self.ref_view)
        right.addWidget(QtWidgets.QLabel("Top-K Results:"))
        right.addWidget(self.result_text, 1)

        main = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addLayout(tl)
        left.addWidget(self.view, 1)
        main.addLayout(left, 2)
        main.addLayout(right, 1)

        # --- State ---
        self.cap = None
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.on_frame)
        self.identifier = None
        self.last_frame = None
        self.switch_model(self.model_combo.currentText())

        # --- Signals ---
        self.btn_upload.clicked.connect(self.on_upload)
        self.btn_cam.clicked.connect(self.on_toggle_cam)
        self.model_combo.currentTextChanged.connect(self.on_model_change)
        self.threshold_spin.valueChanged.connect(self.on_params_changed)
        self.topk_spin.valueChanged.connect(self.on_params_changed)

    # ------------------------ MODEL ------------------------
    def switch_model(self, name):
        self.status.setText(f"Loading {name} model ...")
        QtWidgets.QApplication.processEvents()
        self.identifier = Identifier(name)
        self.status.setText(f"✅ Loaded: {name}")
        if self.last_frame is not None:
            self.identify_and_display(self.last_frame)

    def on_model_change(self, name):
        self.switch_model(name)

    # ------------------------ INTERACTION ------------------------
    def on_params_changed(self):
        """Re-run identification immediately if a parameter changes."""
        if self.last_frame is not None:
            self.identify_and_display(self.last_frame)

    # ------------------------ UPLOAD ------------------------
    def on_upload(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose an Image", "", "Images (*.jpg *.jpeg *.png)")
        if not fn: return
        bgr = cv2.imread(fn)
        if bgr is None:
            self.status.setText("❌ Failed to read image."); self.status.setStyleSheet("color:#e74c3c;")
            return
        self.last_frame = bgr.copy()
        self.identify_and_display(bgr)

    # ------------------------ CAMERA ------------------------
    def on_toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = None
                self.status.setText("❌ Webcam not available."); self.status.setStyleSheet("color:#e74c3c;")
                return
            self.btn_cam.setText("⏹ Stop Webcam")
            self.timer.start(40)
        else:
            self.timer.stop()
            self.cap.release(); self.cap = None
            self.btn_cam.setText("🎥 Start Webcam")

    def on_frame(self):
        ok, frame = self.cap.read()
        if not ok: return
        self.last_frame = frame.copy()
        self.identify_and_display(frame)

    # --------------------- IDENTIFICATION --------------------
    def identify_and_display(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        topk = self.topk_spin.value()
        threshold = self.threshold_spin.value()

        feat, top = self.identifier.identify_rgb(rgb, topk=topk)
        disp = bgr_img.copy()

        try:
            boxes = self.identifier.detect_boxes(bgr_img)
            for (x, y, w, h) in boxes[:1]:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        except Exception:
            pass

        if not top:
            self.status.setText("❌ No face detected."); self.status.setStyleSheet("color:#e74c3c;")
            self.view.show_bgr(disp); self.result_text.clear(); self.ref_view.clear(); return

        best_id, best_sim, best_path = top[0]
        if best_sim >= threshold:
            self.status.setText(f"✅ Match: {best_id} (sim={best_sim:.3f} ≥ {threshold:.3f})")
            self.status.setStyleSheet("color:#2ecc71; font-weight:600;")
        else:
            self.status.setText(f"❌ Unknown (sim={best_sim:.3f} < {threshold:.3f})")
            self.status.setStyleSheet("color:#e74c3c; font-weight:600;")

        self.view.show_bgr(disp)
        self.result_text.setText("\n".join([f"{i:>2}. {pid:<20} cos={sim:.3f}" for i, (pid, sim, _) in enumerate(top, 1)]))
        ref_bgr = cv2.imread(best_path)
        if ref_bgr is not None:
            self.ref_view.show_bgr(ref_bgr)


# =========================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
