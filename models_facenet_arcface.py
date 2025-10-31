# models_facenet_arcface.py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

# ------------------ FaceNet (facenet-pytorch) ------------------
class FaceNetEmbedder:
    def __init__(self, device='cuda'):
        from facenet_pytorch import InceptionResnetV1
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.net = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.tf = T.Compose([
            T.Resize((160, 160)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        x = self.tf(img).unsqueeze(0).to(self.device)  # (1,3,160,160)
        feat = self.net(x).cpu().numpy().squeeze()     # 512-D
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        return feat

# ------------------ ArcFace (InsightFace buffalo_l) -------------
class ArcFaceEmbedder:
    def __init__(self):
        import insightface
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=['CPUExecutionProvider']
        )
        # ctx_id = 0 for GPU (if available). For CPU only, keep ctx_id=0, providers CPU.
        self.app.prepare(ctx_id=0, det_size=(256, 256))

    def embed(self, img: Image.Image) -> np.ndarray:
        # InsightFace expects BGR np.uint8
        import cv2
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if len(faces) == 0:
            # fallback: resize center crop (rare with LFW)
            vec = np.zeros(512, dtype=np.float32)
        else:
            # take the largest face
            f = max(faces, key=lambda z: z.bbox[2]*z.bbox[3])
            vec = f.normed_embedding.astype(np.float32)  # already L2-normalized (512-D)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec
