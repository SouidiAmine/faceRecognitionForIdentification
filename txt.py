import insightface
app = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256,256))
print("InsightFace OK ✅")
