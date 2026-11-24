"""Turn a fine-tuned CLIP visual state_dict (fp16 .pt) into the app's ONNX format:
input 'pixel_values' (N,3,224,224) -> L2-normalised 512-d embedding; weights stored as float16, split <95 MB."""
import sys, torch, numpy as np, onnx, open_clip, io
from onnx import numpy_helper
sd_path, out = sys.argv[1], sys.argv[2]
model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
v = model.visual.float().eval()
if sd_path != "original":
    sd = torch.load(sd_path, map_location="cpu"); v.load_state_dict({k: t.float() for k, t in sd.items()})
class E(torch.nn.Module):
    def __init__(s): super().__init__(); s.v = v
    def forward(s, pixel_values):
        e = s.v(pixel_values); return e / e.norm(dim=-1, keepdim=True)
buf = io.BytesIO()
torch.onnx.export(E(), torch.randn(1, 3, 224, 224), buf, input_names=["pixel_values"], output_names=["image_embeds"],
                  dynamic_axes={"pixel_values": {0: "n"}, "image_embeds": {0: "n"}}, opset_version=17, dynamo=False)
m = onnx.load_from_string(buf.getvalue())
for init in m.graph.initializer:
    if init.data_type == onnx.TensorProto.FLOAT and numpy_helper.to_array(init).size > 1024:
        init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))
data = m.SerializeToString(); P = 95 * 2**20
for i in range(0, len(data), P):
    open(f"{out}.part{i // P}", "wb").write(data[i:i+P])
print("wrote", len(data) // 2**20, "MB in", (len(data) + P - 1) // P, "parts")
