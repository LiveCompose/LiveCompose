import torch
import onnx
from src.model import ActorCritic

def export_onnx(ckpt_path, onnx_path, opset=13):
    model = ActorCritic(n_actions=10)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    dummy_img = torch.randn(1,3,224,224)
    dummy_state = torch.randn(1,4)
    torch.onnx.export(
        model, (model.extract_feats(dummy_img), dummy_state),
        onnx_path, opset_version=opset,
        input_names=["img_feats", "state"],
        output_names=["action_probs","value"]
    )
    print("ONNX 导出完成：", onnx_path)

if __name__ == "__main__":
    export_onnx("best.pth", "adacrop.onnx")
