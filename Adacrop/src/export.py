import torch
import onnx
try:
    from src.model import ActorCritic
except ModuleNotFoundError:
    from model import ActorCritic

def export_onnx(ckpt_path, onnx_path, opset=13):
    #model = ActorCritic(n_actions=11)
    model = ActorCritic(n_actions=7)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    dummy_img = torch.randn(1,3,224,224)
    dummy_state = torch.randn(1,4)
    # torch.onnx.export(
    #     model, (model.extract_feats(dummy_img), dummy_state),
    #     onnx_path, opset_version=opset,
    #     input_names=["img_feats", "state"],
    #     output_names=["action_probs","value"]
    # )
    torch.onnx.export(
        model, (dummy_img, dummy_state),
        onnx_path, opset_version=opset,
        input_names=["img", "state"],
        output_names=["action_probs","value"],
        dynamic_axes={
            "img": {0: "batch"},
            "state": {0: "batch"},
            "action_probs": {0: "batch"},
            "value": {0: "batch"},
        }
    )
    print("ONNX 导出完成：", onnx_path)

if __name__ == "__main__":
    export_onnx("best.pth", "adacrop.onnx")
