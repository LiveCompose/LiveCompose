import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


ADACROP_ROOT = Path(__file__).resolve().parents[1]
if str(ADACROP_ROOT) not in sys.path:
    sys.path.insert(0, str(ADACROP_ROOT))

from distillation.common import ACTIONS, load_teacher  # noqa: E402


class TeacherBBoxOnly(nn.Module):
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher

    def forward(self, full_img: torch.Tensor):
        return self.teacher.backbone_forward(full_img)


class TeacherActorOnly(nn.Module):
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher

    def forward(self, crop_img: torch.Tensor, state: torch.Tensor):
        probs, _ = self.teacher(crop_img, state)
        return probs


def parse_args():
    parser = argparse.ArgumentParser(description="Export teacher ResNet50 Adacrop policy to Core ML.")
    parser.add_argument("--teacher-ckpt", type=Path, default=ADACROP_ROOT.parent / "ppo_best_val_final_score.pth")
    parser.add_argument("--out-dir", type=Path, default=ADACROP_ROOT / "coreml_export" / "teacher")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--ios-target", type=str, default="iOS16", choices=["iOS15", "iOS16", "iOS17", "iOS18"])
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"])
    return parser.parse_args()


def target_from_name(ct, name: str):
    return {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }[name]


def convert_and_save(traced, inputs, output_name: str, save_path: Path, args):
    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit(
            "coremltools is not installed. Install it first, for example:\n"
            "  python -m pip install coremltools\n"
        ) from exc

    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name=output_name)],
        minimum_deployment_target=target_from_name(ct, args.ios_target),
        compute_precision=precision,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(save_path))
    print(f"[save] {save_path}")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_teacher(args.teacher_ckpt, torch.device("cpu")).eval()

    dummy_img = torch.rand(1, 3, args.img_size, args.img_size)
    dummy_state = torch.tensor([[0.5, 0.5, 0.6, 0.6]], dtype=torch.float32)

    bbox_model = TeacherBBoxOnly(teacher).eval()
    actor_model = TeacherActorOnly(teacher).eval()

    with torch.no_grad():
        traced_bbox = torch.jit.trace(bbox_model, dummy_img)
        traced_actor = torch.jit.trace(actor_model, (dummy_img, dummy_state))

    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit(
            "coremltools is not installed. Install it first, for example:\n"
            "  python -m pip install coremltools\n"
        ) from exc

    print(f"[info] actions: {ACTIONS}")
    print(f"[info] teacher checkpoint: {args.teacher_ckpt}")

    convert_and_save(
        traced_bbox,
        inputs=[ct.TensorType(name="full_img", shape=dummy_img.shape)],
        output_name="bbox",
        save_path=args.out_dir / "AdacropTeacherBBox.mlpackage",
        args=args,
    )
    convert_and_save(
        traced_actor,
        inputs=[
            ct.TensorType(name="crop_img", shape=dummy_img.shape),
            ct.TensorType(name="state", shape=dummy_state.shape),
        ],
        output_name="action_probs",
        save_path=args.out_dir / "AdacropTeacherActor.mlpackage",
        args=args,
    )


if __name__ == "__main__":
    main()
