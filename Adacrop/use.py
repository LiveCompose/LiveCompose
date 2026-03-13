import torch
from pathlib import Path
from PIL import Image, ImageDraw

from src.config import Config
from src.env import CropEnv
from src.model import ActorCritic

IMAGE_PATH = r"./Adacrop/data/bad2.jpg" 
# IMAGE_PATH = r"./Adacrop/data/GAIC_dataset/images/test/238563.jpg"          
CKPT_PATH  = r"./Adacrop/logs/run_20260312_191504_gaic_norm_gpu0_env128/ppo_best_train_reward.pth" 

OUT_DIR    = r"./Adacrop/output_img"
MAX_STEPS  = 70
MIN_STEPS_NO_STOP = 15 

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    cfg = Config()  # 默认 ./config.yaml

    img = Image.open(IMAGE_PATH).convert("RGB")
    # inference=True -> 不会调用 NIMA
    env = CropEnv(img, aesthetic_model=None, cfg=cfg, inference=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(n_actions=len(env.actions)).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[load] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    print("[load] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    
    stop_idx = env.actions.index("stop")
    last = model.actor[-1]  # Linear -> n_actions
    print("[actor head] weight shape:", tuple(last.weight.shape), "bias shape:", tuple(last.bias.shape))
    print("[actor head] stop_bias:", float(last.bias[stop_idx]))

    model.eval()

    state = env.reset()
    traj = [tuple(env.box)]

    for t in range(MAX_STEPS):
        prev_box = tuple(float(x) for x in env.box) 
        img_t = state[0].unsqueeze(0).to(device)
        st_t  = state[1].unsqueeze(0).to(device)
        with torch.no_grad():
            probs, _ = model(img_t, st_t)  # [1, n_actions]
        act_idx = int(torch.argmax(probs, dim=1).item())

        topk = torch.topk(probs[0], k=min(3, probs.shape[1]))
        top_str = ", ".join([f"{env.actions[i]}={topk.values[j].item():.3f}"
                             for j, i in enumerate(topk.indices.tolist())])

        probs_np = probs[0].cpu().numpy()
        if t < MIN_STEPS_NO_STOP:
            probs_np[stop_idx] = -1e9  # mask 掉 stop
        act_idx = int(probs_np.argmax())

        if env.actions[act_idx] == "stop":
            print(f"Step {t:02d}: action=stop | box={tuple(round(x,1) for x in prev_box)}")
            break

        state, _, done, _ = env.step(act_idx)
        new_box = tuple(float(x) for x in env.box)
        print(f"Step {t:02d}: action={env.actions[act_idx]:>8} | "
              f"{tuple(round(x,1) for x in prev_box)} -> {tuple(round(x,1) for x in new_box)} | "
              f"top3: {top_str}")
        traj.append(new_box)

        if done:
            break

    x, y, w, h = env.box
    final = img.crop((x, y, x + w, y + h))
    final_path = Path(OUT_DIR) / (Path(IMAGE_PATH).stem + "_crop" + Path(IMAGE_PATH).suffix)
    final.save(final_path, quality=95)

    # 可视化轨迹
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    for i, (bx, by, bw, bh) in enumerate(traj):
        if i == 0:
            color = (0, 0, 255)      # 起始框：蓝色
        elif i == len(traj) - 1:
            color = (255, 0, 0)      # 最终框：红色
        else:
            color = (255, 255, 0)    # 中间轨迹：黄色
        draw.rectangle([bx, by, bx + bw, by + bh], outline=color, width=2)
    traj_path = Path(OUT_DIR) / (Path(IMAGE_PATH).stem + "_traj" + Path(IMAGE_PATH).suffix)
    vis.save(traj_path, quality=95)

    print(f"Saved: {final_path} (traj: {traj_path})  box={env.box}")

if __name__ == "__main__":
    main()