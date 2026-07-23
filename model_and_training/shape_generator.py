from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from diffusion_process.forward_diffusion import Diffusion
from model_and_training.training import NoisePredictor


@torch.no_grad()
def sample_points(
    model,
    diffusion,
    n_points=1000,
    keyframe_timesteps=None,
):
    model.eval()
    device = diffusion.device
    x = torch.randn((n_points, 2), device=device)

    if keyframe_timesteps is None:
        last_timestep = diffusion.noise_steps - 1
        keyframe_timesteps = (
            last_timestep,
            round(last_timestep * 0.75),
            round(last_timestep * 0.50),
            round(last_timestep * 0.20),
            0,
        )

    keyframes = {diffusion.noise_steps - 1: x.clone()}

    for t in tqdm(
        reversed(range(1, diffusion.noise_steps)),
        total=diffusion.noise_steps - 1,
        desc="Generating points",
    ):
        timesteps = torch.full(
            (n_points,),
            t,
            dtype=torch.long,
            device=device,
        )

        predicted_noise = model(x, timesteps)
        alpha_t = diffusion.alphas[t]
        alpha_hat_t = diffusion.alpha_hat[t]
        beta_t = diffusion.betas[t]

        model_mean = (1 / torch.sqrt(alpha_t)) * (
            x
            - (beta_t / torch.sqrt(1 - alpha_hat_t))
            * predicted_noise
        )

        if t > 1:
            alpha_hat_previous = diffusion.alpha_hat[t - 1]
            posterior_variance = beta_t * (
                (1 - alpha_hat_previous) / (1 - alpha_hat_t)
            )
            x = (
                model_mean
                + torch.sqrt(posterior_variance) * torch.randn_like(x)
            )
        else:
            x = model_mean

        current_timestep = t - 1
        if current_timestep in keyframe_timesteps:
            keyframes[current_timestep] = x.clone()

    return x, keyframes


def plot_denoising_process(original_points, keyframes, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].scatter(original_points[:, 0], original_points[:, 1], s=5)
    axes[0].set_title("Original circle")

    for ax, timestep in zip(axes[1:], sorted(keyframes, reverse=True)):
        points = keyframes[timestep]
        ax.scatter(points[:, 0], points[:, 1], s=5)
        ax.set_title(f"Reverse timestep {timestep}")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Circle Denoising Process", fontsize=16)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved denoising plot to {output_path}")
    plt.show()


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run the training module first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    diffusion = Diffusion(
        noise_steps=checkpoint["noise_steps"],
        beta_start=checkpoint["beta_start"],
        beta_end=checkpoint["beta_end"],
    )
    model = NoisePredictor(checkpoint["noise_steps"]).to(diffusion.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, diffusion, checkpoint


def main():
    model, diffusion, checkpoint = load_model(
        "models/circle_model.pt",
    )

    generated_normalized, normalized_keyframes = sample_points(
        model,
        diffusion,
        n_points=1000,
    )

    mean = checkpoint["mean"]
    std = checkpoint["std"]
    generated_points = generated_normalized.cpu() * std + mean
    keyframes = {
        timestep: (points.cpu() * std + mean).numpy()
        for timestep, points in normalized_keyframes.items()
    }

    generated_radii = torch.linalg.vector_norm(generated_points, dim=1)
    print(f"Mean generated radius: {generated_radii.mean().item():.4f}")
    print(f"Generated radius std: {generated_radii.std().item():.4f}")
    print(
        "Mean radial error: "
        f"{torch.abs(generated_radii - 1.0).mean().item():.4f}"
    )

    plot_denoising_process(
        checkpoint["original_points"].numpy(),
        keyframes,
        "plots/circle_denoising_process.png",
    )


if __name__ == "__main__":
    main()
