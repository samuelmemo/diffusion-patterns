from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from diffusion_process.forward_diffusion import Diffusion
from shapes.generate_dataset import generate_dataset


class NoisePredictor(nn.Module):
    """MLP that predicts the noise added to a 2D point."""

    def __init__(self, noise_steps=1000):
        super().__init__()
        self.noise_steps = noise_steps

        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x_t, timesteps):
        timesteps = timesteps.unsqueeze(1).float()
        timesteps = timesteps / (self.noise_steps - 1)
        model_input = torch.cat((x_t, timesteps), dim=1)
        return self.model(model_input)


def training_loop(model, diffusion, learning_rate, train_dataloader, n_epochs):
    device = diffusion.device
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
        )

        for (x_0,) in progress_bar:
            x_0 = x_0.to(device)
            x_t, epsilon, timesteps = diffusion.add_diffusion_noise(x_0)

            optimizer.zero_grad()
            predicted_noise = model(x_t, timesteps)
            loss = loss_function(predicted_noise, epsilon)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        average_loss = epoch_loss / len(train_dataloader)
        tqdm.write(f"Epoch {epoch + 1}: average loss={average_loss:.6f}")

    return model


def save_model(model, diffusion, data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "noise_steps": diffusion.noise_steps,
            "beta_start": diffusion.beta_start,
            "beta_end": diffusion.beta_end,
            "mean": data["mean"],
            "std": data["std"],
            "original_points": torch.tensor(
                data["original_points"],
                dtype=torch.float32,
            ),
        },
        output_path,
    )
    print(f"Saved trained model to {output_path}")


def main():
    data = generate_dataset(shape="two_moons")
    diffusion = Diffusion()
    model = NoisePredictor(diffusion.noise_steps)

    model = training_loop(
        model,
        diffusion,
        learning_rate=1e-3,
        train_dataloader=data["dataloader"],
        n_epochs=2000,
    )

    save_model(
        model,
        diffusion,
        data,
        "models/circle_model.pt",
    )


if __name__ == "__main__":
    main()
