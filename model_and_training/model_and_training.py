import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from shapes.generate_dataset import generate_dataset
from diffusion_process.forward_diffusion import Diffusion
from tqdm import tqdm

def detect_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Detected device: {device}")
    return device

class NoisePredictor(nn.Module):
    """
    A simple Neural Network model.
    """
    def __init__(self, noise_steps=1000):
        super(NoisePredictor, self).__init__()
        self.noise_steps = noise_steps

        self.model = nn.Sequential(
            nn.Linear(3, 128),
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
    optimizer = optim.AdamW(model.parameters(), lr= learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range (n_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
        )

        for (x_0,) in progress_bar:

            # Move input to device 
            x_0 = x_0.to(device)

            #Generate noisy image
            x_t, epsilon, timesteps = diffusion.add_diffusion_noise(x_0)

            # Clear the gradients from the previous iteration
            optimizer.zero_grad()

            # Perform a forward pass to get noise prediction
            predicted_noise = model(x_t, timesteps)

            # Calculate the loss
            loss = loss_function(predicted_noise, epsilon)

            # Perform a backward pass to compute gradients
            loss.backward()

            # Update the model's weights
            optimizer.step()

            #Accumulate epoch loss:
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        average_loss = epoch_loss / len(train_dataloader)
        tqdm.write(f"Epoch {epoch + 1}: average loss={average_loss:.6f}")

    return model


@torch.no_grad()
def sample_points(model, diffusion, n_points=1000):
    model.eval()
    device = diffusion.device
    x = torch.randn((n_points, 2), device=device)

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
            x = model_mean + torch.sqrt(posterior_variance) * torch.randn_like(x)
        else:
            x = model_mean

    return x


def plot_results(original_points, generated_points):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(original_points[:, 0], original_points[:, 1], s=5)
    axes[0].set_title("Original circle")

    axes[1].scatter(generated_points[:, 0], generated_points[:, 1], s=5)
    axes[1].set_title("Generated circle")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()



def main():
    data = generate_dataset()
    diffusion = Diffusion()
    model = NoisePredictor(diffusion.noise_steps)

    model = training_loop(
        model,
        diffusion,
        learning_rate=1e-3,
        train_dataloader=data["dataloader"],
        n_epochs=200,
    )

    generated_normalized = sample_points(
        model,
        diffusion,
        n_points=1000,
    )

    generated_points = (
        generated_normalized.cpu() * data["std"] + data["mean"]
    )

    generated_radii = torch.linalg.vector_norm(generated_points, dim=1)
    print(f"Mean generated radius: {generated_radii.mean().item():.4f}")
    print(f"Generated radius std: {generated_radii.std().item():.4f}")
    print(
        "Mean radial error: "
        f"{torch.abs(generated_radii - 1.0).mean().item():.4f}"
    )

    plot_results(
        data["original_points"],
        generated_points.numpy(),
    )


if __name__ == "__main__":
    main()
    

    
