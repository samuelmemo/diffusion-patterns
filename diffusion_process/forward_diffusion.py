import torch
from matplotlib import pyplot as plt 
from shapes.generate_dataset import generate_dataset


#Detect device
def detect_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Detected device: {device}")
    return device



class Diffusion: 
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = detect_device() 

        self.betas = self.generate_noise_schedule().to(self.device)
        self.alphas = 1.0 - self.betas 
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)


    def generate_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def add_diffusion_noise(self, x_0, timesteps=None): 
        x_0 = x_0.to(self.device) #size: [batch_size, 2]

        if timesteps is None:
            timesteps = self.sample_timesteps(x_0)

        
        epsilon = torch.randn_like(x_0) #size [batch_size, 2]
        alpha_hat_t = self.alpha_hat[timesteps].unsqueeze(1) #size = [batch_size, 1]

        x_t = torch.sqrt(alpha_hat_t)*x_0 + torch.sqrt(1-alpha_hat_t)*epsilon  #size = [batch_size, 2]
        return x_t, epsilon, timesteps
    
    def sample_timesteps(self, x_0):
        #Returns a tensor with shape (batch_size, )
        return torch.randint(low=1, high=self.noise_steps, size=(x_0.shape[0],), device=self.device)
    


def main():

    #Plot the original figure along with 5 different noisy shapes corresponding to different timesteps
    data = generate_dataset()
    points = data["dataset"].tensors[0]

    diffusion = Diffusion()

    timesteps_to_plot = [0, 50, 200, 500, 750, 999]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax, t in zip(axes.flat, timesteps_to_plot):
        timesteps = torch.full(
            (points.shape[0],),
            t,
            dtype=torch.long,
            device=diffusion.device,
        )

        noisy_points, _, _ = diffusion.add_diffusion_noise(
            points,
            timesteps,
        )

        noisy_points = noisy_points.cpu().numpy()

        ax.scatter(noisy_points[:, 0], noisy_points[:, 1], s=5)
        ax.set_title(f"Timestep {t}")
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Forward Diffusion Process", fontsize=16)
    fig.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()


        


