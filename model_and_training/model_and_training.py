import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from shapes.generate_dataset import generate_dataset
from diffusion_process.forward_diffusion import Diffusion

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
            nn.Linear(3, 32), #x,y, timestep
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64, 2) #Predicted epsilon [x,y]
        )

    def forward(self, x_t, timesteps):
       timesteps = timesteps.unsqueeze(1).float() # shape: (32, ) --> (32,1)
       timesteps = timesteps/(self.noise_steps-1)
       input = torch.cat((x_t,timesteps), dim=1)
       return self.model(input)
    

def training_loop(model, device, learning_rate, train_dataloader, n_epochs): 

    model = model.to(Diffusion.device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)
    loss_function = nn.MSELoss()

    for epoch in range (n_epochs):
        epoch_loss = 0.0

        for x_0 in (train_dataloader):

            # Move input to device 
            x_0 = x_0.to(device)

            #Generate noisy image
            x_t, epsilon, timesteps = Diffusion.add_diffusion_noise(x_0)

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

    average_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}: loss={average_loss:.6f}")




    

    
