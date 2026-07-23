import torch 
from torch.utils.data import TensorDataset, DataLoader
from shapes.generate_2d_shapes import generate_circle, plot_figure, generate_spiral

def generate_dataset():
    points = generate_circle(n_points=1000,
                              radius=1.0,
                              noise_std=0.02,
                              seed=42)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    mean = points_tensor.mean(dim=0)
    std = points_tensor.std(dim=0)
    normalized_points = (points_tensor - mean) / std
    points_dataset = TensorDataset(normalized_points)
    points_dataloader = DataLoader(points_dataset, batch_size=128, shuffle=True)
    return {
        "original_points": points,
        "dataset": points_dataset,
        "dataloader": points_dataloader,
        "mean": mean,
        "std": std,
    }
    
def main():
    data_dict = generate_dataset()
    plot_figure(data_dict["original_points"])



if __name__ == '__main__':
    main()