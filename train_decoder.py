# train_decoder.py
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from dataset_loader import LatentToProfileDataset
from attention_decoder import LatentToProfileAttentionDecoder as LatentToProfileDecoder

# Hyperparameters
batch_size = 32
lr = 1e-3
num_epochs = 150

# Load dataset
dataset = LatentToProfileDataset("decoder_input_latents_and_profiles.pkl")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LatentToProfileDecoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

###
def smoothness_loss(profile):
    return torch.mean((profile[:, 2:] - 2 * profile[:, 1:-1] + profile[:, :-2])**2)

def physical_loss(profile):
    non_neg = torch.mean(F.relu(-profile))  # Penalize negative values
    monotonic = torch.mean(F.relu(profile[:, :-1] - profile[:, 1:]))  # Penalize if decreasing
    return non_neg + monotonic

lambda_smooth = 0.1
lambda_phys = 0.1

for z, profile in loader:
    z, profile = z.to(device), profile.to(device)
    output = model(z).squeeze(-1)

    recon_loss = loss_fn(output, profile)
    smooth_loss = smoothness_loss(output)
    phys_loss = physical_loss(output)

    loss = recon_loss + lambda_smooth * smooth_loss + lambda_phys * phys_loss
###


# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for z, profile in loader:
        z, profile = z.to(device), profile.to(device)
        #output = model(z)
        output = model(z).squeeze(-1)  # Make it [batch_size, 31]

        loss = loss_fn(output, profile)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss / len(loader):.4f}")

# Save the model
torch.save(model.state_dict(), "vs_profile_decoder.pth")
