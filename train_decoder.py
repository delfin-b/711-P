import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from dataset_loader import LatentToProfileDataset
from attention_decoder import LatentToProfileAttentionDecoder

batch_size = 32
lr = 1e-3
num_epochs = 500

train_loader = DataLoader(LatentToProfileDataset("train_decoder_input.pkl"), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(LatentToProfileDataset("val_decoder_input.pkl"), batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LatentToProfileAttentionDecoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#    optimizer, mode='min', factor=0.5, patience=10, verbose=True
#)
loss_fn = nn.MSELoss()



def smoothness_loss(profile):
    return torch.mean((profile[:, 2:] - 2*profile[:, 1:-1] + profile[:, :-2])**2)

def physical_loss(profile):
    non_neg = torch.mean(F.relu(-profile))
    monotonic = torch.mean(F.relu(profile[:, :-1] - profile[:, 1:]))
    return non_neg + monotonic

lambda_smooth = 0.5
lambda_phys = 0.2

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for z, profile in train_loader:
        z, profile = z.to(device), profile.to(device)
        output = model(z).squeeze(-1)

        recon_loss = loss_fn(output, profile)
        smooth_loss = smoothness_loss(output)
        phys_loss = physical_loss(output)

        loss = recon_loss + lambda_smooth*smooth_loss + lambda_phys*phys_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for z, profile in val_loader:
            z, profile = z.to(device), profile.to(device)
            output = model(z).squeeze(-1)
            val_loss = loss_fn(output, profile)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "vs_profile_decoder.pth")
