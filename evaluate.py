import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_loader import LatentToProfileDataset
from attention_decoder import LatentToProfileAttentionDecoder
import matplotlib.pyplot as plt
import joblib
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loader = DataLoader(LatentToProfileDataset("test_decoder_input.pkl"), batch_size=32, shuffle=False)

model = LatentToProfileAttentionDecoder().to(device)
model.load_state_dict(torch.load("vs_profile_decoder.pth", weights_only=True))
model.eval()

loss_fn = torch.nn.MSELoss()
total_test_loss = 0
total_mae_loss = 0

preds, gts = [], []

with torch.no_grad():
    for z, profile in test_loader:
        z, profile = z.to(device), profile.to(device)
        output = model(z).squeeze(-1)

        loss = loss_fn(output, profile)
        mae_loss = F.l1_loss(output, profile)
        
        total_test_loss += loss.item()
        total_mae_loss += mae_loss.item()

        preds.append(output.cpu().numpy())
        gts.append(profile.cpu().numpy())

avg_test_loss = total_test_loss / len(test_loader)
avg_mae_loss = total_mae_loss / len(test_loader)

print(f"Test MSE: {avg_test_loss:.2f} | MAE: {avg_mae_loss:.2f}")

preds = np.concatenate(preds, axis=0)
gts = np.concatenate(gts, axis=0)

scaler = joblib.load("vs_profile_scaler.pkl")
preds_inv = scaler.inverse_transform(preds)
gts_inv = scaler.inverse_transform(gts)

# Plot example predictions
for i in [idx for idx in [0,1,2,3,4,5,10,11,12,13,30,46] if idx < len(gts_inv)]:
    plt.figure(figsize=(6, 4))
    plt.plot(gts_inv[i], range(len(gts_inv[i])), label="Ground Truth", linewidth=2)
    plt.plot(preds_inv[i], range(len(preds_inv[i])), '--', label="Predicted", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Vs (m/s)")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
