import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
from attention_decoder import LatentToProfileAttentionDecoder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = LatentToProfileAttentionDecoder().to(device)
model.load_state_dict(torch.load("vs_profile_decoder.pth", map_location=device))
model.eval()

# Load dataset
with open("decoder_input_latents_and_profiles.pkl", "rb") as f:
    df = pickle.load(f)

print(f"Loaded DataFrame with shape: {df.shape}")
print(df.head())

# Extract latent vectors and ground truth profiles
z_all = torch.tensor(df.iloc[:, :64].values, dtype=torch.float32).to(device)
gt_profiles = torch.tensor(df["vs_profile"].tolist(), dtype=torch.float32).to(device)

# Predict with model
with torch.no_grad():
    pred_profiles = model(z_all)  # shape: [batch, 31]
    pred_profiles = pred_profiles.squeeze(-1)

# Convert to CPU numpy arrays
pred_profiles_np = pred_profiles.cpu().numpy()
gt_profiles_np = gt_profiles.cpu().numpy()

# Load scaler and inverse transform
scaler = joblib.load("vs_profile_scaler.pkl")
pred_profiles_np = scaler.inverse_transform(pred_profiles_np)
gt_profiles_np = scaler.inverse_transform(gt_profiles_np)

# Calculate metrics
mse = F.mse_loss(torch.tensor(pred_profiles_np), torch.tensor(gt_profiles_np)).item()
mae = F.l1_loss(torch.tensor(pred_profiles_np), torch.tensor(gt_profiles_np)).item()
print(f"🔎 MSE: {mse:.2f} | MAE: {mae:.2f}")

# Plot selected profiles
# Extract station IDs (if not dropped during data prep)
station_ids = df["StationID"].tolist()

for i in [0, 5, 10,15,105]:
    true_vs = gt_profiles_np[i]
    pred_vs = pred_profiles_np[i]
    depth = list(range(len(true_vs)))

    plt.figure(figsize=(5, 4))
    plt.plot(true_vs, depth, label="Ground Truth", linewidth=2)
    plt.plot(pred_vs, depth, label="Predicted", linestyle='--')
    plt.gca().invert_yaxis()
    plt.xlabel("VS (m/s)")
    plt.ylabel("Depth (m)")
    plt.title(f"Station ID: {station_ids[i]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Save predictions
output_df = pd.DataFrame(pred_profiles_np, columns=[f"vs@{i}" for i in range(pred_profiles_np.shape[1])])
output_df.to_csv("generated_vs_profiles.csv", index=False)
print("Saved generated VS profiles to generated_vs_profiles.csv")

# Smoothness metrics
smoothness_gt = torch.mean(torch.abs(torch.tensor(gt_profiles_np)[:, 1:] - torch.tensor(gt_profiles_np)[:, :-1])).item()
smoothness_pred = torch.mean(torch.abs(torch.tensor(pred_profiles_np)[:, 1:] - torch.tensor(pred_profiles_np)[:, :-1])).item()
print(f"Smoothness — GT: {smoothness_gt:.2f}, Predicted: {smoothness_pred:.2f}")

# Debug output
print("debug")
print("Shape:", pred_profiles_np.shape)
print("Predicted Profile Example:", pred_profiles_np[0])
print("GT:", gt_profiles_np[0])
print("Pred:", pred_profiles_np[0])
