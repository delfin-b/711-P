import pandas as pd
import numpy as np
import pickle

# Load latent vectors
df_latent = pd.read_csv("latent_vectors_with_ids.csv")

# Load profiles
with open("profiles.pkl", "rb") as f:
    profiles_dict = pickle.load(f)

# Filter to only stations that have profiles
df_filtered = df_latent[df_latent["StationID"].astype(str).isin(profiles_dict.keys())].copy()

# Add profile (vs_velocity) as new column
df_filtered["vs_profile"] = df_filtered["StationID"].astype(str).map(lambda sid: profiles_dict[sid]["vs_velocity"])

##
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit and transform profiles
vs_profiles = np.array(df_filtered["vs_profile"].tolist())
scaled_profiles = scaler.fit_transform(vs_profiles)

df_filtered["vs_profile"] = scaled_profiles.tolist()

# Save scaler
import joblib
joblib.dump(scaler, "vs_profile_scaler.pkl")
###

# Optional: Drop StationID
# df_filtered = df_filtered.drop(columns=["StationID"])

# Save to file
df_filtered.to_pickle("decoder_input_latents_and_profiles.pkl")
print("Saved decoder_input_latents_and_profiles.pkl")
