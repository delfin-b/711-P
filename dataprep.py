import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load latent vectors
df_latent = pd.read_csv("latent_vectors_with_ids.csv")

# Load profiles
with open("profiles.pkl", "rb") as f:
    profiles_dict = pickle.load(f)

# Filter to stations with profiles
df_filtered = df_latent[df_latent["StationID"].astype(str).isin(profiles_dict.keys())].copy()

# Add vs profiles
df_filtered["vs_profile"] = df_filtered["StationID"].astype(str).map(lambda sid: profiles_dict[sid]["vs_velocity"])

# Standardize profiles
scaler = StandardScaler()
vs_profiles = np.array(df_filtered["vs_profile"].tolist())
scaled_profiles = scaler.fit_transform(vs_profiles)
df_filtered["vs_profile"] = scaled_profiles.tolist()

# Save scaler
joblib.dump(scaler, "vs_profile_scaler.pkl")

# Split data into train, val, test
train_df, temp_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save splits separately
train_df.to_pickle("train_decoder_input.pkl")
val_df.to_pickle("val_decoder_input.pkl")
test_df.to_pickle("test_decoder_input.pkl")

print("Data split completed:")
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
