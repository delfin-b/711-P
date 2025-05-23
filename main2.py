# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import os
import random
import pickle

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
wandb_dir = os.path.join(os.getcwd(), "wandb_temp")
os.makedirs(wandb_dir, exist_ok=True)
os.environ["WANDB_DIR"] = wandb_dir
'''

# ------------------------------
# 1. Load main dataset from Excel
# ------------------------------
file_path = 'main.xlsx'
data = pd.read_excel(file_path)
data['StationID'] = data['StationID'].astype(str)



# ------------------------------
# 4. Filter merged data
# ------------------------------
columns_to_use = [
    'StationID', 
    'VS30', 
    'Latitude', 'Longitude', 'number of earthquakes',
    'Lithology', 'RockType', 'Elevation', 'Frequency',
    'Slope', 'Aspect', 'Period_Age', 'Epoch_Age']
numeric_cols = [
    'VS30', 
    'Latitude', 'Longitude', 'number of earthquakes',
                'Elevation', 'Frequency', 'Slope', 'Aspect']
categorical_cols = ['Lithology', 'RockType', 'Period_Age', 'Epoch_Age']

# Drop rows with missing values in numeric, categorical, and profile columns
data_filtered = data[columns_to_use].dropna(subset=numeric_cols + categorical_cols)

print("Filtered data shape:", data_filtered.shape)


# ------------------------------
# 5. Numeric feature transformation
# ------------------------------
def check_numeric_distributions(data, numeric_cols):
    skewed_features = []
    for col in numeric_cols:
        skewness = data[col].skew()
        print(f"Feature: {col}, Skewness: {skewness:.2f}")
        if abs(skewness) > 0.75:
            skewed_features.append(col)
    return skewed_features

def apply_log_transformation(data, skewed_features):
    data_transformed = data.copy()
    for col in skewed_features:
        data_transformed[col] = np.log1p(data_transformed[col])
        print(f"Applied log transformation to: {col}")
    return data_transformed

skewed_features = check_numeric_distributions(data_filtered, numeric_cols)
data_filtered_transformed = apply_log_transformation(data_filtered, skewed_features)



scaler = StandardScaler()
data_filtered_transformed[numeric_cols] = scaler.fit_transform(data_filtered_transformed[numeric_cols])
scaler_numeric = scaler
data_standardized = data_filtered_transformed.copy()




# ------------------------------
# 6. (Optional) Plot distributions
# ------------------------------
def plot_combined_distributions(data_original, data_log_transformed, data_standardized, numeric_cols):
    fig, axes = plt.subplots(len(numeric_cols), 3, figsize=(18, 5 * len(numeric_cols)))
    fig.suptitle("Distributions of Numeric Features: Original, Log-Transformed, Standardized", fontsize=16, y=1.02)
    for i, col in enumerate(numeric_cols):
        sns.histplot(data_original[col], kde=True, ax=axes[i, 0], color="blue", bins=30)
        axes[i, 0].set_title(f"Original: {col} (Skew = {data_original[col].skew():.2f})")
        axes[i, 0].set_xlabel("Value")
        axes[i, 0].set_ylabel("Frequency")
        
        sns.histplot(data_log_transformed[col], kde=True, ax=axes[i, 1], color="orange", bins=30)
        axes[i, 1].set_title(f"Log-Transformed: {col} (Skew = {data_log_transformed[col].skew():.2f})")
        axes[i, 1].set_xlabel("Value")
        axes[i, 1].set_ylabel("Frequency")
        
        sns.histplot(data_standardized[col], kde=True, ax=axes[i, 2], color="green", bins=30)
        axes[i, 2].set_title(f"Standardized: {col}")
        axes[i, 2].set_xlabel("Value")
        axes[i, 2].set_ylabel("Frequency")
    plt.tight_layout()
    #plt.show()

data_original = data_filtered.copy()
data_log_transformed = data_filtered.copy()
for col in skewed_features:
    data_log_transformed[col] = np.log1p(data_log_transformed[col])
plot_combined_distributions(data_original, data_log_transformed, data_standardized, numeric_cols)

# ------------------------------
# 7. Process categorical columns
# ------------------------------
def preprocess_categoricals_sorted(data, categorical_cols):
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    processed_data = data.copy()
    for col in categorical_cols:
        clean_col = col.strip("_")
        processed_data[f"{clean_col}_Sorted"] = processed_data[col].apply(
            lambda x: ", ".join(sorted(x.split(", "))) if isinstance(x, str) else ""
        )
        label_encoder = LabelEncoder()
        processed_data[f"{clean_col}_Encoded"] = label_encoder.fit_transform(
            processed_data[f"{clean_col}_Sorted"]
        )
        label_encoders[col] = label_encoder
    return processed_data, label_encoders

processed_data, encoders = preprocess_categoricals_sorted(data_standardized, categorical_cols)
encoded_categorical = processed_data[[f"{col.strip('_')}_Encoded" for col in categorical_cols]]
encoded_categorical = encoded_categorical.astype(int)

categorical_tensor = torch.tensor(encoded_categorical.values, dtype=torch.long).to(device)
numeric_tensor = torch.tensor(data_standardized[numeric_cols].values, dtype=torch.float32).to(device)

print(f"Shape of numeric_tensor: {numeric_tensor.shape}")
print(f"Shape of categorical_tensor: {categorical_tensor.shape}")

# ------------------------------

# ------------------------------
# 9. Train-validation-test split: Keep three separate inputs
# ------------------------------
num_samples = numeric_tensor.shape[0]
indices = np.arange(num_samples)
train_idx, temp_idx = train_test_split(indices, test_size=0.2, shuffle=True, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, shuffle=True, random_state=42)

numeric_train = numeric_tensor[train_idx]
numeric_val = numeric_tensor[val_idx]
numeric_test = numeric_tensor[test_idx]

categorical_train = categorical_tensor[train_idx]
categorical_val = categorical_tensor[val_idx]
categorical_test = categorical_tensor[test_idx]


# Create TensorDatasets returning 6 items:
# (numeric, categorical, profile) as inputs and (numeric, categorical, profile) as targets
train_dataset = TensorDataset(numeric_train, categorical_train, 
                              numeric_train, categorical_train)
val_dataset = TensorDataset(numeric_val, categorical_val, 
                            numeric_val, categorical_val)
test_dataset = TensorDataset(numeric_test, categorical_test,
                             numeric_test, categorical_test)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

batch_size = 64
# check
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of batches in train_loader: {len(train_loader)}")
print(f"Number of batches in val_loader: {len(val_loader)}")
print(f"Number of batches in test_loader: {len(test_loader)}")

# Save tensors and station IDs
torch.save(numeric_tensor.cpu(), "numeric_tensor.pt")
torch.save(categorical_tensor.cpu(), "categorical_tensor.pt")
np.save("station_ids.npy", data_filtered["StationID"].values)

# Optional: save scalers and encoders if needed later
with open("scaler_numeric.pkl", "wb") as f:
    pickle.dump(scaler_numeric, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Saved numeric_tensor.pt, categorical_tensor.pt, station_ids.npy")





class BetaVAE(nn.Module):
    def __init__(self, num_numeric, categorical_group_sizes, embedding_dims, latent_dim, dropout_prob):
        super(BetaVAE, self).__init__()
        self.num_numeric = num_numeric
        self.categorical_group_sizes = categorical_group_sizes
        self.embedding_dims = embedding_dims
        self.latent_dim = latent_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=group_size, embedding_dim=embed_dim)
            for group_size, embed_dim in zip(categorical_group_sizes, embedding_dims)
        ])

        input_dim = num_numeric + sum(embedding_dims)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_numeric + sum(embedding_dims))
        )

        self.decoding_layers = nn.ModuleList([
            nn.Linear(embed_dim, group_size)
            for group_size, embed_dim in zip(categorical_group_sizes, embedding_dims)
        ])

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, apply_softmax=False):
        h = self.decoder(z)
        output_numeric = h[:, :self.num_numeric]
        output_cat_embeds = h[:, self.num_numeric:]

        decoded_categoricals = []
        start = 0
        for i, layer in enumerate(self.decoding_layers):
            dim = self.embedding_dims[i]
            end = start + dim
            logits = layer(output_cat_embeds[:, start:end])
            decoded_categoricals.append(logits)
            start = end

        return output_numeric, decoded_categoricals

    def forward(self, numeric_tensor, categorical_tensor, apply_softmax=False):
        embedded = [embed(categorical_tensor[:, i]) for i, embed in enumerate(self.embeddings)]
        embedded_tensor = torch.cat(embedded, dim=1)
        x = torch.cat([numeric_tensor, embedded_tensor], dim=1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output_numeric, decoded_categoricals = self.decode(z, apply_softmax)
        return output_numeric, decoded_categoricals, mu, logvar



# Correlation Penalty
def compute_correlation_penalty(mu):
    """
    Computes the correlation penalty (epsilon term) for the latent variables.
    Args:
        mu: Mean of the latent space distribution (batch_size x latent_dim).
    Returns:
        correlation_penalty: Scalar penalty term for correlation in latent space.
    """
    batch_size, latent_dim = mu.size()
    covariance = (mu.T @ mu) / batch_size  # Covariance matrix
    diag = torch.diag(covariance)  # Diagonal elements
    off_diag = covariance - torch.diag_embed(diag)  # Off-diagonal elements
    correlation_penalty = torch.sum(off_diag ** 2)  # Frobenius norm of off-diagonal elements
    return correlation_penalty

def loss_function(
    recon_numeric, 
    recon_categoricals_logits, 
    target_numeric, 
    target_categoricals, 
    mu, 
    logvar, 
    lambda_weights, 
    beta, 
    epsilon_weight,
):
    """
    Custom loss function for Beta-VAE with:
    - Numeric MSE loss
    - CE for categorical embeddings (with per-category lambda weights)
    - KL divergence
    - Correlation penalty
    Args:
        recon_numeric: Reconstructed numeric tensor.
        recon_categoricals_logits: List of reconstructed logits for each categorical group.
        target_numeric: Ground truth numeric tensor.
        target_categoricals: List of ground truth categorical tensors for each group.
        mu: Mean of the latent space distribution.
        logvar: Log variance of the latent space distribution.
        lambda_weights: List of lambda weights for each categorical group.
        beta: Weight for KL divergence term.
        epsilon_weight: Weight for correlation penalty term.
    Returns:
        total_loss: Total weighted loss.
        individual_losses: Dictionary with individual loss components.
    """
    # 1. Numeric Reconstruction Loss (MSE)
    L_rec_numeric = nn.MSELoss(reduction='mean')(recon_numeric, target_numeric)

    # 2. Categorical Reconstruction Loss (CE with individual lambda weights)
    L_rec_categorical = 0.0
    for logits, target, lambda_weight in zip(recon_categoricals_logits, target_categoricals, lambda_weights):
        L_rec_categorical += lambda_weight * nn.CrossEntropyLoss(reduction='mean')(logits, target.long())
        
    # 3. Profile Reconstruction Loss (MSE)
    #L_rec_profile = nn.MSELoss(reduction='mean')(recon_profile, target_profile)

    # 4. KL Divergence Loss
    L_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    #5. Correlation Penalty
    correlation_penalty = compute_correlation_penalty(mu)
    
    # 6. Total Reconstruction and Regularization Losses
    L_rec = L_rec_numeric + L_rec_categorical #+ lambda_profile * L_rec_profile
    L_reg = beta * L_kl + epsilon_weight * correlation_penalty

    # 7. Total Loss
    total_loss = L_rec + L_reg



    # Return total loss and individual components for monitoring
    individual_losses = {
        "total_loss": total_loss.item(),
        "reconstruction_loss": L_rec.item(),
        "regularization_loss": L_reg.item(),
        "L_rec_numeric": L_rec_numeric.item(),
        "L_rec_categorical": L_rec_categorical.item(),
        #"L_rec_profile": L_rec_profile.item(),
        "L_kl": L_kl.item(),
        "correlation_penalty": correlation_penalty.item(),
    }

    return total_loss, individual_losses


#### check

# ==== Set Your Best Hyperparameters ====
latent_dim = 64
dropout_prob = 0.2
learning_rate = 0.001
beta = 0.1
epsilon_weight = 0.1
lambda_weights = [1.0, 1.0, 1.0, 0.5]
num_epochs = 150

#embedding_dims = [4, 4, 4, 4]  # Example embedding dimensions for categorical groups


def train_vae(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    num_numeric, 
    categorical_group_sizes, 
    embedding_dims,
    #lambda_weights, 
    num_epochs=150, 
    device='cuda'
):
    # Reinitialize the model using hyperparameters from wandb.config
    # Note: embedding_dims remains unchanged as computed globally
    model = BetaVAE(
        num_numeric=num_numeric,
        categorical_group_sizes=categorical_group_sizes,
        embedding_dims=embedding_dims,
        latent_dim=latent_dim,  # from sweep
        #profile_dim=31,
        #profile_reduced_dim=15,
        dropout_prob=dropout_prob  # from sweep (or default 0.2)
    ).to(device)

    # Reinitialize the optimizer with the current learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    history = {
        'total_loss': [], 
        'reconstruction_loss': [], 
        'categorical_loss': [], 
        #'profile_loss': [],    # track profile reconstruction loss
        'kl_loss': [], 
        'correlation_loss': [],  # Training correlation penalty
        'val_total_loss': [], 
        'val_rec_loss': [], 
        'val_cat_loss': [], 
        #'val_profile_loss': [],  #  vaidation profile loss
        'val_kl_loss': [],
        'val_corr_loss': []
    }
    
    
    best_val_loss = float('inf')


    for epoch in range(150):
        # training phase
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_cat_loss = 0.0
        #total_profile_loss = 0.0
        total_kl_loss = 0.0
        total_corr_loss = 0.0 

        for batch in train_loader:
            numeric_tensor = batch[0].to(device)
            categorical_tensor = batch[1].to(device)
            target_numeric = batch[2].to(device)  # ✅ correct index
            target_categoricals = [batch[3][:, i].to(device) for i in range(len(categorical_group_sizes))]

            #target_profile = batch[5].to(device)

            optimizer.zero_grad()

            # Forward pass: 3 inputse
            output_numeric, decoded_categoricals, mu, logvar = model(
                numeric_tensor, 
                categorical_tensor, 
                #profile_tensor, 
                apply_softmax=False
            )

            # Compute the loss (including the profile reconstruction MSE)
            loss, losses_dict = loss_function(
                recon_numeric=output_numeric,
                recon_categoricals_logits=decoded_categoricals,
                #recon_profile=recon_profile,
                target_numeric=target_numeric,
                target_categoricals=target_categoricals,
                #target_profile=target_profile,
                mu=mu,
                logvar=logvar,
                lambda_weights=lambda_weights,  # 
                beta=beta,                      # 
                epsilon_weight=epsilon_weight,  #
                #lambda_profile=wandb.config.lambda_profile   #
            )

            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_rec_loss += losses_dict["L_rec_numeric"]
            total_cat_loss += losses_dict["L_rec_categorical"]
            #total_profile_loss += losses_dict["L_rec_profile"]
            total_kl_loss += losses_dict["L_kl"]
            total_corr_loss += losses_dict["correlation_penalty"]

        # Store average training losses for the epoch
        history['total_loss'].append(total_loss / len(train_loader))
        history['reconstruction_loss'].append(total_rec_loss / len(train_loader))
        history['categorical_loss'].append(total_cat_loss / len(train_loader))
        #history['profile_loss'].append(total_profile_loss / len(train_loader))
        history['kl_loss'].append(total_kl_loss / len(train_loader))
        history['correlation_loss'].append(total_corr_loss / len(train_loader))

        # Validation loop 
        model.eval()
        val_loss = 0.0
        val_rec_loss = 0.0
        val_cat_loss = 0.0
        #val_profile_loss = 0.0
        val_kl_loss = 0.0
        val_corr_loss = 0.0
        val_cat_acc_total = 0.0
        # for error tracking
        val_reconstruction_loss = 0.0
        val_regularization_loss = 0.0

        
        
        numeric_importance_accum = {col: [] for col in numeric_cols}  # For feature importance
        categorical_importance_accum = {col: [] for col in categorical_cols}  # For feature importance
        #profile_importance_accum = []  # For feature importance

        with torch.no_grad():
            for batch in val_loader:
                numeric_tensor = batch[0].to(device)
                categorical_tensor = batch[1].to(device)
                target_numeric = batch[2].to(device)  # ✅ correct index
                target_categoricals = [batch[3][:, i].to(device) for i in range(len(categorical_group_sizes))]


                #target_profile = batch[5].to(device)
                

                output_numeric, decoded_categoricals, mu, logvar = model(
                    numeric_tensor, 
                    categorical_tensor, 
                    #profile_tensor, 
                    apply_softmax=False
                )

                loss, losses_dict = loss_function(
                    recon_numeric=output_numeric,
                    recon_categoricals_logits=decoded_categoricals,
                    #recon_profile=recon_profile,
                    target_numeric=target_numeric,
                    target_categoricals=target_categoricals,
                    #target_profile=target_profile,
                    mu=mu,
                    logvar=logvar,
                    lambda_weights=lambda_weights,  # 
                    beta=beta,                      # 
                    epsilon_weight=epsilon_weight,  #
                    #lambda_profile=wandb.config.lambda_profile   #
                )
                
                

                val_loss += loss.item()
                val_rec_loss += losses_dict["L_rec_numeric"]
                val_cat_loss += losses_dict["L_rec_categorical"]
                #val_profile_loss += losses_dict["L_rec_profile"]
                val_kl_loss += losses_dict["L_kl"]
                val_corr_loss += losses_dict["correlation_penalty"]
                val_reconstruction_loss += losses_dict["reconstruction_loss"]
                val_regularization_loss += losses_dict["regularization_loss"]

                
                
                # 2. Also compute for feature importance
                num_err = torch.abs(target_numeric - output_numeric).cpu().numpy()
                for i, col in enumerate(numeric_cols):
                    numeric_importance_accum[col].extend(num_err[:, i])
                
                for i, col in enumerate(categorical_cols):
                    cat_err = (target_categoricals[i] != torch.argmax(decoded_categoricals[i], dim=1)).float().cpu().numpy()
                    categorical_importance_accum[col].extend(cat_err)
                
                #profile_err = torch.mean(torch.abs(target_profile - recon_profile), dim=1).cpu().numpy()
                #profile_importance_accum.extend(profile_err)
            
            

        mean_val_cat_acc = val_cat_acc_total / (len(val_loader) * len(categorical_cols))
        
        history['val_total_loss'].append(val_loss / len(val_loader))
        history['val_rec_loss'].append(val_rec_loss / len(val_loader))
        history['val_cat_loss'].append(val_cat_loss / len(val_loader))
        #history['val_profile_loss'].append(val_profile_loss / len(val_loader))
        history['val_kl_loss'].append(val_kl_loss / len(val_loader))
        history['val_corr_loss'].append(val_corr_loss / len(val_loader))
        scheduler.step(val_loss)


        print(
            f"Epoch {epoch + 1}/{num_epochs}\n"
            f"  Train Total Loss: {total_loss / len(train_loader):.4f}, "
            f"  Val Total Loss: {val_loss / len(val_loader):.4f}\n"
            f"  Train KL: {total_kl_loss / len(train_loader):.4f}, "
            f"  Val KL: {val_kl_loss / len(val_loader):.4f}\n"
            f"  Train Cat Loss: {total_cat_loss / len(train_loader):.4f}, "
            f"  Val Cat Loss: {val_cat_loss / len(val_loader):.4f}\n"
            #f"  Train Profile Loss: {total_profile_loss / len(train_loader):.4f}, "
            #f"  Val Profile Loss: {val_profile_loss / len(val_loader):.4f}\n"
            f"  Train Corr Penalty: {total_corr_loss / len(train_loader):.4f}, "
            f"  Val Corr Penalty: {val_corr_loss / len(val_loader):.4f}\n"
            f"  Val Cat Accuracy (overall): {mean_val_cat_acc:.4f}"
        ) 


        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated.")


    return model, history




# === 1. Define embedding dimensions dynamically (recommended)
embedding_dims = [min(50, (len(encoders[col].classes_) + 1) // 2) for col in categorical_cols]

# === 2. Prepare and call train_vae
model, history = train_vae(
    model=None,  # model will be created inside the function
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=None,  # will be initialized inside
    num_numeric=len(numeric_cols),
    categorical_group_sizes=[len(encoders[col].classes_) for col in categorical_cols],
    embedding_dims=embedding_dims,
    num_epochs=num_epochs,
    device=device
)

# === 3. Save model history (optional)
import pickle
with open("vae_training_history.pkl", "wb") as f:
    pickle.dump(history, f)

print("Training completed and best model saved as best_model.pth.")


#//////////////////////////////////////////////////////////// PLOTS               updated, some losses are combined                           


# ========== LATENT VECTOR EXTRACTION ==========

print("🔍 Extracting latent vectors from best_model.pth...")

# Reload encoders (in case training was skipped and we run only this block)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Reconstruct embedding dims
categorical_cols = ['Lithology', 'RockType', 'Period_Age', 'Epoch_Age']
embedding_dims = [min(50, (len(encoders[col].classes_) + 1) // 2) for col in categorical_cols]

# Reload saved tensors
numeric_tensor = torch.load("numeric_tensor.pt").to(device)
categorical_tensor = torch.load("categorical_tensor.pt").to(device)
station_ids = np.load("station_ids.npy", allow_pickle=True)

# Create a DataLoader (no shuffle!)
from torch.utils.data import TensorDataset, DataLoader
inference_loader = DataLoader(TensorDataset(numeric_tensor, categorical_tensor), batch_size=64, shuffle=False)

# Reload model
model = BetaVAE(
    num_numeric=len(numeric_cols),
    categorical_group_sizes=[len(encoders[col].classes_) for col in categorical_cols],
    embedding_dims=embedding_dims,
    latent_dim=latent_dim,
    dropout_prob=dropout_prob
).to(device)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Extract mu vectors
all_mu = []
with torch.no_grad():
    for batch in inference_loader:
        numeric, categorical = batch
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        _, _, mu, _ = model(numeric, categorical, apply_softmax=False)
        all_mu.append(mu.cpu().numpy())

mu_matrix = np.vstack(all_mu)

# Save to CSV
import pandas as pd
df_mu = pd.DataFrame(mu_matrix, columns=[f"z{i}" for i in range(mu_matrix.shape[1])])
df_mu["StationID"] = station_ids
df_mu.to_csv("latent_vectors_with_ids.csv", index=False)

print("✅ Latent vectors saved to latent_vectors_with_ids.csv")
