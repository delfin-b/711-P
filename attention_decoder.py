import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionOverLatent(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(latent_dim + hidden_dim, latent_dim)
        self.context_fc = nn.Linear(latent_dim, hidden_dim)

    def forward(self, hidden_state, latent):
        # hidden_state: [batch, hidden_dim]
        # latent: [batch, latent_dim]
        combined = torch.cat([hidden_state, latent], dim=1)  # [batch, hidden + latent]
        attn_weights = torch.softmax(self.attn_fc(combined), dim=1)  # [batch, latent_dim]

        # Element-wise attention over latent (NOT reducing with sum)
        context = attn_weights * latent  # [batch, latent_dim]

        return self.context_fc(context)  # [batch, hidden_dim]



class LatentToProfileAttentionDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=128, profile_len=31, depth_embed_dim=16):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.profile_len = profile_len

        # Depth embeddings (learnable)
        self.depth_embeddings = nn.Embedding(profile_len, depth_embed_dim)

        # Attention over latent vector
        self.attn = AttentionOverLatent(latent_dim, hidden_dim)

        # Input projection to LSTM
        self.input_fc = nn.Linear(depth_embed_dim + hidden_dim, hidden_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Output layer to predict VS value
        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        """
        Args:
            z: [batch, latent_dim]
        Returns:
            vs_profile: [batch, profile_len]
        """
        batch_size = z.size(0)

        # Prepare depth embedding sequence: [0, 1, ..., 29]
        depth_steps = torch.arange(self.profile_len, device=z.device).unsqueeze(0).repeat(batch_size, 1)  # [batch, profile_len]
        depth_embeds = self.depth_embeddings(depth_steps)  # [batch, profile_len, depth_embed_dim]

        # Initialize hidden and cell state
        h_t = torch.zeros(1, batch_size, self.hidden_dim, device=z.device)
        c_t = torch.zeros(1, batch_size, self.hidden_dim, device=z.device)

        outputs = []
        for t in range(self.profile_len):
            depth_embed = depth_embeds[:, t, :]  # [batch, depth_embed_dim]

            # Attention over z using current hidden state
            context = self.attn(h_t.squeeze(0), z)  # [batch, hidden_dim]

            # Combine context + depth embedding
            lstm_input = torch.relu(self.input_fc(torch.cat([depth_embed, context], dim=1)))  # [batch, hidden_dim]
            lstm_input = lstm_input.unsqueeze(1)  # [batch, 1, hidden_dim]

            # Run LSTM one step
            output, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))  # output: [batch, 1, hidden_dim]

            # Predict vs@depth_t
            vs_t = self.output_fc(output).squeeze(1)  # [batch, 1] → [batch]
            outputs.append(vs_t)

        vs_profile = torch.stack(outputs, dim=1)  # [batch, profile_len]
        return vs_profile
