from attention_decoder import LatentToProfileAttentionDecoder
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LatentToProfileAttentionDecoder().to(device)
torch.save(model.state_dict(), "vs_profile_decoder.pth")
print("Saved decoder model as vs_profile_decoder.pth")