import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Model, Qwen2Config

from .istft import ISTFTHead


class Encoder(nn.Module):
    def __init__(self, encoder_args, input_dim=320, hop_size=320, latent_dim=64):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=encoder_args)
        self.encoder = Qwen2Model(config)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, latent_dim*2)
        self.norm = nn.LayerNorm(config.hidden_size)

    def get_frames(self, x):
        num_frames_total = (x.size(-1) + self.hop_size - 1) // self.hop_size  # 向上取整的帧数
        expected_len = (num_frames_total - 1) * self.hop_size + self.input_dim
        padding_needed = expected_len - x.size(-1)
        waveform = F.pad(x, (0, padding_needed), value=0.0)

        frames = waveform.unfold(dimension=-1, size=self.input_dim, step=self.hop_size)   # [B, T, d]
        return frames

    def forward(self, waveform):
        x = self.get_frames(waveform)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.encoder(inputs_embeds=x)
        x = x.last_hidden_state

        x = self.fc3(x)
        return x, waveform.unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, decoder_args, output_dim=320, latent_dim=64, semantic_model=None, patch_size=-1):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=decoder_args)
        self.decoder = Qwen2Model(config)
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, config.hidden_size)

        if semantic_model is not None:
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(config.hidden_size, semantic_model.audio_emb_dim)
            self.semantic_model = semantic_model
            self.fc3 = nn.Linear(semantic_model.audio_emb_dim, config.hidden_size)
        else:
            self.semantic_model = None

        self.hop_length = output_dim
        self.head = ISTFTHead(dim=config.hidden_size, n_fft=self.hop_length * 4, hop_length=self.hop_length, padding="same")
        self.patch_size = patch_size

    def forward(self, x, only_semantic_emb=False, past_key_values=None, use_cache=False):
        x = self.fc1(x)

        if self.semantic_model is not None:
            x = self.fc2(self.gelu(x))
            x, past_key_values = self.semantic_model(whipser_feats=x, past_key_values=past_key_values, use_cache=use_cache)
            unified_emb = x
            if only_semantic_emb:
                return unified_emb, past_key_values
            x = self.fc3(x)
        else:
            unified_emb = None

        x = self.decoder(inputs_embeds=x)
        x = x.last_hidden_state

        x, _ = self.head(x)

        return x, unified_emb

    def low_level_reconstruct(self, x, past_key_values=None, use_cache=False, audio_buffer=None, window_buffer=None, last_chunk=False):
        x = self.fc1(x)
        outputs = self.decoder(inputs_embeds=x, past_key_values=past_key_values, use_cache=use_cache)
        past_key_values = outputs.past_key_values
        x = outputs.last_hidden_state

        x, _, audio_buffer, window_buffer = self.head(x, streaming=use_cache, audio_buffer=audio_buffer, window_buffer=window_buffer, last_chunk=last_chunk)

        return x, audio_buffer, window_buffer, past_key_values
