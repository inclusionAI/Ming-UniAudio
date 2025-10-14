#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging

from configuration_bailingmm import BailingMMConfig
from fm.flowloss import FlowLoss
from modeling_bailing_moe import BailingMoeForCausalLM

from modeling_utils import (
    build_modality_mask,
    encode_audio_segments,
    patch_continuous_features,
    LinearPooling
)


from audio_tokenizer.modeling_audio_vae import AudioVAE

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMMConfig"


@dataclass
class BailingMMCausalLMOutputWithPast(ModelOutput):
    """
    Base class for BailingMM causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class BailingMMNativeForConditionalGeneration(PreTrainedModel):
    config_class = BailingMMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingAudioModel"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: BailingMMConfig,
        empty_load=False, 
    ):
        super().__init__(config)
        self.config: BailingMMConfig = config
        self.audio = None
        self.model = None
        if empty_load:
            return
        
        self.llm_dytpe = torch.bfloat16
        self.model = BailingMoeForCausalLM(self.config.llm_config)
        self.audio = AudioVAE(self.config.audio_tokenizer_config)
        self.audio_downsampler_args = {}
        self.linear_proj_audio = LinearPooling(
            hidden_size=self.audio.semantic_emb_dim,
            llm_input_dim=self.model.config.hidden_size,
        )

        self.flowloss = FlowLoss(
            z_channels=self.config.audio_tokenizer_config.enc_kwargs['latent_dim'],
            llm_cond_dim=self.model.config.hidden_size,
            **self.config.ditar_config
        )
        self.stop_head = nn.Linear(self.model.config.hidden_size, 2, bias=True)

        self.post_init()
        
    def get_rope_index(
        self,
        input_ids,
        image_token_id,
        video_token_id,
        image_start_token_id,
        video_start_token_id,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
        spatial_merge_size=2,
        tokens_per_second=2,
        second_per_grid_ts=None,
        inputs_embeds=None
    ):
        use_abs_time_pos = second_per_grid_ts is not None

        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                if image_grid_thw is not None:
                    vision_start_indices = torch.argwhere(
                        input_ids == image_start_token_id
                    ).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                if video_grid_thw is not None:
                    vision_start_indices = torch.argwhere(
                        input_ids == video_start_token_id
                    ).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    if use_abs_time_pos:
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        time_tensor_long = time_tensor.long()
                    else:
                        time_tensor_long = expanded_range.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
        else:
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            length = inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1)
            bsz = inputs_embeds.size(0) if inputs_embeds is not None else input_ids.size(0)
            dtype = inputs_embeds.dtype if inputs_embeds is not None else input_ids.dtype

            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(length, device=device)
                    .view(1, 1, -1)
                    .expand(3, bsz, -1)
                )
                mrope_position_deltas = torch.zeros(
                    [bsz, 1],
                    device=device,
                    dtype=dtype,
                )

        return position_ids, mrope_position_deltas

    def extract_audio_feature(self, waveform, waveform_length):
        audio_embeds, audio_embeds_lengths = encode_audio_segments(
            encoder=self.audio,
            proj_layer=self.linear_proj_audio,
            waveforms=waveform,
            waveforms_lengths=waveform_length,
        )

        return audio_embeds.to(waveform.dtype), audio_embeds_lengths

    def prompt_wrap_audio(
        self,
        input_ids,
        inputs_embeds,
        audio_embeds,
        audio_embeds_lengths,
        placeholder_audio_loc_lens,
    ):
        inputs_embeds = patch_continuous_features(
            input_embeddings=inputs_embeds,
            placeholder_loc_lens=placeholder_audio_loc_lens,
            encoded_feats=audio_embeds,
            encoded_feat_lens=audio_embeds_lengths,
        )
        audio_router_mask = build_modality_mask(
            placeholder_audio_loc_lens, inputs_embeds.shape[:-1]
        ).to(inputs_embeds.device)
        return inputs_embeds, audio_router_mask

    def prompt_wrap_navit(
        self,
        input_ids,
        query_embeds_audio=None,
        query_embeds_audio_lengths=None,
        placeholder_audio_loc_lens=None,
        target_embeds=None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if (
            query_embeds_audio is None
            and target_embeds is None
        ):
            return inputs_embeds

        audio_mask = None

        if query_embeds_audio is not None:
            inputs_embeds, audio_mask = self.prompt_wrap_audio(
                input_ids,
                inputs_embeds,
                query_embeds_audio,
                query_embeds_audio_lengths,
                placeholder_audio_loc_lens,
            )
        return inputs_embeds, audio_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_whisper_encoder: bool = False,
    ) -> Union[Tuple, BailingMMCausalLMOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (
            pixel_values is not None or pixel_values_videos is not None or audio_feats is not None
        ) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos/pixel_values_audios and inputs_embeds at the same time, and must specify either one"
            )

        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)
        if audio_feats is not None:
            audio_embeds, audio_embeds_lengths = self.extract_audio_feature(
                audio_feats, audio_feats_lengths, use_whisper_encoder=use_whisper_encoder
            )

        if (
            image_embeds is None and video_embeds is None and audio_embeds is None
        ) or input_ids.size(1) == 1:
            words_embeddings = self.model.get_input_embeddings()(
                input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1)
            )
            audio_mask = None

        else:
            words_embeddings, audio_mask = self.prompt_wrap_navit(
                input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1),
                audio_embeds,
                audio_embeds_lengths,
                audio_placeholder_loc_lens,
                None,  # noqa
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=words_embeddings,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_mask=None,
            audio_mask=audio_mask,
        )

        return BailingMMCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        waveform: Optional[torch.FloatTensor] = None,
        waveform_length: Optional[torch.FloatTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        **generate_kwargs,
    ):
        audio_embeds, audio_embeds_lengths = None, None

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            audio_embeds, audio_embeds_lengths = self.extract_audio_feature(
                waveform, waveform_length
            )
            words_embeddings, audio_mask = self.prompt_wrap_navit(
                input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1),
                audio_embeds,
                audio_embeds_lengths,
                audio_placeholder_loc_lens,
                None,  # noqa
            )

            if (
                self.config.llm_config.rope_scaling is not None
                and self.config.llm_config.rope_scaling["type"] == "3D"
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_token_id=self.config.llm_config.image_patch_token,
                    video_token_id=self.config.llm_config.image_patch_token,
                    image_start_token_id=self.config.llm_config.image_start_token,
                    video_start_token_id=self.config.llm_config.video_start_token,
                    image_grid_thw=None,
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                )
            else:
                rope_deltas = None

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=words_embeddings,
                use_cache=use_cache,
                image_mask=None,
                audio_mask=audio_mask,
                rope_deltas=rope_deltas,
                **generate_kwargs,
            )
        return outputs

    
    def sample_tokens(self, text="", num_iter=64, cfg=1.0, progress=False, lang='zh', **kwargs):
        patch_size = kwargs["patch_size"]
        if lang == 'zh':
            num_iter = len(text) // 3  * 25
            print('evaluating zh')
        else:
            num_iter = len(text.split(' ')) // 3 * 25
            print('evaluating en')
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        prompt_text = kwargs["prompt_text"]

        z_prompt_vae_latent = kwargs["z_prompt_vae_latent"]
        text_inputs = self.tokenizer.encode(text)
        bsz, seq_len, z_dim = z_prompt_vae_latent.shape
        # process prompt_vae_latent in multi_patch mode
        if seq_len % patch_size != 0:
            cut_len = seq_len % patch_size
            z_prompt_vae_latent = z_prompt_vae_latent[:, :-cut_len, :]
        prompt_vae_latent, tok_past_key_values = self.audio.encode_unified_emb_from_latent(z_prompt_vae_latent.reshape(bsz, -1, z_dim), past_key_values=None, use_cache=True)

        
        prompt_vae_latent = prompt_vae_latent.reshape(-1, patch_size, prompt_vae_latent.shape[-1])
        z_prompt_vae_latent = z_prompt_vae_latent.reshape(-1, patch_size, z_prompt_vae_latent.shape[-1])
        vae_embeds = self.linear_proj_audio(prompt_vae_latent)
        vae_embeds = vae_embeds.reshape(bsz, -1, vae_embeds.shape[-1]) # [bsz, patch_num, vae_latent_dim]

        text_input_ids = (
            self.tokenizer.encode("<role>HUMAN</role>") +
            self.tokenizer.encode(prompt_text) +
            text_inputs +
            self.tokenizer.encode("<role>ASSISTANT</role>")
        )
        audio_prompt_part = [self.tokenizer.convert_tokens_to_ids("<audio>")] + [self.tokenizer.convert_tokens_to_ids('<audioPatch>')] * vae_embeds.size(1)
        input_ids = text_input_ids + audio_prompt_part
        print(self.tokenizer.decode(input_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).cuda()
        attention_mask = torch.ones(input_ids.shape).to(input_ids.device)
        
        if (
            self.config.llm_config.rope_scaling is not None
            and self.config.llm_config.rope_scaling["type"] == "3D"
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_token_id=self.config.llm_config.image_patch_token,
                video_token_id=self.config.llm_config.image_patch_token,
                image_start_token_id=self.config.llm_config.image_start_token,
                video_start_token_id=self.config.llm_config.video_start_token,
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            rope_deltas = None
                
        inputs_embeds = None
        past_key_values = None
        # generate latents
        result, stop_flag = [], False

        for step in indices:
            if inputs_embeds is None:
                inputs_embeds = self.model.get_input_embeddings()(input_ids)
                vae_latent_length = vae_embeds.shape[1]
                vae_insert_loc = len(text_input_ids) + 1
                inputs_embeds[0, vae_insert_loc:vae_insert_loc + vae_latent_length, :] = vae_embeds[0, :vae_latent_length, :]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    audio_mask=None,
                    image_mask=None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values 
                )
            past_key_values = outputs.past_key_values
            z_diff = outputs.hidden_states[-1][:, -1:, :]
            if step == 0:
                latent_history = z_prompt_vae_latent[-1:, :, :]


            sampled_token_latent,  trajectory= self.flowloss.sample(z_diff, latent_history, cfg, patch_size)
            result.append(sampled_token_latent)
            if self.stop_head(z_diff)[0][0].softmax(dim=-1)[1]>0.5 and step > 16:
                if not stop_flag:
                    print(f'StopInfo: {step} {num_iter-1}')
                    stop_flag = True
                yield sampled_token_latent, True
                break
            else:
                yield sampled_token_latent, False

            sampled_token_high_latent, tok_past_key_values = self.audio.encode_unified_emb_from_latent(sampled_token_latent.reshape(bsz, -1, z_dim), past_key_values=tok_past_key_values, use_cache=True)
            sampled_token_high_latent = sampled_token_high_latent.reshape(-1, patch_size, sampled_token_high_latent.shape[-1])
            z_prompt_vae_latent = z_prompt_vae_latent.reshape(-1, patch_size, z_dim)
            sampled_token_high_latent = sampled_token_high_latent[-1:, :, :]

            inputs_embeds = self.linear_proj_audio(sampled_token_high_latent)

            if rope_deltas is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
                if past_key_values and self.rope_deltas:
                    delta = past_key_values[0][1].shape[2] + self.rope_deltas
                elif past_key_values:
                    delta = torch.tensor(past_key_values[0][1].shape[2]).to(inputs_embeds.device)
                else:
                    delta = torch.tensor(0).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                
            attention_mask = torch.ones(inputs_embeds.shape[0], 1).to(inputs_embeds.device)
            latent_history = sampled_token_latent

        if not stop_flag:
            print(f'StopInfo: {step} {num_iter-1}')
            stop_flag = True

    @torch.inference_mode()
    def generate_tts(
        self,
        text,
        prompt_wav_path,
        prompt_text,
        patch_size=5,
        lang='zh',
        num_iter=10,
        cfg=2,
        sample_rate=16000,
        device='cuda:0',
        output_wav_path='out.wav',
        tokenizer=None,
        progress=True
    ):
        self.tokenizer = tokenizer
        waveform, sr = torchaudio.load(prompt_wav_path, backend='soundfile')
        if sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)   # [T]
        waveform = waveform.to(device)
        waveform_length = torch.tensor([waveform.size(-1)], dtype=torch.int).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            z_prompt_vae_latent, _ = self.audio.encode_latent(waveform, waveform_length)

        prompt_text = f"Please translate the text to speech.\n{prompt_text}"

        kwargs = {
            "z_prompt_vae_latent": z_prompt_vae_latent,
            "prompt_text": prompt_text,
            "patch_size": patch_size,
        }
        audio_buffer = None
        window_buffer = None
        past_key_values = None
        use_cache = True
        speech = []
        # sampled_tokens_list = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for sampled_tokens, last_chunk in self.sample_tokens(
                text=text,
                lang=lang,
                num_iter=num_iter, 
                cfg=cfg,
                progress=progress,
                **kwargs
            ):
                speech_tmp, audio_buffer, window_buffer, past_key_values = self.audio.decode(sampled_tokens, past_key_values=past_key_values, use_cache=use_cache, audio_buffer=audio_buffer, window_buffer=window_buffer, last_chunk=last_chunk)
                speech.append(speech_tmp)
                # sampled_tokens_list.append(sampled_tokens)

        speech = torch.cat(speech, dim=-1)
        torchaudio.save(output_wav_path, speech.cpu()[0], sample_rate=sample_rate)

        # for non-streaming decode
        # sampled_tokens = torch.cat(sampled_tokens_list, dim=1)
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        #     speech = self.audio.decode(sampled_tokens, past_key_values=None, use_cache=False)[0]
        # torchaudio.save(output_wav_path, speech.cpu()[0], sample_rate=sample_rate)

        return speech.cpu()[0]


    @torch.inference_mode()
    def generate_edit(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        waveform: Optional[torch.FloatTensor] = None,
        waveform_length: Optional[torch.FloatTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        patch_size=5,
        cfg=2,
        sample_rate=16000,
        output_wav_path='out.wav',
        tokenizer=None,
    ):
        self.tokenizer = tokenizer
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            z_latents, audio_embeds_lengths = self.audio.encode_latent(waveform, waveform_length)
            audio_embeds_lengths = audio_embeds_lengths.unsqueeze(0)
            bsz, seq_len, z_dim = z_latents.shape
            if seq_len % patch_size != 0:
                cut_len = seq_len % patch_size
                z_latents = z_latents[:, :-cut_len, :]
            audio_embeds_lengths = audio_embeds_lengths // patch_size

            audio_embeds, tok_past_key_values = self.audio.encode_unified_emb_from_latent(z_latents.reshape(bsz, -1, z_dim), past_key_values=None, use_cache=True)
            audio_embeds = audio_embeds.reshape(-1, patch_size, audio_embeds.shape[-1])
            z_latents = z_latents.reshape(-1, patch_size, z_latents.shape[-1])
            audio_embeds = self.linear_proj_audio(audio_embeds)
            audio_embeds = audio_embeds.reshape(bsz, -1, audio_embeds.shape[-1]) # [bsz, patch_num, vae_latent_dim]

            inputs_embeds, audio_mask = self.prompt_wrap_navit(
                input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1),
                audio_embeds,
                audio_embeds_lengths,
                audio_placeholder_loc_lens,
                None,  # noqa
            )

            if (
                self.config.llm_config.rope_scaling is not None
                and self.config.llm_config.rope_scaling["type"] == "3D"
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_token_id=self.config.llm_config.image_patch_token,
                    video_token_id=self.config.llm_config.image_patch_token,
                    image_start_token_id=self.config.llm_config.image_start_token,
                    video_start_token_id=self.config.llm_config.video_start_token,
                    image_grid_thw=None,
                    video_grid_thw=None,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                rope_deltas = None

            step = 0
            max_steps = max(int(input_ids.size(1) * 2.5), 100)
            past_key_values = None
            edited_text_result = []
            edited_audio_result = []
            gene_audio_turn = False
            audio_step = 0

            while step < max_steps:
                outputs = self.model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                    image_mask=None,
                    audio_mask=None,
                )
                logits = outputs.logits[:, -1, :]
                next_token = torch.softmax(logits, dim=-1).argmax(dim=-1)

                if not gene_audio_turn:
                    inputs_embeds = self.model.get_input_embeddings()(next_token).unsqueeze(0)
                    edited_text_result.append(next_token.item())
                else:
                    z_diff = outputs.hidden_states[-1][:, -1:, :]
                    if audio_step == 0:
                        latent_history = torch.zeros_like(z_latents[-1:, :, :]).to(z_latents.device)

                    sampled_token_latent, trajectory= self.flowloss.sample(z_diff, latent_history, cfg, patch_size)
                    edited_audio_result.append(sampled_token_latent)

                    sampled_token_high_latent, tok_past_key_values = self.audio.encode_unified_emb_from_latent(sampled_token_latent.reshape(bsz, -1, z_dim), past_key_values=tok_past_key_values, use_cache=True)
                    sampled_token_high_latent = sampled_token_high_latent.reshape(-1, patch_size, sampled_token_high_latent.shape[-1])

                    inputs_embeds = self.linear_proj_audio(sampled_token_high_latent)
                    latent_history = sampled_token_latent

                    audio_step += 1
                    if self.stop_head(z_diff)[0][0].softmax(dim=-1)[1]>0.5 and audio_step > 16:
                        break

                if self.tokenizer.decode(next_token) == "<gen_audio>":
                    gene_audio_turn = True

                past_key_values = outputs.past_key_values
                if rope_deltas is not None:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    if past_key_values and self.rope_deltas:
                        delta = past_key_values[0][1].shape[2] + self.rope_deltas
                    elif past_key_values:
                        delta = torch.tensor(past_key_values[0][1].shape[2]).to(inputs_embeds.device)
                    else:
                        delta = torch.tensor(0).to(inputs_embeds.device)
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                attention_mask = torch.ones(next_token.shape[0], 1).to(next_token.device)
                step += 1

            edited_audio_latents = torch.cat(edited_audio_result, dim = 1)
            edited_speech = self.audio.decode(edited_audio_latents)
            torchaudio.save(output_wav_path, edited_speech.cpu()[0], sample_rate=sample_rate)
            edited_text = self.tokenizer.decode(edited_text_result)
            return edited_speech, edited_text
