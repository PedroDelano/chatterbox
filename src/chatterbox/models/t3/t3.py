# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm
from transformers import LlamaConfig, LlamaModel
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

from ..utils import AttrDict
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .llama_configs import LLAMA_CONFIGS
from .modules.cond_enc import T3Cond, T3CondEnc
from .modules.learned_pos_emb import LearnedPositionEmbeddings
from .modules.t3_config import T3Config

logger = logging.getLogger(__name__)
SPEECH_VOCAB_SIZE = 6561


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, (
        "missing start_text_token"
    )
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, (
        "missing stop_text_token"
    )


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=T3Config()):
        super().__init__()
        self.hp = hp
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False
        self.patched_model = None

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(
            self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False
        )
        self.speech_head = nn.Linear(
            self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False
        )
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if (
            t3_cond.cond_prompt_speech_tokens is not None
            and t3_cond.cond_prompt_speech_emb is None
        ):
            t3_cond.cond_prompt_speech_emb = self.speech_emb(
                t3_cond.cond_prompt_speech_tokens
            ) + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack(
            [
                torch.cat((ce, te, se))
                for ce, te, se in zip(cond_emb, text_emb, speech_emb)
            ]
        )  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[
            -1
        ]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, : ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, : stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = (
            torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]
        )  # (B, len_text)
        mask_speech = (
            torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]
        )  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(
            out.text_logits, masked_text, ignore_index=IGNORE_ID
        )
        loss_speech = F.cross_entropy(
            out.speech_logits, masked_speech, ignore_index=IGNORE_ID
        )

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor] = None,
        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor] = None,
        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        min_p=0.05,
        top_p=1.00,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(
            dtype=torch.long, device=self.device
        )

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(
                text_tokens[:, :1]
            )

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        self.compiled = False

        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if self.patched_model is None:
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=None,
            )
            self.patched_model = patched_model
            self.compiled = True

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     min_p=min_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor(
            [[self.hp.start_speech_token]], dtype=torch.long, device=device
        )
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input if cfg_weight > 0
        if cfg_weight > 0:
            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        else:
            inputs_embeds = embeds

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        predicted = []  # To store the predicted tokens

        # Instantiate the logits processors.
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=float(repetition_penalty)
        )

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,  # Changed to False - SDPA doesn't support True
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            logits = output.logits[:, -1, :]

            # CFG
            if cfg_weight > 0.0:
                logits_cond = logits[0:1]
                logits_uncond = logits[1:2]
                logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

            logits = logits.squeeze(1)

            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature

            # Apply repetition penalty and top‑p filtering.
            try:
                logits = repetition_penalty_processor(generated_ids, logits)
            except Exception as e:
                logger.warning(f"Error in repetition penalty processor: {e}")
                # Continue without repetition penalty if it fails

            logits = min_p_warper(None, logits)
            logits = top_p_warper(None, logits)

            # Mask out invalid tokens to prevent generation of out-of-bounds tokens
            if logits.size(-1) > self.hp.speech_tokens_dict_size:
                logits[..., self.hp.speech_tokens_dict_size :] = float("-inf")

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token.
            if next_token.view(-1) == self.hp.stop_speech_token:
                break

            # Get embedding for the new token.
            # Ensure token is within embedding bounds
            clamped_token = torch.clamp(
                next_token, 0, self.speech_emb.num_embeddings - 1
            )
            if not torch.equal(clamped_token, next_token):
                logger.warning(
                    f"Token {next_token.item()} clamped to {clamped_token.item()}"
                )

            next_token_embed = self.speech_emb(clamped_token)
            next_token_embed = (
                next_token_embed
                + self.speech_pos_emb.get_fixed_embedding(
                    min(i + 1, self.speech_pos_emb.emb.num_embeddings - 1)
                )
            )

            #  For CFG
            if cfg_weight > 0.0:
                next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=False,  # Changed to False - SDPA doesn't support True
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values

        # Concatenate all predicted tokens along the sequence dimension.
        predicted_tokens = torch.cat(predicted, dim=1)  # shape: (B, num_tokens)
        return predicted_tokens

    @torch.inference_mode()
    def streaming_inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        s3gen_model,  # S3Gen model for audio generation
        ref_dict: dict,  # Reference dictionary for S3Gen
        chunk_size: int = 25,  # Number of tokens per audio chunk
        overlap_size: int = 5,  # Number of overlapping tokens between chunks
        min_tokens_for_audio: int = 10,  # Minimum tokens needed before generating audio
        initial_speech_tokens: Optional[Tensor] = None,
        # HF generate args
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        min_p=0.05,
        top_p=1.00,
        repetition_penalty=1.2,
        cfg_weight=0,
    ):
        """
        Streaming inference that yields audio chunks as tokens are generated.

        Args:
            t3_cond: T3 conditioning data
            text_tokens: Input text tokens
            s3gen_model: S3Gen model for converting tokens to audio
            ref_dict: Reference dictionary for S3Gen conditioning
            chunk_size: Number of tokens to accumulate before generating audio
            overlap_size: Number of tokens to overlap between chunks for smooth audio
            min_tokens_for_audio: Minimum number of tokens needed before first audio generation
            initial_speech_tokens: Optional initial speech tokens
            **kwargs: Standard generation parameters

        Yields:
            torch.Tensor: Audio chunks as they are generated
        """
        # Validate / sanitize inputs
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(
            dtype=torch.long, device=self.device
        )

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(
                text_tokens[:, :1]
            )

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # Prepare the patched model if needed
        if self.patched_model is None:
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=None,
            )
            self.patched_model = patched_model

        device = embeds.device
        max_new_tokens = max_new_tokens or self.hp.max_speech_tokens

        bos_token = torch.tensor(
            [[self.hp.start_speech_token]], dtype=torch.long, device=device
        )
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        if cfg_weight > 0:
            bos_embed = torch.cat([bos_embed, bos_embed])
            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        else:
            inputs_embeds = embeds

        # Track generated token ids; start with the BOS token.
        generated_ids = bos_token.clone()
        all_generated_tokens = []  # Store all tokens for audio generation
        last_audio_end_idx = 0  # Track where the last audio chunk ended

        # Instantiate the logits processors.
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=float(repetition_penalty)
        )

        # ---- Initial Forward Pass (no kv_cache yet) ----
        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=False,  # Changed to False - SDPA doesn't support True
            output_hidden_states=True,
            return_dict=True,
        )
        # Initialize kv_cache with the full context.
        past = output.past_key_values

        # Get max position embedding size for bounds checking
        max_pos_embedding = self.speech_pos_emb.emb.num_embeddings - 1
        max_speech_embedding = self.speech_emb.num_embeddings - 1

        logger.info(
            f"Max position embedding: {max_pos_embedding}, Max speech embedding: {max_speech_embedding}"
        )
        logger.info(f"Speech tokens dict size: {self.hp.speech_tokens_dict_size}")
        logger.info(
            f"Start token: {self.hp.start_speech_token}, Stop token: {self.hp.stop_speech_token}"
        )

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Streaming TTS", dynamic_ncols=True):
            logits = output.logits[:, -1, :]

            # CFG
            if cfg_weight > 0.0:
                logits_cond = logits[0:1]
                logits_uncond = logits[1:2]
                logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

            logits = logits.squeeze(1)

            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature

            # Apply repetition penalty and top‑p filtering.
            try:
                logits = repetition_penalty_processor(generated_ids, logits)
            except Exception as e:
                logger.warning(f"Error in repetition penalty processor: {e}")

            logits = min_p_warper(None, logits)
            logits = top_p_warper(None, logits)

            # Mask out invalid tokens
            if logits.size(-1) > self.hp.speech_tokens_dict_size:
                logits[..., self.hp.speech_tokens_dict_size :] = float("-inf")

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

            # Validate token is within vocabulary
            if next_token.item() >= self.hp.speech_tokens_dict_size:
                logger.warning(
                    f"Generated token {next_token.item()} exceeds vocabulary size {self.hp.speech_tokens_dict_size}"
                )
                next_token = torch.tensor([[self.hp.stop_speech_token]], device=device)

            # Ensure token is non-negative
            if next_token.item() < 0:
                logger.warning(
                    f"Generated negative token {next_token.item()}, setting to start token"
                )
                next_token = torch.tensor([[self.hp.start_speech_token]], device=device)

            all_generated_tokens.append(next_token.item())
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS token BEFORE filtering (this is important!)
            if stop_on_eos and next_token.view(-1) == self.hp.stop_speech_token:
                # Generate final audio chunk with remaining tokens
                if len(all_generated_tokens) > last_audio_end_idx:
                    try:
                        remaining_tokens = all_generated_tokens[last_audio_end_idx:]
                        valid_remaining = [
                            token
                            for token in remaining_tokens
                            if token < SPEECH_VOCAB_SIZE
                        ]

                        if valid_remaining:
                            remaining_tensor = torch.tensor(
                                valid_remaining, dtype=torch.long, device=device
                            ).unsqueeze(0)

                            wav, _ = s3gen_model.inference(
                                speech_tokens=remaining_tensor,
                                ref_dict=ref_dict,
                            )
                            yield wav.squeeze(0).detach().cpu()
                    except Exception as e:
                        logger.warning(f"Failed to generate final audio chunk: {e}")
                break  # This break is crucial!

            # Generate audio chunk when we have enough tokens
            tokens_available = len(all_generated_tokens)
            if (
                tokens_available >= min_tokens_for_audio
                and tokens_available - last_audio_end_idx >= chunk_size
            ):
                try:
                    # Determine the range of tokens for this chunk
                    chunk_start = max(0, last_audio_end_idx - overlap_size)
                    chunk_end = min(
                        last_audio_end_idx + chunk_size, len(all_generated_tokens)
                    )

                    # Extract tokens for this chunk
                    chunk_tokens = all_generated_tokens[chunk_start:chunk_end]

                    valid_tokens = [
                        token for token in chunk_tokens if token < SPEECH_VOCAB_SIZE
                    ]

                    # Only proceed if we have valid tokens
                    if not valid_tokens:
                        logger.warning(
                            "No valid tokens in chunk, skipping audio generation"
                        )
                        last_audio_end_idx = chunk_end
                        continue

                    # Create tensor WITHOUT any start token
                    chunk_tensor = torch.tensor(
                        valid_tokens, dtype=torch.long, device=device
                    ).unsqueeze(0)

                    # Generate audio for this chunk
                    wav, _ = s3gen_model.inference(
                        speech_tokens=chunk_tensor,
                        ref_dict=ref_dict,
                    )

                    # Handle overlap trimming if needed...
                    if chunk_start < last_audio_end_idx and last_audio_end_idx > 0:
                        overlap_tokens = last_audio_end_idx - chunk_start
                        overlap_samples = int(overlap_tokens * 24000 / 25)
                        if overlap_samples < wav.shape[-1]:
                            wav = wav[..., overlap_samples:]

                    yield wav.squeeze(0).detach().cpu()
                    last_audio_end_idx = chunk_end

                except Exception as e:
                    logger.warning(
                        f"Failed to generate audio chunk at token {tokens_available}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    # Continue generation even if audio generation fails

            # Get embedding for the new token with bounds checking
            # Ensure token is within embedding bounds
            clamped_token = torch.clamp(
                next_token, 0, self.speech_emb.num_embeddings - 1
            )
            if not torch.equal(clamped_token, next_token):
                logger.warning(
                    f"Token {next_token.item()} clamped to {clamped_token.item()} for speech embedding"
                )

            next_token_embed = self.speech_emb(clamped_token)

            # Bounds check for position embeddings
            pos_idx = min(i + 1, max_pos_embedding)
            next_token_embed = (
                next_token_embed + self.speech_pos_emb.get_fixed_embedding(pos_idx)
            )

            #  For CFG
            if cfg_weight > 0.0:
                next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=False,  # Changed to False - SDPA doesn't support True
                output_hidden_states=True,
                return_dict=True,
            )
            # Update the kv_cache.
            past = output.past_key_values
