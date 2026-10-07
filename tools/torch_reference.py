#!/usr/bin/env python3
"""Validation-only FP32 implementation of the official Qwen3-ASR equations.

Requires CPU PyTorch and NumPy, not imported or invoked by either application.
Uses explicit PyTorch operators rather than Burn/CubeCL to detect layout,
normalization, attention-mask and rotary-position mistakes independently.
"""
import argparse
import json
import os
import struct
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class Weights:
    def __init__(self, directory):
        self.file = open(directory / "model.safetensors", "rb")
        length = struct.unpack("<Q", self.file.read(8))[0]
        self.meta = json.loads(self.file.read(length))
        self.start = length + 8
        self.cache = {}

    def __call__(self, name):
        if name not in self.cache:
            meta = self.meta[name]
            start, end = meta["data_offsets"]
            self.file.seek(self.start + start)
            value = np.frombuffer(self.file.read(end - start), dtype={"BF16": "<u2", "F32": "<f4"}[meta["dtype"]]).copy()
            value = torch.from_numpy(value)
            if meta["dtype"] == "BF16":
                value = value.view(torch.bfloat16).float()
            self.cache[name] = value.reshape(meta["shape"])
        return self.cache[name]

    def linear(self, x, name, bias=False):
        return F.linear(x, self(name + ".weight"), self(name + ".bias") if bias else None)

    def norm(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self(name + ".weight"), self(name + ".bias"), 1e-5)

    def rms(self, x, name):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self(name + ".weight")


def mel_features(path):
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16000 and wav.getsampwidth() == 2
        channels = wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32).reshape(-1, channels).mean(1) / 32768.0
    frames = len(samples) // 160
    padded = np.pad(samples, (200, 200), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 400)[::160][:frames]
    hann = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    power = np.abs(np.fft.rfft(windows * hann, axis=1)) ** 2
    maximum_mel = 15 + np.log(8) * 27 / np.log(6.4)
    edges_mel = np.linspace(0, maximum_mel, 130)
    edges = np.where(edges_mel < 15, edges_mel * 200 / 3, 1000 * np.exp((edges_mel - 15) * np.log(6.4) / 27))
    freq = np.arange(201) * 40
    rising = (freq[None, :] - edges[:-2, None]) / (edges[1:-1, None] - edges[:-2, None])
    falling = (edges[2:, None] - freq[None, :]) / (edges[2:, None] - edges[1:-1, None])
    filters = (np.maximum(0, np.minimum(rising, falling)) * 2 / (edges[2:, None] - edges[:-2, None])).astype(np.float32)
    mel = np.log10(np.maximum(filters @ power.T.astype(np.float32), 1e-10))
    mel = (np.maximum(mel, mel.max() - 8) + 4) / 4
    return torch.from_numpy(mel.astype(np.float32)), len(samples)


def encode(w, mel):
    p = "thinker.audio_tower"
    chunks = list(mel.T.split(100))
    lengths = [(len(c) + 7) // 8 for c in chunks]
    x = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2).unsqueeze(1)
    for i in range(1, 4):
        x = F.gelu(F.conv2d(x, w(f"{p}.conv2d{i}.weight"), w(f"{p}.conv2d{i}.bias"), stride=2, padding=1))
    b, c, f, t = x.shape
    x = w.linear(x.permute(0, 3, 1, 2).reshape(b, t, c * f), p + ".conv_out")
    frequency = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(448) / 447)
    phase = torch.arange(t)[:, None] * frequency[None, :]
    x = x + torch.cat([phase.sin(), phase.cos()], 1)
    x = torch.cat([value[:length] for value, length in zip(x, lengths)])
    window = t * 8
    for i in range(18):
        q = f"{p}.layers.{i}"
        n = w.norm(x, q + ".self_attn_layer_norm")
        queries = w.linear(n, q + ".self_attn.q_proj", True).reshape(-1, 14, 64).transpose(0, 1)
        keys = w.linear(n, q + ".self_attn.k_proj", True).reshape(-1, 14, 64).transpose(0, 1)
        values = w.linear(n, q + ".self_attn.v_proj", True).reshape(-1, 14, 64).transpose(0, 1)
        attention = []
        for start in range(0, len(x), window):
            Q, K, V = [value[:, start:start + window] for value in (queries, keys, values)]
            attention.append((torch.softmax(Q @ K.transpose(1, 2) / 8, -1) @ V).transpose(0, 1).reshape(-1, 896))
        x = x + w.linear(torch.cat(attention), q + ".self_attn.out_proj", True)
        x = x + w.linear(F.gelu(w.linear(w.norm(x, q + ".final_layer_norm"), q + ".fc1", True)), q + ".fc2", True)
    return w.linear(F.gelu(w.linear(w.norm(x, p + ".ln_post"), p + ".proj1", True)), p + ".proj2", True)


def decoder(w, x, cache, start):
    count = len(x)
    phase = torch.arange(start, start + count)[:, None] * (1000000.0 ** (-torch.arange(64).float() * 2 / 128))[None, :]
    cos = torch.cat([phase.cos(), phase.cos()], 1)
    sin = torch.cat([phase.sin(), phase.sin()], 1)
    causal = torch.arange(start + count)[None, :] <= torch.arange(start, start + count)[:, None]
    def rotate(x):
        return x * cos + torch.cat([-x[..., 64:], x[..., :64]], -1) * sin
    for i in range(28):
        p = f"thinker.model.layers.{i}"
        n = w.rms(x, p + ".input_layernorm")
        q = rotate(w.rms(w.linear(n, p + ".self_attn.q_proj").reshape(count, 16, 128), p + ".self_attn.q_norm").transpose(0, 1))
        k = rotate(w.rms(w.linear(n, p + ".self_attn.k_proj").reshape(count, 8, 128), p + ".self_attn.k_norm").transpose(0, 1))
        v = w.linear(n, p + ".self_attn.v_proj").reshape(count, 8, 128).transpose(0, 1)
        if cache[i] is not None:
            k = torch.cat([cache[i][0], k], 1)
            v = torch.cat([cache[i][1], v], 1)
        cache[i] = (k, v)
        k, v = [value.repeat_interleave(2, dim=0) for value in (k, v)]
        scores = q @ k.transpose(1, 2) / 128**0.5
        scores = scores.masked_fill(~causal, -1e30)
        attention = (scores.softmax(-1) @ v).transpose(0, 1).reshape(count, 2048)
        x = x + w.linear(attention, p + ".self_attn.o_proj")
        n = w.rms(x, p + ".post_attention_layernorm")
        x = x + w.linear(F.silu(w.linear(n, p + ".mlp.gate_proj")) * w.linear(n, p + ".mlp.up_proj"), p + ".mlp.down_proj")
        if os.environ.get("ASR_TRACE_DIR") and start < 80:
            directory = Path(os.environ["ASR_TRACE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in [("hidden", x), ("key", cache[i][0]), ("value", cache[i][1])]:
                value.contiguous().numpy().tofile(directory / f"{start}-layer-{i}-{name}.f32")
    return w.rms(x[-1:], "thinker.model.norm")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-ASR-0.6B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("auto", "chinese", "english"), default="auto")
    parser.add_argument("--encoder-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    w = Weights(args.model)
    mel, count = mel_features(args.audio)
    mel.numpy().tofile(args.output / "mel.f32")
    audio = encode(w, mel)
    audio.numpy().tofile(args.output / "encoder.f32")
    print(f"encoder {audio.shape}, {time.monotonic() - start:.2f}s", flush=True)
    if args.encoder_only:
        return
    prefix = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
    suffix = [151670, 151645, 198, 151644, 77091, 198]
    if args.language != "auto":
        vocab = json.loads((args.model / "vocab.json").read_text())
        suffix += [vocab["language"], vocab["Ġ" + args.language.capitalize()], 151704]
    embedding = w("thinker.model.embed_tokens.weight")
    x = torch.cat([embedding[prefix], audio, embedding[suffix]])
    cache = [None] * 28
    hidden = decoder(w, x, cache, 0)
    hidden.numpy().tofile(args.output / "prefill-hidden.f32")
    logits = F.linear(hidden, embedding)
    logits.numpy().tofile(args.output / "prefill-logits.f32")
    token = int(logits.argmax())
    tokens = []
    position = len(x)
    for _ in range(args.max_tokens):
        if token in (151643, 151645):
            break
        tokens.append(token)
        hidden = decoder(w, embedding[[token]], cache, position)
        position += 1
        token = int(F.linear(hidden, embedding).argmax())
    result = {"tokens": tokens, "audio_seconds": count / 16000, "elapsed_seconds": time.monotonic() - start, "torch": torch.__version__}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
