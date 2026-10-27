"""OVLA policy: olfaction + vision + language -> action.

  language: word embeddings + GRU                    -> 128
  vision:   2-layer CNN (egocentric), FiLM-conditioned on language,
            spatial soft-argmax keypoints + max-pool  -> 256
  smell:    log receptor stage + MLP over 20 stereo-sniff concentrations -> 64,
            plus language-driven attention over odour channels -> attended gradient (9)
  fusion:   MLP over [vision, smell, language]        -> 256 ("fused" layer, probed later)
  head:     linear -> 6 action logits

use_smell=False gives the VLA baseline: the smell input is zeroed, so the network
has the same capacity but no nose.
"""

import torch
import torch.nn as nn

from .env import ACTIONS, VIEW, ODORS, TEMPLATES, CONTAINER_ODORS, COLORS


def build_vocab():
    words = {"<pad>", "<unk>"}
    for templates in TEMPLATES.values():
        for t in templates:
            words.update(t.replace("{}", "").split())
    words.update(CONTAINER_ODORS)
    words.update(COLORS)
    return {w: i for i, w in enumerate(sorted(words))}


VOCAB = build_vocab()
MAX_LEN = 10
N_KEYPOINTS = 8
RECEPTOR_FLOOR = 1e-5


def tokenize(instructions):
    out = torch.zeros(len(instructions), MAX_LEN, dtype=torch.long)
    for i, s in enumerate(instructions):
        ids = [VOCAB.get(w, VOCAB["<unk>"]) for w in s.split()][:MAX_LEN]
        out[i, : len(ids)] = torch.tensor(ids)
    return out


class OVLAPolicy(nn.Module):
    def __init__(self, use_smell=True):
        super().__init__()
        self.use_smell = use_smell
        self.embed = nn.Embedding(len(VOCAB), 64, padding_idx=VOCAB["<pad>"])
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.film = nn.Linear(128, 2 * 32)
        self.keypoints = nn.Conv2d(32, N_KEYPOINTS, 1)
        self.register_buffer("grid", torch.linspace(-1, 1, VIEW))
        self.vis_out = nn.Linear(2 * N_KEYPOINTS + 32, 256)
        self.smell = nn.Sequential(nn.Linear(len(ODORS) * 5, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.odor_attn = nn.Linear(128, len(ODORS))
        self.fuse = nn.Sequential(nn.Linear(256 + 64 + 9 + 128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.head = nn.Linear(256, len(ACTIONS))

    def forward(self, vision, smell, tokens, return_fused=False):
        lengths = (tokens != VOCAB["<pad>"]).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embed(tokens), lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        lang = h[-1]

        v = torch.relu(self.conv1(vision))
        gamma, beta = self.film(lang).chunk(2, dim=-1)
        v = torch.relu(self.conv2(v) * (1 + gamma[..., None, None]) + beta[..., None, None])
        # Spatial soft-argmax: each keypoint map becomes an expected (x, y) offset from
        # the agent (the image centre), so "where is the red thing" is translation-invariant.
        maps = self.keypoints(v).flatten(2).softmax(-1).view(-1, N_KEYPOINTS, VIEW, VIEW)
        coords = torch.cat([(maps.sum(2) * self.grid).sum(-1), (maps.sum(3) * self.grid).sum(-1)], dim=-1)
        v = torch.relu(self.vis_out(torch.cat([coords, v.amax(dim=(2, 3))], dim=-1)))

        if not self.use_smell:
            smell = torch.zeros_like(smell)
        # Receptor stage: log response (Weber-Fechner). For a Gaussian plume the
        # left-right difference of log concentration is linear in the source offset.
        log_c = torch.log(smell + RECEPTOR_FLOOR) / 5
        s = self.smell(log_c)
        # Top-down odour attention: the instruction picks which channel to follow.
        # Output: the attended channel at the 5 sniff points, and its 4 spatial differences.
        w = self.odor_attn(lang).softmax(-1)
        attended = (log_c.view(-1, 5, len(ODORS)) * w[:, None, :]).sum(-1)
        gradient = attended[:, 1:] - attended[:, :1]

        fused = self.fuse(torch.cat([v, s, attended, gradient, lang], dim=-1))
        logits = self.head(fused)
        return (logits, fused) if return_fused else logits
