#!/usr/bin/env python3
"""Ensemble policies with a leading M (model) dimension: M independent policies run as ONE batched GPU
kernel via grouped parameters (einsum 'm b i, m i o -> m b o'). Each policy m has its own weights and acts
only on its own worlds, so there is no cross-policy gradient leakage (verified by G3 in train_gpu.py).

Inputs/outputs use shape (M, U, N, .) where M=policies, U=world-timestep units, N=agents. MLP acts per agent;
GNN runs L rounds of learned message passing over the per-(M,U) proximity graph (adjacency (M,U,N,N))."""
import math
import torch
import torch.nn as nn


class GroupedLinear(nn.Module):
    """M independent linear layers applied in parallel. x:(M,B,fin) -> (M,B,fout)."""
    def __init__(self, M, fin, fout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(M, fin, fout))
        self.bias = nn.Parameter(torch.zeros(M, 1, fout))
        bound = 1.0 / math.sqrt(fin)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):                       # x (M,B,fin)
        return torch.einsum("mbi,mio->mbo", x, self.weight) + self.bias


class EnsembleMLP(nn.Module):
    """Per-agent feedforward actor+critic, M independent copies. (IPPO: local per-agent critic.)"""
    def __init__(self, M, F, h=64, n_act=3):
        super().__init__()
        self.a1 = GroupedLinear(M, F, h); self.a2 = GroupedLinear(M, h, h); self.ao = GroupedLinear(M, h, n_act)
        self.c1 = GroupedLinear(M, F, h); self.c2 = GroupedLinear(M, h, h); self.co = GroupedLinear(M, h, 1)

    def forward(self, feats, adj=None):          # feats (M,U,N,F)
        M, U, N, F = feats.shape
        x = feats.reshape(M, U * N, F)
        a = torch.relu(self.a1(x)); a = torch.relu(self.a2(a)); logits = self.ao(a)
        c = torch.relu(self.c1(x)); c = torch.relu(self.c2(c)); val = self.co(c).squeeze(-1)
        return logits.reshape(M, U, N, -1), val.reshape(M, U, N)


class EnsembleGNN(nn.Module):
    """Learned message passing (L rounds, shared-round weights) over the world graph, M independent copies.
    agg_i = mean_{j~i} msg(h_j);  h_i <- relu(upd([h_i, agg_i]))."""
    def __init__(self, M, F, h=64, L=3, n_act=3):
        super().__init__()
        self.L = L; self.h = h
        self.enc = GroupedLinear(M, F, h)
        self.msg = GroupedLinear(M, h, h)
        self.upd = GroupedLinear(M, 2 * h, h)
        self.ao = GroupedLinear(M, h, n_act); self.co = GroupedLinear(M, h, 1)

    def forward(self, feats, adj):               # feats (M,U,N,F), adj (M,U,N,N) float {0,1}
        M, U, N, F = feats.shape
        h = torch.relu(self.enc(feats.reshape(M, U * N, F))).reshape(M, U, N, self.h)
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)          # (M,U,N,1)
        for _ in range(self.L):
            m = self.msg(h.reshape(M, U * N, self.h)).reshape(M, U, N, self.h)
            agg = torch.einsum("muij,mujh->muih", adj, m) / deg  # mean over neighbours
            cat = torch.cat([h, agg], dim=-1).reshape(M, U * N, 2 * self.h)
            h = torch.relu(self.upd(cat)).reshape(M, U, N, self.h)
        hx = h.reshape(M, U * N, self.h)
        logits = self.ao(hx).reshape(M, U, N, -1)
        val = self.co(hx).squeeze(-1).reshape(M, U, N)
        return logits, val
