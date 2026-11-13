import torch
import torch.nn.functional as F
from fla.modules.causal_conv1d import causal_conv1d
from fla.ops.attn import parallel_attn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.nsa import parallel_nsa


def _cu(lens, device):
    cu_cpu = torch.zeros(len(lens) + 1, dtype=torch.long)
    cu_cpu[1:] = torch.tensor(lens).cumsum(0)
    return cu_cpu.to(device), cu_cpu


def _gdn_prefill(m, h, cu, cu_cpu):
    qkv, conv_state = causal_conv1d(x=m.qkv(h), weight=m.conv, activation="silu", backend="triton",
                                    cu_seqlens=cu, cu_seqlens_cpu=cu_cpu, output_final_state=True)
    q, k, v = qkv.unflatten(-1, (3, m.h, m.dk)).unbind(-3)
    a, b = m.ab(h).chunk(2, -1)
    o, S = chunk_gated_delta_rule(q, k, v, g=a, beta=b, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                                  use_gate_in_kernel=True, A_log=m.A_log, dt_bias=m.dt_bias, cu_seqlens=cu,
                                  cu_seqlens_cpu=cu_cpu, output_final_state=True)
    o = m.o_norm(o, m.g_proj(h).unflatten(-1, (m.h, m.dk)))
    return m.o_proj(o.flatten(-2)), [conv_state, S]


def _gdn_step(m, h, st):
    qkv, st[0] = causal_conv1d(x=m.qkv(h), weight=m.conv, activation="silu", backend="triton",
                               initial_state=st[0], output_final_state=True)
    q, k, v = qkv.unflatten(-1, (3, m.h, m.dk)).unbind(-3)
    a, b = m.ab(h).chunk(2, -1)
    o, st[1] = fused_recurrent_gated_delta_rule(q, k, v, g=a, beta=b, initial_state=st[1], output_final_state=True,
                                                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=m.A_log,
                                                dt_bias=m.dt_bias, use_beta_sigmoid_in_kernel=True)
    o = m.o_norm(o, m.g_proj(h).unflatten(-1, (m.h, m.dk)))
    return m.o_proj(o.flatten(-2))


def _nsa_qkvg(m, h):
    q = m.q_norm(m.q_proj(h).unflatten(-1, (m.hq, m.dh)))
    k, v = m.kv_proj(h).unflatten(-1, (2, m.hkv, m.dh)).unbind(-3)
    k = m.k_norm(k)
    g, og = m.gate_proj(h).split([3 * m.hq, m.hq * m.dh], -1)
    g_cmp, g_slc, g_swa = g.unflatten(-1, (m.hq, 3)).sigmoid().unbind(-1)
    return q, k, v, g_cmp, g_slc, g_swa, og


def _nsa_prefill(m, h, cu, lens, tmax):
    q, k, v, g_cmp, g_slc, g_swa, og = _nsa_qkvg(m, h)
    o = parallel_nsa(q, k, v, g_cmp=g_cmp, g_slc=g_slc, block_counts=m.block_counts, block_size=m.block_size,
                     window_size=0, cu_seqlens=cu)
    if m.window > 0:
        o = o + parallel_attn(q, k, v, window_size=m.window, cu_seqlens=cu) * g_swa.unsqueeze(-1)
    B = len(lens)
    kb = k.new_zeros(B, tmax, m.hkv, m.dh)
    vb = v.new_zeros(B, tmax, m.hkv, m.dh)
    pos = torch.cat([torch.arange(n, device=h.device) for n in lens])
    row = torch.repeat_interleave(torch.arange(B, device=h.device), torch.tensor(lens, device=h.device))
    kb[row, pos] = k[0]
    vb[row, pos] = v[0]
    return m.o_proj(o.flatten(-2) * torch.sigmoid(og)), [kb, vb]


def _nsa_step(m, h, st, lens):
    q, k, v, g_cmp, g_slc, g_swa, og = _nsa_qkvg(m, h)
    kb, vb = st
    B = h.shape[0]
    ar = torch.arange(B, device=h.device)
    kb[ar, lens] = k[:, 0]
    vb[ar, lens] = v[:, 0]
    n = lens + 1
    valid = torch.arange(kb.shape[1], device=h.device)[None] < n[:, None]
    cu_k = F.pad(n.cumsum(0), (1, 0))
    cu_q = torch.arange(B + 1, device=h.device)
    o = parallel_nsa(q.transpose(0, 1), kb[valid][None], vb[valid][None], g_cmp=g_cmp.transpose(0, 1),
                     g_slc=g_slc.transpose(0, 1), block_counts=m.block_counts, block_size=m.block_size,
                     window_size=0, cu_seqlens=(cu_q, cu_k)).transpose(0, 1)
    if m.window > 0:
        idx = n[:, None] - m.window + torch.arange(m.window, device=h.device)[None]
        ok = idx >= 0
        idx = idx.clamp(min=0)
        kw = kb[ar[:, None], idx][:, :, 0].float()
        vw = vb[ar[:, None], idx][:, :, 0].float()
        s = torch.einsum("bhd,bwd->bhw", q[:, 0].float(), kw) * m.dh ** -0.5
        s = s.masked_fill(~ok[:, None], float("-inf"))
        sw = torch.einsum("bhw,bwd->bhd", s.softmax(-1), vw)
        o = o + (sw * g_swa[:, 0, :, None]).to(o.dtype)[:, None]
    return m.o_proj(o.flatten(-2) * torch.sigmoid(og))


class Cache:
    def __init__(self, states, lens):
        self.states = states
        self.lens = lens

    def keep(self, idx):
        self.lens = self.lens[idx]
        self.states = [[t[idx] for t in st] for st in self.states]


@torch.no_grad()
def prefill(model, prompts, tmax):
    dev = model.embed.weight.device
    lens = [len(p) for p in prompts]
    cu, cu_cpu = _cu(lens, dev)
    x = model.embed(torch.tensor([t for p in prompts for t in p], device=dev)[None])
    states = []
    for blk in model.blocks:
        h = blk.norm(x)
        if blk.kind == "G":
            o, st = _gdn_prefill(blk.mixer, h, cu, cu_cpu)
        else:
            o, st = _nsa_prefill(blk.mixer, h, cu, lens, tmax)
        states.append(st)
        x = blk.mlp(x + o)
    last = model.norm(x)[0, cu[1:] - 1]
    return F.linear(last, model.embed.weight).float(), Cache(states, torch.tensor(lens, device=dev))


@torch.no_grad()
def step(model, tokens, cache):
    x = model.embed(tokens[:, None])
    for blk, st in zip(model.blocks, cache.states):
        h = blk.norm(x)
        o = _gdn_step(blk.mixer, h, st) if blk.kind == "G" else _nsa_step(blk.mixer, h, st, cache.lens)
        x = blk.mlp(x + o)
    cache.lens = cache.lens + 1
    return F.linear(model.norm(x)[:, 0], model.embed.weight).float()


def sample(logits, temperature, top_p, gen):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, -1)
    if top_p < 1.0:
        sp, si = probs.sort(-1, descending=True)
        sp = sp * ((sp.cumsum(-1) - sp) < top_p)
        return si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)).squeeze(-1)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


@torch.no_grad()
def generate(model, prompts, max_new, stop_ids, temperature=1.0, top_p=1.0, gen=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tmax = max(len(p) for p in prompts) + max_new
        logits, cache = prefill(model, prompts, tmax)
        dev = logits.device
        stop = torch.tensor(sorted(stop_ids), device=dev)
        alive = torch.arange(len(prompts), device=dev)
        outs = [[] for _ in prompts]
        done = [False] * len(prompts)
        for t in range(max_new):
            nxt = sample(logits, temperature, top_p, gen)
            hit = torch.isin(nxt, stop)
            for i, tok, h in zip(alive.tolist(), nxt.tolist(), hit.tolist()):
                outs[i].append(tok)
                done[i] = h
            if t == max_new - 1:
                break
            if hit.any():
                keep = (~hit).nonzero().squeeze(-1)
                if keep.numel() == 0:
                    break
                alive, nxt = alive[keep], nxt[keep]
                cache.keep(keep)
            logits = step(model, nxt, cache)
    return outs, done


@torch.no_grad()
def stream(model, prompt, max_new, stop_ids, temperature=1.0, top_p=1.0, gen=None):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, cache = prefill(model, [prompt], len(prompt) + max_new)
    for t in range(max_new):
        if t:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = step(model, nxt, cache)
        nxt = sample(logits, temperature, top_p, gen)
        tok = nxt.item()
        if tok in stop_ids:
            return
        yield tok
