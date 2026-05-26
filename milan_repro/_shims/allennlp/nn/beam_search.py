"""Minimal beam-search shim for MILAN's decoder.

MILAN uses `allennlp.nn.beam_search.BeamSearch` like this (decoders.py:467):

    runner = beam_search.BeamSearch(stop_index, max_steps=L, beam_size=K)
    tokens, scores = runner.search(start_predictions, start_state, step)

Where:
    start_predictions: (B,)
    start_state: dict[str, Tensor]  (leading dim B)
    step(tokens, state) -> (log_probs, new_state)
                          log_probs: (N, V), N is current batch size
    returns:
        tokens: (B, K, T)
        scores: (B, K)

This implementation runs a real, batched beam search over a vocab of size V.
It's not feature-complete with allennlp (no constraints, samplers, length
normalisation) but matches the contract MILAN relies on.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

State = Dict[str, torch.Tensor]
StepFn = Callable[[torch.Tensor, State], Tuple[torch.Tensor, State]]


def _same_type_dict(template: State, data: Dict[str, torch.Tensor]) -> State:
    """Return a dict of the same concrete class as `template`, with `data`.

    MILAN passes `AllenNLPDecoderState` (a dict subclass) and asserts the
    state retains that type across step calls. Constructing via __new__ +
    dict.__init__ sidesteps the custom __init__ that wants a DecoderState.
    """
    cls = type(template)
    if cls is dict:
        return data
    inst = cls.__new__(cls)
    dict.__init__(inst, data)
    return inst


def _replicate_state(state: State, k: int) -> State:
    """Expand each tensor's leading dim from B to B*k by repeat-interleave."""
    new = {key: t.repeat_interleave(k, dim=0) for key, t in state.items()}
    return _same_type_dict(state, new)


def _reorder_state(state: State, parent_beams: torch.Tensor,
                   B: int, K: int) -> State:
    """Reorder rows of state[B*K, ...] by `parent_beams[B, K]` → new B*K layout."""
    flat_idx = (torch.arange(B, device=parent_beams.device).unsqueeze(1) * K
                + parent_beams).reshape(-1)
    new = {key: t.index_select(0, flat_idx) for key, t in state.items()}
    return _same_type_dict(state, new)


class BeamSearch:
    """Drop-in replacement for `allennlp.nn.beam_search.BeamSearch` (subset)."""

    def __init__(self, end_index: int, max_steps: int, beam_size: int,
                 **_: object):
        self.end_index = int(end_index)
        self.max_steps = int(max_steps)
        self.beam_size = int(beam_size)

    @torch.no_grad()
    def search(self,
               start_predictions: torch.Tensor,
               start_state: State,
               step: StepFn) -> Tuple[torch.Tensor, torch.Tensor]:
        B = start_predictions.size(0)
        K = self.beam_size
        device = start_predictions.device
        NEG_INF = -1e9

        # --- Step 0: expand to K beams from the single start. -------------
        log_probs, state = step(start_predictions, start_state)   # (B, V)
        V = log_probs.size(-1)
        top_logp, top_tok = log_probs.topk(K, dim=-1)             # (B, K)

        # Replicate state to B*K rows.
        state = _replicate_state(state, K)

        # Initialise beams.
        preds = top_tok.unsqueeze(-1)                              # (B, K, 1)
        scores = top_logp                                          # (B, K)
        finished = top_tok.eq(self.end_index)                      # (B, K)

        # End-distribution: if a beam ended, it can only emit end_index.
        end_dist = torch.full((V,), NEG_INF, device=device,
                              dtype=log_probs.dtype)
        end_dist[self.end_index] = 0.0

        # --- Decode the remaining max_steps-1 tokens. ---------------------
        for _t in range(1, self.max_steps):
            if finished.all():
                break

            cur_tok = preds[..., -1].reshape(-1)                  # (B*K,)
            log_probs, state = step(cur_tok, state)               # (B*K, V)
            log_probs = log_probs.view(B, K, V)

            # Force finished beams to stay finished.
            if finished.any():
                log_probs = torch.where(
                    finished.unsqueeze(-1),
                    end_dist.expand_as(log_probs),
                    log_probs,
                )

            total = scores.unsqueeze(-1) + log_probs              # (B, K, V)
            flat = total.view(B, K * V)
            top_scores, top_idx = flat.topk(K, dim=-1)            # (B, K)
            parent_beams = top_idx // V                            # (B, K)
            token_idx = top_idx % V                                # (B, K)

            # Gather past predictions along the chosen parent beams.
            t_size = preds.size(-1)
            preds = preds.gather(
                1, parent_beams.unsqueeze(-1).expand(-1, -1, t_size))
            preds = torch.cat([preds, token_idx.unsqueeze(-1)], dim=-1)
            scores = top_scores

            # Reorder state by parent beams.
            state = _reorder_state(state, parent_beams, B, K)

            # Update finished mask.
            finished = (finished.gather(1, parent_beams)
                        | token_idx.eq(self.end_index))

        # Pad to max_steps with end_index for shape consistency.
        if preds.size(-1) < self.max_steps:
            pad = torch.full(
                (B, K, self.max_steps - preds.size(-1)),
                self.end_index, device=device, dtype=preds.dtype)
            preds = torch.cat([preds, pad], dim=-1)

        return preds, scores
