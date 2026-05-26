"""Tiny shim package that lets MILAN's `from allennlp.nn import beam_search`
import succeed on Python 3.13 where the real allennlp can't install.

We only implement the slice of the API that MILAN's `src/milan/decoders.py`
actually calls: `BeamSearch(end_index, max_steps, beam_size).search(...)`.
"""
