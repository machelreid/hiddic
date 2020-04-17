decode_strategy = BeamSearch(
    self.beam_size,
    batch_size=batch.batch_size,
    pad=self._tgt_pad_idx,
    bos=self._tgt_bos_idx,
    eos=self._tgt_eos_idx,
    n_best=self.n_best,
    global_scorer=self.global_scorer,
    min_length=self.min_length,
    max_length=self.max_length,
    return_attention=attn_debug or self.replace_unk,
    block_ngram_repeat=self.block_ngram_repeat,
    exclusion_tokens=self._exclusion_idxs,
    stepwise_penalty=self.stepwise_penalty,
    ratio=self.ratio,
)

GNMTGlobalScorer(alpha=None, beta=None, length_penalty="average", coverage_penalty=None)
