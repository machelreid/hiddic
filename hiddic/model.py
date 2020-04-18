import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.span_extractors import (
    EndpointSpanExtractor,
    SelfAttentiveSpanExtractor,
)
from allennlp.modules.scalar_mix import ScalarMix
from utils import sequence_mask, find_subtensor
from dotmap import DotMap
import random
from beam import BeamSearch
import onmt.modules.GlobalAttention as AttentionLayer


class DefinitionProbing(nn.Module):
    def __init__(
        self,
        encoder,
        encoder_pretrained,
        decoder_hidden,
        embeddings,
        max_layer=12,
        src_pad_idx=0,
        encoder_hidden=None,
    ):
        super(DefinitionProbing, self).__init__()

        self.encoder_hidden = encoder_hidden
        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        if encoder_pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder_hidden = self.encoder.config.hidden_size
        self.max_layer = max_layer
        self.span_extractor = SelfAttentiveSpanExtractor(self.encoder_hidden)
        self.decoder = LSTM_Decoder(
            embeddings.tgt,
            hidden=decoder,
            encoder_hidden=encoder.config.hidden_size,
            num_layers=2,
            teacher_forcing_p=0.3,
            attention=None,
            dropout=DotMap({"input": 0.5, "output": 0.3}),
        )
        self.scalar_mix = ScalarMix(self.max_layer + 1)
        self.global_scorer = GNMTGlobalScorer(
            alpha=None, beta=None, length_penalty="average", coverage_penalty=None
        )

    def forward(self, input, seq_lens, span_token_ids, target):
        batch_size, tgt_len = target.shape

        # (batch_size,seq_len,hidden_size), (batch_size,hidden_size), (num_layers,batch_size,seq_len,hidden_size)
        last_hidden_layer, pooled_representation, all_hidden_layers = self.encoder(
            input, attention_mask=sequence_mask(seq_lens)
        )

        span_ids = self._id_extractor(tokens=span_token_ids, batch=input, lens=seq_lens)

        span_representation = self._span_aggregator(
            all_hidden_layers, sequence_mask(seq_lens), span_ids
        )

        predictions, logits = self.decoder(target, span_representation)

        loss = F.cross_entropy(
            logits.view(batch_size * tgt_len - 1),
            target[:, 1:].view(-1),
            ignore_index=self.embeddings.tgt.padding_idx,
        )

        return DotMap({"predicitions": predictions, "logits": logits, "loss": loss})

    def _validate(
        self, input, seq_lens, span_token_ids, target, tgt_lens, decode_strategy
    ):
        batch_size, tgt_len = target.shape

        # (batch_size,seq_len,hidden_size), (batch_size,hidden_size), (num_layers,batch_size,seq_len,hidden_size)
        last_hidden_layer, pooled_representation, all_hidden_layers = self.encoder(
            input, attention_mask=sequence_mask(seq_lens)
        )

        span_ids = self._id_extractor(tokens=npan_token_ids, batch=input, lens=seq_lens)

        memory_bank = last_hidden_layer if self.decoder.attention else None
        _, logits = self.decoder(target, span_representation, memory_bank)

        loss = F.cross_entropy(
            logits.view(batch_size * tgt_len - 1),
            target[:, 1:].view(-1),
            ignore_index=self.embeddings.tgt.padding_idx,
        )

        ppl = loss.exp()
        beam_results = self._strategic_decode(
            target, tgt_lens, decode_strategy, span_representation
        )
        return DotMap(
            {
                "predictions": beam_results["predictions"],
                "logits": logits,
                "loss": loss,
                "perplexity": ppl,
            }
        )

    def _strategic_decode(self, target, tgt_lens, decode_strategy, span_representation):
        """Translate a batch of sentences step by step using cache.
        Args:
            batch: a batch of sentences, yield by data iterator.
            src_vocabs (list): list of torchtext.data.Vocab if can_copy.
            decode_strategy (DecodeStrategy): A decode strategy to use for
                generate translation step by step.
        Returns:
            results (dict): The translation results.
        """
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        # (0) Prep the components of the search.
        # use_src_map = self.copy_attn
        batch_size, max_len = target.shape

        memory_bank = last_hidden_layer if self.decoder.attention else None

        # Initialize the hidden states
        self.model.decoder.init_state(span_representation)

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            # "gold_score": self._gold_score(
            #    batch, memory_bank, src_lengths, src_vocabs, use_src_map,
            #    enc_states, batch_size, src)
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None  # batch.src_map if use_src_map else None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank, tgt_lens, src_map
        )
        if fn_map_state is not None:
            self.model.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            logits, attn = self.decoder.generate(decoder_input)

            decode_strategy.advance(F.log_softmax(log_probs, 1), attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if memory_bank is not None:
                    if isinstance(memory_bank, tuple):
                        memory_bank = tuple(
                            x.index_select(1, select_indices) for x in memory_bank
                        )
                    else:
                        memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        if self.report_align:
            results["alignment"] = self._align_forward(
                batch, decode_strategy.predictions
            )
        else:
            results["alignment"] = [[] for _ in range(batch_size)]
        return results

    def _span_aggregator(
        self, hidden_states, input_mask, span_ids, layer_no: int = None,
    ):

        if layer_no is not None:
            hidden_states = hidden_states[layer_no]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[: self.max_layer + 1]
            hidden_states = self.scalar_mix(hidden_states, mask=input_mask)
        span = self.span_extractor(hidden_states, span_ids).squeeze(
            1
        )  # As we will only be extracting one span per sequence
        return span

    def _id_extractor(self, tokens, batch, lens):
        """
        Extracts span indices given a sequence, if none found returns the span as the start and end of sequence as the span
        """
        output_ids = []
        for w in tokens:
            output_ids.append(
                torch.tensor(list(filter((self.src_pad_idx).__ne__, w.tolist()))[1:-1])
            )

        output_indices = []
        for i in range(batch.shape[0]):
            tensor = torch.tensor(find_subtensor(output_ids[i], batch[i]))
            if tensor is None:
                try:
                    tensor = torch.tensor(find_subtensor(output_ids[i][:-1], batch[i]))
                except IndexError:
                    tensor = torch.tensor([1, lens[i].item() - 1])
                if tensor is None:
                    tensor = torch.tensor([1, lens[i].item() - 1])
            output_indices.append(tensor)
        return torch.stack(output_indices).unsqueeze(1)


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        embeddings=None,
        hidden=None,
        encoder_hidden=None,
        num_layers=1,
        teacher_forcing_p=0.0,
        attention=None,
        dropout=None,
    ):

        super(LSTM_Decoder, self).__init__()
        # TODO Use Fairseq attention
        self.embeddings = embeddings
        self.hidden = hidden
        self.embedding_dropout = nn.Dropout(dropout.input)

        self.hidden_state_dropout = nn.Dropout(dropout.output)

        self.encoder_hidden_proj = (
            nn.Linear(encoder_hidden, hidden)
            if (encoder_hidden != hidden or encoder_hidden is None)
            else lambda x: x
        )
        self.lstm_decoder = nn.ModuleList()
        self.num_layers = num_layers

        # self.lstm_decoder = nn.ModuleList([
        #    nn.LSTMCell(self.embeddings.embedding_dim, self.hidden)
        #    for i in range(self.num_layers) if i==0 else
        #    nn.LSTMCell(self.hidden,self.hidden)
        # ])

        for i in range(self.num_layers):
            if i == 0:
                self.lstm_decoder.append(
                    nn.LSTMCell(self.embeddings.embedding_dim, self.hidden)
                )
            else:
                self.lstm_decoder.append(nn.LSTMCell(self.hidden, self.hidden))
        self.proj_layer = nn.Linear(self.hidden, self.embeddings.num_embeddings)
        self.teacher_forcing_p = 0.1
        self.state = {
            "hidden": [None] * self.num_layers,
            "cell": [None] * self.num_layers,
        }
        if attention is not None:
            self.enc_hidden_att_komp = nn.Linear(encoder_hidden, hidden)
            self.attention = AttentionLayer(
                self.hidden, attn_type="general", attn_func="softmax"
            )

    def forward(
        self,
        input_ids,
        initial_state=None,
        context_batch_mask=None,
        encoder_hidden_states=None,
    ):
        self.init_state(initial_state)

        all_logits = []
        all_preds = []

        for i in range(input_ids.shape[1] - 1):
            p = random.random()

            input_id = (
                all_preds[-1]
                if (p <= self.teacher_forcing_p and all_preds and self.training)
                else input_ids[:, i]
            )
            logits, prev_hiddens, prev_cells, attn = self.generate(
                input_id, prev_hiddens, prev_cells
            )
            all_logits.append(logits)

            pred = torch.argmax(F.softmax(logits, 1), 1)
            all_preds.append(pred)

        # batch_size, seq_len
        all_logits = torch.stack(all_logits, 1)
        all_preds = torch.stack(all_preds, 1)
        return all_logits, all_preds

    def generate(self, input_id, prev_hiddens, prev_cells):
        input = self.embedding_dropout(self.embeddings(input_id))
        for i, rnn in enumerate(self.lstm_decoder):
            # recurrent cell
            hidden, cell = rnn(input, (self.state["hidden"][i], self.state["cell"][i]))

            # hidden state becomes the input to the next layer
            input = self.hidden_state_dropout(hidden)

            # save state for next time step
            self.state["hidden"][i] = hidden
            self.state["cell"][i] = cell

        logits = self.proj_layer(hidden)
        return logits, None

    def init_state(self, initial_state):
        if initial_state is None:
            self.state["hidden"] = [None] * self.num_layers
            self.state["cell"] = [None] * self.num_layers
        else:
            self.state["hidden"] = [
                self.encoder_hidden_proj(initial_state)
            ] * self.num_layers
            self.state["cell"] = [
                self.encoder_hidden_proj(initial_state)
            ] * self.num_layers

    def map_state(self, fn):
        self.state["hidden"] = [fn(h, 0) for h in self.state["hidden"]]
        self.state["cell"] = [fn(c, 0) for c in self.state["cell"]]
        # self.state["input_feed"] = fn(self.state["input_feed"], 1)
