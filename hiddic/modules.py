import transformers
from allennlp.modules.elmo import Elmo
from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn

__ELMO_OPTIONS__ = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
__ELMO_WEIGHTS__ = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


class ELMo_wrapper(nn.Module):
    def __init__(self):
        super(ELMo_wrapper, self).__init__()

        self.model = Elmo(options_file, weight_file, 2, dropout=0)

    def forward(self, input_ids, lens):
        embeddings = self.model(input_ids)
        avg_embeddings = self._avg_pool(embeddings, lens)
        return (embeddings, avg_embeddings, embeddings)

    def _avg_pool(self, embeddings, lens):
        _sum = torch.sum(embeddings, 1)
        assert lens.shape[0] == embeddings.shape[0]
        assert lens.shape[1] == 1
        return _sum / lens


def get_pretrained_transformer(path):
    config = AutoConfig.from_pretrained(path, output_hidden_states=True)
    return AutoModel.from_pretrained(path, config=config)
