from copy import deepcopy as cp
import torchtext
from torchtext.vocab import Vocab
from torchtext import data
from nltk.tokenize import word_tokenize
from dotmap import DotMap
from utils import removeDuplicates, elmo_batch_to_ids
from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer, AutoTokenizer
from collections import defaultdict, Counter
import os
import torch


bos_token = "<sos>"
eos_token = "<eos>"
pad_token = "<pad>"
unk_token = "<unk>"
cls_token = "<cls>"
mask_token = "<mask>"
special_tokens = {
    "cls_token": cls_token,
    "unk_token": unk_token,
    "pad_token": pad_token,
    "bos_token": bos_token,
    "eos_token": eos_token,
    "mask_token": mask_token,
}
special_tokens_list = [cls_token]


class Field(data.Field):
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer")
        super().__init__(**kwargs)


def get_huggingface_tokenizer(tokenizer, pretrained_model: str):

    if pretrained_model is not None:
        tokenizer = tokenizer.from_pretrained(pretrained_model)
        return tokenizer


class DataMaker(object):
    def __init__(self, fields=None, dataset_path: str = None):

        """
        Args:
            fields (dict || list of dicts): Configurations for each field to read from the JSON. Example below:
                -------------
                {
                    "name": "src",     # field in question
                    "field": None,     # How it should be referred to by the dataloader
                    "tokenizer": BertTokenizer.from_pretrained('bert-base-uncased').encode, #Tokenizer function. If not speficied (i.e. `None`) defualts to nltk.tokenize.word_tokenize 
                    "sequential": True, #Sequential Data?
                    ""
                    "eos": True,        #Add eos token?
                    "sos": False,       #Add sos token?
                }
                -------------
            data_path (str): The main path for all the data
        """
        if isinstance(fields, dict):
            fields = [fields]
        self.defualt_fields = {
            "tokenizer": "NaN",
            "tokenize": None,
            "postprocess": None,
            "eos": False,
            "sos": False,
            "sequential": True,
            "use_vocab": True,
        }
        self.dataset_fields = [{**self.defualt_fields, **field} for field in fields]
        self.dataset_fields = [i for i in map(lambda field: DotMap(field), fields)]
        self.dataset_path = dataset_path

    def build_data(
        self,
        dataset_name=None,
        shared_vocab_fields=None,
        max_len=None,
        lowercase=False,
    ):
        _fields = {
            field.name: (
                field.field if field.field != None else field.name,
                Field(
                    sequential=field.sequential,
                    tokenize="spacy" if field.tokenize is None else field.tokenize,
                    lower=lowercase,
                    fix_length=max_len,
                    init_token=bos_token if (field.sos and field.sequential) else None,
                    eos_token=eos_token if (field.eos and field.sequential) else None,
                    include_lengths=field.include_lengths,
                    batch_first=True,
                    tokenizer=field.tokenizer,
                    use_vocab=field.use_vocab,
                    pad_token=field.pad_token,
                    postprocessing=field.postprocess,
                ),
            )
            for field in self.dataset_fields
        }

        if shared_vocab_fields:
            if isinstance(shared_vocab_fields[0], list) == False:
                shared_vocab_fields = [shared_vocab_fields]
            self.share_vocab_fields(_fields, shared_vocab_fields)
        if shared_vocab_fields:
            assert (
                _fields[shared_vocab_fields[0][0]][1]
                == _fields[shared_vocab_fields[0][1]][1]
            )
        seperate_fields = removeDuplicates([_fields[i][1] for i in _fields])

        self.train, self.valid, self.test = data.TabularDataset.splits(
            path=os.path.join(self.dataset_path, dataset_name),
            train="_train.json",
            validation="_valid.json",
            test="_test.json",
            format="json",
            fields=_fields,
        )

        for f in seperate_fields:
            try:
                if f.use_vocab:
                    f.build_vocab(self.train, specials=special_tokens_list)

            except:
                pass

        self.vocab = {}

        for key in _fields:
            try:
                self.vocab[_fields[key][0]] = _fields[key][1].vocab
            except:
                self.vocab[_fields[key][0]] = _fields[key][1].tokenizer
        self.vocab = DotMap(self.vocab)

    def share_vocab_fields(self, field_data, shared_fields):
        """
        Function to share vocab fields in the case of a shared vocab
        Note that the first item in each list of shared fields and its correpsonding properties will be copied to each following field

        Args:
            field_data(dict): The field data object generated for every field in question. See func. `self.build_data`
            shared_fields(list): A 2d list of fields which should have the same vocab object. If it is 1d, it is unsqueezed to make it a 2d list.
        """
        if isinstance(shared_fields[0], list) == False:
            shared_fields = [shared_fields]

        for _set in shared_fields:
            relevant_field = field_data[_set[0]][1]
            for _field in _set[1:]:
                field_data[_field] = list(field_data[_field])
                field_data[_field][1] = relevant_field
                field_data[_field] = tuple(field_data[_field])

    def get_iterator(
        self, dataset: str, batch_size: int = None, shuffle=None, device=None
    ):
        """
        Args:
            dataset (str): representing the partition of the data to iterate over


        Returns:
            `torchtext.data.BucketIterator` object to iterate over the partition that was specified
        """
        if dataset not in ["test", "train", "valid"]:
            raise NotImplementedError
        if batch_size is None:
            raise NotImplementedError

        if dataset == "train":
            return data.BucketIterator(
                self.train, batch_size, shuffle=shuffle, device=device
            )
        elif dataset == "valid":
            return data.BucketIterator(
                self.valid, batch_size, shuffle=shuffle, device=device
            )
        elif dataset == "test":
            return data.BucketIterator(
                self.test, batch_size, shuffle=shuffle, device=device
            )

    def decode(self, input, vocab_partition, batch=False):
        """
        Args:
            input (list, torch.Tensor): Input of ids
            vocab_partition (str): partition of vocab to convert using
            batch (bool): 2d if True else 1d list
        """
        _vocab = getattr(self.vocab, vocab_partition)
        output_sentences = []
        if not batch:
            if isinstance(input, torch.Tensor):
                input = input.tolist()
            input = [input]
        if isinstance(input, torch.Tensor):
            input = [i.tolist() for i in input]
        try:
            for element in input:
                sentence = []
                for token_id in element:
                    token = _vocab.itos[token_id]
                    if token not in ["<pad>", "<sos>"]:
                        sentence.append(token)
                    else:
                        pass
                output_sentences.append(" ".join(sentence))
        except:
            for element in input:
                output_sentences.append(
                    _vocab.decode(element, skip_special_tokens=True)
                )
        if not batch:
            return output_sentences[0]
        else:
            return output_sentences


_bert_tokenizer = get_huggingface_tokenizer(BertTokenizer, "bert-base-uncased")
_roberta_tokenizer = get_huggingface_tokenizer(RobertaTokenizer, "roberta-base")


def get_dm_conf(_type, field_name):
    if _type is None:
        _type = "normal"
    if _type in [
        "elmo",
        "normal",
    ]:
        conf = cp(dm_conf[_type])
        conf["field"] = conf["name"] = field_name
        return conf
    else:
        try:
            conf = cp(dm_conf.transformer_base)
            tokenizer = AutoTokenizer.from_pretrained(_type)
            conf["tokenizer"] = tokenizer
            conf["tokenize"] = tokenizer.encode
            conf["pad_token"] = tokenizer.pad_token_id
            conf["field"] = conf["name"] = field_name
            return conf
        except Exception as e:
            raise NotImplementedError(
                "We don't have that preprocessing mechanism yet " + str(e)
            )


dm_conf = DotMap(
    {
        "transformer_base": {
            "name": None,
            "field": "src",
            "tokenize": None,  # tokenizer.encode
            "tokenizer": None,  # tokenizer
            "sequential": True,
            "eos": False,
            "sos": False,
            "use_vocab": False,
            "pad_token": None,
            "postprocess": None,
            "include_lengths": True,
        },
        "elmo": {
            "name": None,
            "field": "src",
            "tokenize": elmo_batch_to_ids,
            "tokenizer": None,
            "sequential": False,
            "eos": False,
            "sos": False,
            "use_vocab": False,
            "pad_token": 261,
            "postprocess": None,
            "include_lengths": True,
        },
        "normal": {
            "name": None,
            "field": "src",
            "tokenize": None,
            "tokenizer": None,
            "sequential": True,
            "eos": True,
            "sos": True,
            "use_vocab": True,
            "pad_token": "<pad>",
            "postprocess": None,
            "include_lengths": True,
        },
    }
)


def word_idx_getter(input, *args, **kwargs):
    return input[1:-1]
