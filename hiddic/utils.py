from allennlp.modules.elmo import batch_to_ids
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu


def get_output_attribute(out, attribute_name, cuda_device, reduction="sum"):
    """
    This function handles processing/reduction of output for both
    DataParallel or non-DataParallel situations.
    For the case of multiple GPUs, This function will
    sum all values for a certain output attribute in various batches
    together.
    Parameters
    ---------------------
    :param out: Dictionary, output of model during forward pass,
    :param attribute_name: str,
    :param cuda_device: list or int
    :param reduction: (string, optional) reduction to apply to the output. Default: 'sum'.
    """
    if isinstance(cuda_device, list):
        if reduction == "sum":
            return out[attribute_name].sum()
        elif reduction == "mean":
            return out[attribute_name].sum() / float(len(out[attribute_name]))
        else:
            raise ValueError("invalid reduction type argument")
    else:
        return out[attribute_name]


def removeDuplicates(listofElements):

    # Create an empty list to store unique elements
    uniqueList = []

    # Iterate over the original list and for each element
    # add it to uniqueList, if its not already there.
    for elem in listofElements:
        if elem not in uniqueList:
            uniqueList.append(elem)

    # Return the list of unique elements
    return uniqueList


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (
        torch.arange(0, max_len)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )


def elmo_batch_to_ids(_str):
    batch = [word_tokenize(_str)]
    return [i.tolist() for i in batch_to_ids(batch)[0]]


def find_subtensor(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if False not in (l[ind : ind + sll] == sl):
            return torch.tensor([ind, ind + sll - 1])


def batch_bleu(ref, hyp, reduction="sum"):
    """
    Calculates bleu score for a batch,

    """

    def check_bleu(r, h, ngrams=4):
        """
        If the BLEU Score is goes to zero (no ngram matches),  decreases ngrams by 1
        """
        temp = sentence_bleu([r], h, weights=[1 / ngrams] * ngrams)
        if temp >= 0.0001:
            return temp
        else:
            return check_bleu(r, h, ngrams=ngrams - 1)

    total_score = 0
    assert len(ref) == len(hyp)

    for _ref, _hyp in zip(ref, hyp):
        total_score += check_bleu(_ref, _hyp)

    if reduction == "sum":
        return total_score
    elif reduction == "average":
        return total_score / len(ref)
    else:
        raise NotImplementedError(
            "f{reduction} not in supported reductions: ['sum','average']"
        )
