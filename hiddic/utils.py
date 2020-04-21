from allennlp.modules.elmo import batch_to_ids
import torch
import numpy as np
from sacrebleu import corpus_bleu as bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import tqdm
import os

cc = SmoothingFunction()


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


def find_subtensor(sl, l, device="cuda"):
    sl = sl.to(device)
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if False not in (l[ind : ind + sll] == sl):
            return torch.tensor([ind, ind + sll - 1]).to(device)


def batch_bleu(ref, hyp, reduction="average"):
    """
    Calculates bleu score for a batch,

    """

    def check_bleu(r, h, smooth=None, auto_reweigh=False):
        """
        Uses equivalent of MOSES's `multi-bleu.pl` script using the sacrebleu library (https://www.aclweb.org/anthology/W18-6319)

        If ngrams < 4, uses Smoothing Function 4, as shown in A "Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU" (Chen and Cherry, 2014) doi:10.3115/v1/W14-3346 
        """
        if smooth is not None:
            try:
                score = (
                    sentence_bleu(
                        [word_tokenize(r)],
                        word_tokenize(h),
                        auto_reweigh=auto_reweigh,
                        smoothing_function=smooth,
                    )
                    * 100
                )
            except ZeroDivisionError:
                score = 0.0

            return score
        else:
            score = bleu([h], [[r]]).score

        if score >= 1e-8:
            return score
        else:
            # Improvement to the NIST geometric sequence smoothing shown in doi:10.3115/v1/W14-3346; also follows auto_reweigh procedure explained here: https://github.com/nltk/nltk/issues/1554
            return check_bleu(r, h, smooth=cc.method4, auto_reweigh=True)

    total_score = 0
    assert len(ref) == len(hyp)

    for _ref, _hyp in tqdm.tqdm(
        zip(ref, hyp), desc="Calculating BLEU: ", total=len(ref)
    ):
        total_score += check_bleu(_ref, _hyp)

    if reduction == "sum":
        return total_score
    elif reduction == "average":
        return total_score / len(ref)
    else:
        raise NotImplementedError(
            f"{reduction} not in supported reductions: ['sum','average']"
        )


def mkdir(path):
    try:
        os.mkdir(path)
    except OSError as error:
        pass
    return path
