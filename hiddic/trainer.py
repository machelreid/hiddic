import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from beam import BeamSearch
from tensorboardX import SummaryWriter
from utils import batch_bleu
import os
import logging


def build_trainer(
    args, model, data_fields, metric_should_decrease=True,
):
    if phase not in ["train"]:
        raise NotImplementedError(
            "PRETRAIN and TUNE modes to be implemented, only TRAIN mode is supported"
        )

    trainer = Trainer(
        model,
        patience=args.patience,
        # val_interval=100,
        val_metric="loss",
        serialization_dir=None,
        # max_vals=50,
        device="cuda",
        clip_grad_norm_val=args.clip,
        initial_lr=args.lr,
        min_lr=args.min_lr,
        lr_patience=None,
        keep_all_checkpoints=args.keep_all_checkpoints,
        val_data_limit=args.val_data_limit,
        max_epochs=args.val_data_limit,
        training_data_fraction=args.training_data_fraction,
        beam_size=args.beam_size,
        min_length=args.min_length,
        max_length=args.max_lengh,
        n_best=1,
        ratio=None,
        datapath=args.datapath,
        data_args=data_fields,
        dataset=args.dataset,
        lr_scheduling_metric=args.lr_scheduling_metric,
        metric_decreases=args.metric_decreases,
    )

    return trainer


log = logging.getLogger()


class Trainer(obejct):
    def __init__(
        self,
        model,
        patience=4,
        # val_interval=100,
        val_metric="loss",
        serialization_dir=None,
        device="cuda",
        clip_grad_norm_val=None,
        initial_lr=None,
        lr_decay=None,
        min_lr=None,
        lr_patience=None,
        keep_all_checkpoints=False,
        val_data_limit=None,
        max_epochs=-1,
        training_data_fraction=0,
        beam_size=1,
        min_length=3,
        max_length=512,
        n_best=1,
        ratio=None,
        datapath=None,
        data_args=None,
        dataset=None,
        lr_scheduling_metric=None,
        metric_decreases=None,
    ):
        """
        The training coordinator. Unusually complicated to handle MTL with tasks of
        diverse sizes.
        Parameters
        ----------
        model : ``Model``, required.
            An PyTorch model to be optimized. Can  be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        patience , optional (default=2)
            Number of validations to be patient before early stopping.
        val_metric , optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model after each validation. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        serialization_dir , optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device , optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        keep_all_checkpoints : If set, keep checkpoints from every validation. Otherwise, keep only
            best and (if different) most recent.
        val_data_limit: During training, use only the first N examples from the validation set.
            Set to -1 to use all.
        training_data_fraction: If set to a float between 0 and 1, load only the specified
            percentage of examples. Hashing is used to ensure that the same examples are loaded
            each epoch.
        """
        self._model = model

        self._patience = patience
        self._val_interval = val_interval
        self._serialization_dir = serialization_dir
        self._device = device
        self._clip_grad_norm_val = clip_gram_norm_val
        self._lr_decay = lr_decay
        self._min_lr = min_lr
        self._keep_all_checkpoints = keep_all_checkpoints
        self._val_data_limit = val_data_limit
        self._max_epochs = max_epochs
        self._training_data_fraction = training_data_fraction
        self._datapath = datapath
        self._data_args = data_args
        self._dataset = dataset
        self._initial_lr = initial_lr
        self._lr_scheduling_metric = lr_scheduling_metric
        self._metric_decreases = metric_decreases

        self._metric_infos = {}

        self._trainable_params = filter(
            lambda p: p.requires_grad, self._model.parameters()
        )
        self._optimizer = optim.Adam(self._trainable_params, lr=self._initial_lr)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min" if self._metric_decreases else "max",
            patience=self._lr_patience,
            verbose=True,
            min_lr=self._min_lr if self._min_lr else 0,
        )

        if beam_size == 1:
            log.warining(
                "WARNING: Beam size is 1, note that this is equivalent to greedy search"
            )
        self._beam_size = beam_size
        self._n_best = n_best
        self._min_length = min_length
        self._max_length = max_length
        self._ratio = ratio

        self._datamaker = data.DataMaker(self._data_args, self._datapath)
        self._datamaker.build_data(self._dataset)

        self._epoch_steps = 0
        self._train_counter = 0
        self._validation_counter = 0

        self._tgt_pad_idx = self._datamaker.vocab.defintion.stoi["<pad>"]
        self._tgt_bos_idx = self._datamaker.vocab.defintion.stoi["<sos>"]
        self._tgt_eos_idx = self._datamaker.vocab.defintion.stoi["<eos>"]
        self._tgt_unk_idx = self._datamaker.vocab.defintion.stoi["<unk>"]
        self._exclusion_idxs = {self._tgt_unk_idx, self._tgt_pad_idx, self._tgt_bos_idx}

        self._TB_dir = None
        if self._serialization_dir is not None:
            self._TB_dir = os.path.join(self._serialization_dir, "tensorboard")
            self._TB_train_log = SummaryWriter(os.path.join(self._TB_dir, "train"))
            self._TB_validation_log = SummaryWriter(os.path.join(self._TB_dir, "val"))

            self._validation_log_dir = os.path.join(self._serialization_dir, "valid")
            self._train_log_dir = os.path.join(self._serialization_dir, "train")

    def _check_metric_history(
        self, metric_history, current_score, should_decrease=False
    ):
        """
        Given a the history of the performance on a metric
        and the current score, check if current score is
        best so far and if out of patience.
        """
        assert current_score in metric_history

        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == current_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        if should_decrease:
            index_of_last_improvement = metric_history.index(min(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (
                patience + 1
            )
        else:
            index_of_last_improvement = metric_history.index(max(metric_history))
            out_of_patience = index_of_last_improvement <= len(metric_history) - (
                patience + 1
            )

        return best_so_far, out_of_patience

    def _update_metric_history(
        self, val_pass, metric, current_value, metric_infos, metric_decreases,
    ):
        """
        This function updates metric history with the best validation score so far.
        Parameters
        ---------
        val_pass: int.
        all_val_metrics: dict with performance on current validation pass.
        metric: str, name of metric
        task_name: str, name of task
        metric_infos: dict storing information about the various metrics
        metric_decreases: bool, marker to show if we should increase or
        decrease validation metric.
        should_save: bool, for checkpointing
        new_best: bool, indicator of whether the previous best preformance score was exceeded
        Returns
        ________
        metric_infos: dict storing information about the various metrics
        this_val_metric: dict, metric information for this validation pass, used for optimization
            scheduler
        should_save: bool
        new_best: bool
        """

        metric_exists = self._metric_infos.get(metric)
        if not metric_exists:
            self._metric_infos[metric] = {}
        metric_history = self._metric_infos[metric].get(["hist"])
        if metric_history is None:
            self._metric_infos[metric]["hist"] = []
        metric_history.append(current_value)
        is_best_so_far, out_of_patience = self._check_history(
            metric_history, current_value, metric_decreases
        )
        if is_best_so_far:
            log.info("Best result seen so far for %s.", task_name)
            self._metric_infos[metric]["best"] = (val_pass, all_val_metrics)
            should_save = True
        if out_of_patience:
            self._metric_infos[metric]["stopped"] = True
        return is_best_so_far, out_of_patience

    def _calculate_validation_performance(
        self,
        task,
        task_infos,
        tasks,
        batch_size,
        all_val_metrics,
        n_examples_overall,
        print_output=True,
    ):
        """
        Builds validation generator, evaluates on each task and produces validation metrics.
        Parameters
        ----------
        task: current task to get validation performance of
        task_infos: Instance of information about the task (see _setup_training for definition)
        tasks: list of task objects to train on
        batch_size: int, batch size to use for the tasks
        all_val_metrics: dictionary. storing the validation performance
        n_examples_overall: int, current number of examples the model is validated on
        print_output: bool, prints one example per validation
        Returns
        -------
        n_examples_overall: int, current number of examples
        task_infos: updated Instance with reset training progress
        all_val_metrics: dictinary updated with micro and macro average validation performance
        """
        TODO = TODO

        # # Get scheduler, and update using macro score
        # # micro has no scheduler updates
        # if task_name == "macro" and isinstance(
        #     self._scheduler.lr_scheduler, ReduceLROnPlateau
        # ):

        return all_val_metrics, should_save, new_best

    def _train(self, batch_size):

        self._epoch_steps += 1
        if self._epoch_steps > self._max_epochs:
            log.info(f"Max Epoch Steps {self._max_epochs} reached. Training Stopped.")
            return
        elif self._patience_exceeded:
            log.info(
                f"Patience has already been exceeded for every metric. In other words, I've become IMPATIENT. Training Stopped."
            )
            return

        train_iterator = self._datamaker.get_iterator(
            "train", batch_size, device=self._device
        )

        generations = []
        targets = []
        sources = []
        for batch in train_iterator:
            self._train_counter += 1
            self._model.zero_grad()

            example, example_lens = batch.example
            definition, definition_lens = batch.definition
            word, word_lens = batch.word

            model_out = self._forward(
                "train",
                input_ids=example,
                seq_lens=example_lens,
                span_ids=word,
                target=definition,
            )
            generations.extend(
                self.datamaker.decode(model_out.predictions, "definition", batch=True)
            )
            targets.extend(self.datamaker.decode(definition, "definition", batch=True))
            sources.extend(self.datamaker.decode(example, "example", batch=True))

            torch.nn.utils.clip_grad_norm_(
                self._trainable_params, self._clip_grad_norm_val
            )
            model_out.loss.backward()
            self._optimizer.step()

        return DotMap({"src": sources, "tgt": targets, "gen": generations})

    def _validate(self, batch_size):

        valid_iterator = self._datamaker.get_iterator(
            "valid", batch_size, device=self._device
        )

        generations = []
        targets = []
        sources = []
        logits_for_ppl_calc = []
        tgt_idxs_for_ppl_calc = []
        decode_strategy = BeamSearch(
            self.beam_size,
            batch_size,
            pad=self._tgt_pad_idx,
            bos=self._tgt_bos_idx,
            eos=self._tgt_eos_idx,
            n_best=1 if self._n_best is None else self._n_best,
            global_scorer=self._model.global_scorer,
            min_length=self._min_length,
            max_length=self._max_length,
            return_attention=False,
            block_ngram_repeat=3,
            exclusion_tokens=self._exclusion_idxs,
            stepwise_penalty=None,
            ratio=self.ratio,
        )

        for i, batch in enumerate(valid_iterator):
            self._validation_counter += 1
            self._model.zero_grad()
            self._model.eval()

            example, example_lens = batch.example
            definition, definition_lens = batch.definition
            word, word_lens = batch.word

            current_batch_size = word.shape[0]
            model_out = self._forward(
                "valid",
                input_ids=example,
                seq_lens=example_lens,
                span_ids=word,
                target=definition,
                tgt_lens=definition_lens,
                decode_strategy=decode_strategy,
            )
            generations.extend(
                self.datamaker.decode(model_out.predictions, "definition", batch=True)
            )
            targets.extend(self.datamaker.decode(definition, "definition", batch=True))
            sources.extend(self.datamaker.decode(example, "example", batch=True))

            logits_for_ppl_calc.append(model_out.logits)
            tgt_idxs_for_ppl_calc.append(definition[:, 1:].contiguous().view(-1))
            self._TB_validation_log.add_scalar(
                "batch_perplexity", model_out.ppl.item(), self._validation_counter
            )

            current_bleu = batch_bleu(
                targets[-current_batch_size:],
                generations[-current_batch_size:],
                reduction="average",
            )
            self._TB_validation_log.add_scalar(
                "batch_BLEU", current_bleu, self._validation_counter
            )

            if self._max_val_data:
                if i * batch_size > self._max_val_data:
                    break
        bleu = batch_bleu(targets, generations, reduction="average")
        self._TB_validation_log.add_scalar(
            "BLEU", model_out.ppl.item(), self._epoch_steps
        )

        ppl = (
            F.cross_entropy(
                torch.cat(logits_for_ppl_calc, 0),
                torch.cat(tgt_idxs_for_ppl_calc),
                ignore_index=self._model.embeddings.tgt.padding_idx,
            )
            .exp()
            .item()
        )

        self._TB_validation_log.add_scalar("Perplexity", ppl, self._epoch_steps)

        metric_dict = {"bleu": bleu, "perplexity": ppl}

        bleu_best, bleu_patience = self._update_metric_history(
            self._epoch_steps, "bleu", bleu, self._metric_infos, metric_decreases=False,
        )

        ppl_best, ppl_patience = self._update_metric_history(
            self._epoch_steps,
            "perplexity",
            ppl,
            self._metric_infos,
            metric_decreases=True,
        )

        if self._keep_all_checkpoints:
            torch.save(
                self._model.state_dict(),
                os.path.join(
                    self._serialization_dir, "model", f"iter_{self._epoch_steps}.pth"
                ),
            )
        if bleu_best:
            torch.save(
                self._model.state_dict(),
                os.path.join(
                    self._serialization_dir,
                    "model",
                    f"BLEU_BEST_{self._epoch_steps}.pth",
                ),
            )
        if ppl_best:
            torch.save(
                self._model.state_dict(),
                os.path.join(
                    self._serialization_dir,
                    "model",
                    f"PPL_BEST_{self._epoch_steps}.pth",
                ),
            )
        if not bleu_patience and not ppl_patience:
            log.info("Ran out of patience for both BLEU and perplexity")

        with open(
            os.path.join(self._validation_log_dir, f"iter_{self._epoch_steps}.json"),
            "w",
        ) as f:
            f.write(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "src": sources[i],
                                "tgt": targets[i],
                                "gen": generations[i],
                            }
                        )
                        for i in range(len(generations))
                    ]
                )
            )

        lr_scheduling_metric = metric_dict.get(self._lr_scheduling_metric)
        if lr_scheduling_metric is None:
            lr_scheduling_metric = ppl
            log.warning(
                f"WARNING: {self._lr_scheduling_metric} not found as a metric for validation performance calculation. REVERTING TO PERPLEXITY INSTEAD"
            )

        log.info(f"Updating LR scheduler with {self._lr_scheduling_metric}:")

        self._scheduler.step(lr_scheduling_metric)
        log.info(
            "\tBest result seen so far for %s: %.3f", metric, self._scheduler.best,
        )
        log.info(
            "\t# validation passes without improvement: %d",
            self._scheduler.num_bad_epochs,
        )

        self._scheduler.step(model_out.loss)

        return DotMap({"src": sources, "tgt": targets, "gen": generations})

    def _reset_steps(self):
        self._epoch_steps = 0
        self._validation_counter = 0
        self._train_counter = 0

    def _forward(self, phase: str = "train", **batch):

        if phase not in ["train", "valid", "test"]:
            raise NotImplementedError(f"{phase} must be in ['train','test','valid']")
        if phase == "train":
            return self._model(
                batch["input_ids"],
                batch["seq_lens"],
                batch["span_ids"],
                batch["target"],
            )
        elif phase in ["valid", "test"]:
            return self._model._validate(
                batch["input_ids"],
                batch["seq_lens"],
                batch["span_ids"],
                batch["target"],
                batch["tgt_lens"],
                batch["decode_strategy"],
            )
        else:
            raise NotImplementedError
