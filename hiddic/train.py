import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from beam import BeamSearch
from tensorboardX import SummaryWriter
from utils import batch_bleu


def build_trainer(
    args,
    cuda_device,
    task_names,
    model,
    run_dir,
    metric_should_decrease=True,
    train_type="SamplingMultiTaskTrainer",
    phase="pretrain",
):
    if phase not in ["pretrain", "train", "tune"]:
        raise NotImplementedError


class Trainer(obejct):
    def __init__(
        self,
        model,
        patience=3,
        # val_interval=100,
        val_metric="loss",
        serialization_dir=None,
        max_vals=50,
        cuda_device=-1,
        grad_norm=None,
        grad_clipping=None,
        lr_decay=None,
        min_lr=None,
        lr_patience=None,
        keep_all_checkpoints=False,
        val_data_limit=500,
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
        self._max_vals = max_vals
        self._val_interval = val_interval
        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._lr_decay = lr_decay
        self._min_lr = min_lr
        self._keep_all_checkpoints = keep_all_checkpoints
        self._val_data_limit = val_data_limit
        self._max_epochs = max_epochs
        self._dec_val_scale = dec_val_scale
        self._training_data_fraction = training_data_fraction
        self._datapath = datapath
        self._data_args = data_args
        self._dataset = dataset
        self._task_infos = None
        self._metric_infos = None

        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            patience=self._lr_patience,
            verbose=True,
            min_lr=self._min_lr if self._min_lr else 0,
        )

        self._beam_size = beam_size
        self.n_best = n_best
        self.min_length = 3
        self.max_length = 10
        self.ratio = ratio

        self._datamaker = data.DataMaker(self._data_args, self._datapath)
        self._datamaker.build_data(self._dataset)
        self._accumulation_steps = 0

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
        self,
        val_pass,
        all_val_metrics,
        metric,
        task_name,
        metric_infos,
        metric_decreases,
        should_save,
        new_best,
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
        this_val_metric = all_val_metrics[metric]
        metric_history = metric_infos[metric]["hist"]
        metric_history.append(this_val_metric)
        is_best_so_far, out_of_patience = self._check_history(
            metric_history, this_val_metric, metric_decreases
        )
        if is_best_so_far:
            log.info("Best result seen so far for %s.", task_name)
            metric_infos[metric]["best"] = (val_pass, all_val_metrics)
            should_save = True
            if task_name == "macro":
                new_best = True
        if out_of_patience:
            metric_infos[metric]["stopped"] = True
            # Commented out the below line as more confusing than helpful. May make sense to
            # restore if we wind up using more complex stopping strategies.
            # log.info("Out of early stopping patience. Stopped tracking %s.", task_name)
        return metric_infos, this_val_metric, should_save, new_best

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

    def _validate(self, val_pass, tasks, batch_size, periodic_save=True):

        """
        Validate on all tasks and return the results and whether to save this validation
        pass or not.
        Parameters
        ----------
        val_pass: int
        tasks: list of task objects to train on
        batch_size: int, the batch size to use for the tasks.periodic_save
        periodic_save: bool, value of whether or not to save model and progress periodically
        Returns
        __________
        all_val_metrics: dictinary updated with micro and macro average validation performance
        should_save: bool, determines whether to save a checkpoint
        new_best: bool, whether or not the macro performance increased
        """
        task_infos, metric_infos = self._task_infos, self._metric_infos
        self._model.eval()
        all_val_metrics = {("%s_loss" % task.name): 0.0 for task in tasks}
        all_val_metrics["macro_avg"] = 0.0
        all_val_metrics["micro_avg"] = 0.0
        n_examples_overall = 0.0

        # Get validation numbers for each task
        for task in tasks:
            (
                n_examples_overall,
                task_infos,
                all_val_metrics,
            ) = self._calculate_validation_performance(
                task, task_infos, tasks, batch_size, all_val_metrics, n_examples_overall
            )
        # scale the micro avg contributions w/ total size of validation set.
        if "micro_avg" in all_val_metrics:
            all_val_metrics["micro_avg"] /= n_examples_overall
        # Track per task patience
        should_save = periodic_save  # whether to save this validation pass or not.
        # Currently we save every validation in the main training runs.
        new_best = False  # whether this validation pass is a new best

        # update metric infos
        for task in tasks + ["micro", "macro"]:
            if task in ["micro", "macro"]:
                metric = "%s_avg" % task
                metric_decreases = (
                    tasks[0].val_metric_decreases if len(tasks) == 1 else False
                )
                task_name = task
            else:
                metric = task.val_metric
                metric_decreases = task.val_metric_decreases
                task_name = task.name
            if metric_infos[metric]["stopped"]:
                continue
            (
                metric_infos,
                this_val_metric,
                should_save,
                new_best,
            ) = self._update_metric_history(
                val_pass,
                all_val_metrics,
                metric,
                task_name,
                metric_infos,
                metric_decreases,
                should_save,
                new_best,
            )

            # Get scheduler, and update using macro score
            # micro has no scheduler updates
            if task_name == "macro" and isinstance(
                self._scheduler.lr_scheduler, ReduceLROnPlateau
            ):
                log.info("Updating LR scheduler:")
                self._scheduler.step(this_val_metric, val_pass)
                log.info(
                    "\tBest result seen so far for %s: %.3f",
                    metric,
                    self._scheduler.lr_scheduler.best,
                )
                log.info(
                    "\t# validation passes without improvement: %d",
                    self._scheduler.lr_scheduler.num_bad_epochs,
                )

        return all_val_metrics, should_save, new_best

    def _train(self, batch_size):

        train_iterator = self._datamaker.get_iterator("train", batch_size)

        generations = []
        targets = []
        sources = []
        for batch in train_iterator:
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

            model_out.loss.backward()
            self._optimizer.step()

        return DotMap({"src": sources, "tgt": targets, "gen": generations})

    def _validate(self, batch_size):

        valid_iterator = self._datamaker.get_iterator("valid", batch_size)

        generations = []
        targets = []
        sources = []
        ppl = 0

        decode_strategy = BeamSearch(
            self.beam_size,
            batch_size,
            pad=self._tgt_pad_idx,
            bos=self._tgt_bos_idx,
            eos=self._tgt_eos_idx,
            n_best=1 if self.n_best is None else self.n_best,
            global_scorer=self._model.global_scorer,
            min_length=self.min_length,
            max_length=self.max_length,
            return_attention=False,
            block_ngram_repeat=3,
            exclusion_tokens=self._exclusion_idxs,
            stepwise_penalty=None,
            ratio=self.ratio,
        )

        for i, batch in enumerate(valid_iterator):
            self._model.zero_grad()
            self._model.eval()

            example, example_lens = batch.example
            definition, definition_lens = batch.definition
            word, word_lens = batch.word

            model_out = self._forward(
                "valid",
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

            if self._max_val_iters:
                if i * batch_size > self._max_val_data:
                    break
        bleu = batch_bleu(targets, generations, reduction="average")

        bleu_best, bleu_patience = self._update_metric_history(
            self.val_pass,
            all_val_metrics,
            bleu,
            self.metric_infos,
            metric_decreases,
            should_save,
            new_best,
        )

        ppl_best, ppl_patience = self._update_metric_history(
            self.val_pass,
            all_val_metrics,
            metric,
            self.metric_infos,
            metric_decreases,
            should_save,
            new_best,
        )

        if self.save_every_iter:
            torch.save(
                self._model.state_dict(),
                os.path.join(
                    self._serialization_dir, "model", f"iter_{self.val_pass}.pth"
                ),
            )
        if bleu_best:
            torch.save(
                self._model.state_dict(),
                os.path.join(self._serialization_dir, "model", "BLEU_BEST.pth"),
            )
        if ppl_best:
            torch.save(
                self._model.state_dict(),
                os.path.join(self._serialization_dir, "model", "PPL_BEST.pth"),
            )

        self._scheduler.step(model_out.loss)

        return DotMap({"src": sources, "tgt": targets, "gen": generations})

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
            return self._model(
                batch["input_ids"],
                batch["seq_lens"],
                batch["span_ids"],
                self.datamaker.vocab,
            )
        else:
            raise NotImplementedError
