import torch
import inspect
import warnings
import torch.nn as nn
from datasets import Dataset
from dataclasses import dataclass
from dataclasses import FrozenInstanceError, replace
from accelerate import PartialState
from typing import Any, Callable, Optional, Union

from trl import RewardTrainer, RewardConfig
from trl.data_utils import maybe_apply_chat_template

from transformers import (
    DataCollator,
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)

from transformers.utils import is_peft_available
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from trl.trainer.utils import (
    disable_dropout_in_model,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


def _tokenize(examples, tokenizer):
    """
    Tokenize chosen and rejected responses for reward modeling.
    Compatible with newer TRL versions.
    Handles both single examples and batches.
    """
    # Handle batched input (when called with batched=True in dataset.map)
    tokenized_chosen = tokenizer(
        examples["chosen"], 
        truncation=True,
        padding=False,  # Don't pad yet, will be done by collator
        max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 1024
    )
    tokenized_rejected = tokenizer(
        examples["rejected"], 
        truncation=True,
        padding=False,  # Don't pad yet, will be done by collator
        max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 1024
    )
    
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }


def compute_accuracy(eval_pred):
    """
    Compute accuracy for reward model predictions.
    Compatible with newer TRL versions.
    
    Args:
        eval_pred: EvalPrediction with predictions and labels
        
    Returns:
        dict: Dictionary with accuracy metric
    """
    # Handle both tuple and EvalPrediction formats
    if isinstance(eval_pred, tuple):
        predictions = eval_pred[0]
    else:
        predictions = eval_pred.predictions
    
    # If predictions is a tuple (rewards_chosen, rewards_rejected), extract them
    if isinstance(predictions, tuple):
        rewards_chosen, rewards_rejected = predictions
        import numpy as np
        accuracy = (rewards_chosen > rewards_rejected).astype(float).mean()
    else:
        # predictions contains [rewards_chosen, rewards_rejected] stacked
        # Model is correct if reward_chosen > reward_rejected
        import numpy as np
        if len(predictions.shape) == 1:
            # Flat array, reshape to [batch, 2]
            predictions = predictions.reshape(-1, 2)
        accuracy = (predictions[:, 0] > predictions[:, 1]).astype(float).mean()
    
    return {"accuracy": accuracy}


class SharedRepRMTrainer(RewardTrainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[dict] = None,
        use_tanh: bool = False,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
                Processing class used to process the data. If provided, will be used to automatically process the inputs
                for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
                reuse the fine-tuned model.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`list[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            peft_config (`dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`.",
                            UserWarning,
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

                model = get_peft_model(model, peft_config)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"Trainable parameter: {name}")

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(model)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "A processing_class must be specified when using the default data collator"
                )

            # Use our custom SharedRepRewardDataCollatorWithPadding
            data_collator = SharedRepRewardDataCollatorWithPadding(
                tokenizer=processing_class,
                padding=True,
                return_tensors="pt",
            )

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using custom data collator, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Reward, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "input_ids_chosen" and "input_ids_rejected". As a result,
        # the trainer issues the warning: "Could not estimate the number of tokens of the input, floating-point
        # operations will not be computed." To suppress this warning, we set the "estimate_tokens" key in the model's
        # "warnings_issued" dictionary to True. This acts as a flag to indicate that the warning has already been
        # issued.
        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        # Let the parent RewardTrainer handle tokenization via its _prepare_dataset method
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Set flag to indicate that the model needs tanh
        self.use_tanh = use_tanh


    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        # Get group_ids if available, otherwise create default group (group 0)
        group_ids = inputs.get("group_id")
        if group_ids is None:
            # If no group_id, assume all samples are from group 0
            batch_size = inputs["input_ids_chosen"].shape[0]
            group_ids = torch.zeros(batch_size, dtype=torch.long, device=inputs["input_ids_chosen"].device)
        
        rewards_chosen = model(
            group_ids=group_ids,
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        rewards_rejected = model(
            group_ids=group_ids,
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]

        if self.use_tanh:
            rewards_chosen = torch.tanh(rewards_chosen)
            rewards_rejected = torch.tanh(rewards_rejected)

        # calculate loss, optionally modulate with margin
        if "margin" in inputs:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        else:
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if self.args.center_rewards_coefficient is not None:
            loss += self.args.center_rewards_coefficient * torch.mean((rewards_chosen + rewards_rejected) ** 2)

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss

@dataclass
class SharedRepRewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        pad_to_multiple_of (`int` or `None`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        group_id = []
        # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        has_group_id = "group_id" in features[0]
        for feature in features: # loop over batches
            # Handle both old and new TRL column naming conventions
            # New TRL (0.27+): chosen_input_ids, rejected_input_ids
            # Old TRL: input_ids_chosen, input_ids_rejected
            if "chosen_input_ids" in feature:
                # New TRL format
                chosen_ids = feature["chosen_input_ids"]
                rejected_ids = feature["rejected_input_ids"]
                # attention_mask may not exist, create it from input_ids
                chosen_mask = feature.get("chosen_attention_mask", [1] * len(chosen_ids))
                rejected_mask = feature.get("rejected_attention_mask", [1] * len(rejected_ids))
            elif "input_ids_chosen" in feature:
                # Old TRL format
                chosen_ids = feature["input_ids_chosen"]
                rejected_ids = feature["input_ids_rejected"]
                chosen_mask = feature.get("attention_mask_chosen", [1] * len(chosen_ids))
                rejected_mask = feature.get("attention_mask_rejected", [1] * len(rejected_ids))
            else:
                raise ValueError(
                    f"The features should include either `chosen_input_ids`/`rejected_input_ids` or `input_ids_chosen`/`input_ids_rejected`. Got keys: {list(feature.keys())}"
                )

            features_chosen.append(
                {
                    "input_ids": chosen_ids,
                    "attention_mask": chosen_mask,
                }
            )
            features_rejected.append(
                {
                    "input_ids": rejected_ids,
                    "attention_mask": rejected_mask,
                }
            )
            if has_margin:
                margin.append(feature["margin"])
            if has_group_id:
                group_id.append(feature["group_id"])
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        if has_margin:
            margin = torch.tensor(margin, dtype=torch.float)
            batch["margin"] = margin
        if has_group_id:
            group_id = torch.tensor(group_id, dtype=torch.long)
            batch["group_id"] = group_id
        return batch

