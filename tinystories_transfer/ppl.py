import datasets
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

import evaluate
from evaluate import logging


_CITATION = """"""

_DESCRIPTION = """"""

_KWARGS_DESCRIPTION = """
Args:
    model (str): model used for calculating Perplexity
    predictions (list of str): input text, each separate text snippet
        is one list entry.
    batch_size (int): the batch size to run texts through the model. Defaults to 16.
    add_start_token (bool): whether to add the start token to the texts,
        so the perplexity can include the probability of the first word. Defaults to True.
    device (str): device to run on, defaults to 'cuda' when available
Returns:
    perplexity: dictionary containing the perplexity scores for the texts
        in the input list, as well as the mean perplexity. If one of the input texts is
        longer than the max input length of the model, then it is truncated to the
        max length for the perplexity computation.

"""


# @evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Perplexity(evaluate.Metric):
    def __init__(self, model, tokenizer, device=None):
        super(evaluate.Metric, self).__init__()
        # Set the device properly for all cases
        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                self.device = "cuda"
            else:
                # Add this else branch to handle the "cuda" and "cpu" cases
                self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.tokenizer = tokenizer


    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
        self, predictions, batch_size: int = 16, add_start_token: bool = False,  max_length=None
    ):
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1))
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            )

        ppls = []
        per_token_losses = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            unreduced_loss = loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch
            loss_batch = unreduced_loss.sum(1) / shift_attention_mask_batch.sum(1)
            perplexity_batch = torch.exp(loss_batch)

            ppls += perplexity_batch.tolist()

            for i in range(unreduced_loss.shape[0]):  # for every item in the batch
                per_token_losses += list(filter(lambda x: x != 0, unreduced_loss[i].tolist()))

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls),
                "mean_per_token_loss": np.mean(per_token_losses),'losses':per_token_losses}
