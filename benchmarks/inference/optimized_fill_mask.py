from typing import Dict
from transformers import Pipeline, FillMaskPipeline
from transformers.pipelines.base import PIPELINE_INIT_ARGS, GenericTensor, Pipeline, PipelineException
 
import numpy as np
import torch

class OptimizedFillMask(FillMaskPipeline):
    def _sanitize_parameters(self, top_k=None, targets=None):
        forward_params = {}

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            forward_params["target_ids"] = target_ids

        if top_k is not None:
            forward_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "opt-fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        return {}, forward_params, {}

    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs

    def _forward(self, model_inputs, top_k=5, target_ids=None):
        if target_ids is not None and target_ids.shape[0] < top_k:
            top_k = target_ids.shape[0]
        model_outputs = self.model(**model_inputs)
        model_outputs["input_ids"] = model_inputs["input_ids"]
        input_ids = model_outputs["input_ids"][0]
        outputs = model_outputs["logits"]
        masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
        # Fill mask pipeline supports only one ${mask_token} per sample

        logits = outputs[0, masked_index, :]
        probs = logits.softmax(dim=-1)
        if target_ids is not None:
            probs = probs[..., target_ids]

        values, predictions = probs.topk(top_k)
        return [values, predictions, input_ids, target_ids, masked_index]

    def postprocess(self, model_outputs):
        values = model_outputs[0]
        predictions = model_outputs[1]
        input_ids = model_outputs[2]
        target_ids = model_outputs[3]
        masked_index = model_outputs[4]
        result = []
        single_mask = values.shape[0] == 1
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = input_ids.numpy().copy()
                if target_ids is not None:
                    p = target_ids[p].tolist()

                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = self.tokenizer.decode(tokens, skip_special_tokens=single_mask)
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.decode([p]), "sequence": sequence}
                row.append(proposition)
            result.append(row)
        if single_mask:
            return result[0]
        return result

