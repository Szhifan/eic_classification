import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import SequenceClassifierOutput
from .modeling_llama import LlamaModel
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel


class LlamaForSequenceClassification(LlamaPreTrainedModel):
    """
    Simple wrapper that adds a classification head to the custom LlamaModel
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Use the custom LlamaModel as backbone
        self.model = LlamaModel(config)
        
        # Simple classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # Use last token's hidden state as sentence representation
        # If there's no padding, use the last token; otherwise use the last non-padded token
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1  # Get actual sequence lengths
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            pooled_output = hidden_states[batch_indices, sequence_lengths]
        else:
            pooled_output = hidden_states[:, -1]  # Use last token
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
