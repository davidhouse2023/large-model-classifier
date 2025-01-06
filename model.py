import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LlamaClassifier(nn.Module):
    def __init__(self, MODEL_PATH, num_classes, layer_index, tokenizer):
        super(LlamaClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        print(self.model)
        self.model.layers = self.model.layers[:layer_index]
        for param in self.model.parameters():
            param.requires_grad = False
        self.pad_token_id = tokenizer.pad_token_id
        self.score = nn.Linear(self.model.config.hidden_size, num_classes, bias=False)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        hidden_states = transformer_outputs[0]  # transformer_outputs.last_hidden_state

        logits = self.score(hidden_states)  # [batch,seq_len,num_classes]

        batch_size = input_ids.shape[0]
        if input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]  # [batch,num_classes]

        loss = None
        if labels is not None:
            loss = self.loss_fct(pooled_logits, labels)

        return pooled_logits, loss

class Qwen25Classifier(nn.Module):
    def __init__(self, MODEL_PATH, num_classes, layer_index, tokenizer):
        super(Qwen25Classifier, self).__init__()
        self.model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
        # print(self.model)
        self.model.layers = self.model.layers[:layer_index]
        for param in self.model.parameters():
            param.requires_grad = False
        self.pad_token_id = tokenizer.pad_token_id
        self.score = nn.Linear(self.model.config.hidden_size, num_classes, bias=False)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )
        hidden_states = transformer_outputs[0]  # transformer_outputs.last_hidden_state
        # print("transformer_outputs", hidden_states.shape)

        logits = self.score(hidden_states)  # [batch,seq_len,num_classes]

        batch_size = input_ids.shape[0]
        if input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]  # [batch,num_classes]

        loss = None
        if labels is not None:
            loss = self.loss_fct(pooled_logits, labels)

        return pooled_logits, loss

# if __name__ == '__main__':
#     MODEL_PATH = "/opt/ailab_mnt1/LLM_MODELS/LLAMA/Qwen2.5_14B_Instruct"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     tokenizer.pad_token = tokenizer.eos_token
#     model = Qwen25Classifier(MODEL_PATH, num_classes=6, layer_index=12, tokenizer=tokenizer).to(
#         dtype=torch.bfloat16)
#     print(model)

