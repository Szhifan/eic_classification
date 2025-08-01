import torch
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np
from transformers import Trainer, TrainingArguments                         
import evaluate
accuracy = evaluate.load("accuracy")


from torch.nn.utils.rnn import pad_sequence

def collate_fn(examples, device=None, tokenizer=None):
    # Extract input_ids and labels from examples
    batch_input_ids = []
    batch_attention_masks = []
    for example in examples:
        # Convert to tensors if they're not already

        input_ids = torch.tensor(example['input_ids_text'], dtype=torch.long)
        att_mask = torch.tensor(example['attention_mask_text'], dtype=torch.long)
        batch_input_ids.append(input_ids)
        batch_attention_masks.append(att_mask) 
    labels = torch.stack([torch.tensor(example["label"], dtype=torch.long) for example in examples]) 
    # Get pad_token_id from tokenizer, ensure it's never None
    if tokenizer and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        # Use a safe default value (0 is commonly used for padding)
        pad_token_id = 0
    
    # Use pad_sequence for efficient padding
    input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_id)
    
    # Create attention masks (1 for real tokens, 0 for padding)
    attention_masks = pad_sequence(batch_attention_masks, batch_first=True, padding_value=0)

    if device is not None:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
    
    return {"input_ids": input_ids, 
            "attention_mask": attention_masks,
            "labels": labels}

def compute_metrics(eval_pred):   
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

class ModelFinetuner:
    def __init__(self) -> None:
        ''

    def print_trainable_parameters(self, model, use_4bit = False):
        """Prints the number of trainable parameters in the model.
        :param model: PEFT model
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        if use_4bit:
            trainable_params /= 2
        print(
            f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
        )
    
    def fine_tune(self, 
                  model,
                  tokenizer,
                  train_ds = None,
                  val_ds = None,
                  lora_r = 128,
                  lora_alpha = 128,
                  lora_dropout = 0.1,
                  learning_rate = 2e-4,
                  per_device_train_batch_size = 32,
                  train_epochs = 10,
                  output_dir = None,
                  bias = 'none',
                  target_modules="all-linear",
                  task_type = "SEQ_CLS",
                  max_seq_length = 4096,
                  use_multi_gpu = False,
                  ddp_find_unused_parameters = False
                  ):
        print('fine-tuning....')
        
        # Check GPU availability and setup multi-GPU training
        device_count = torch.cuda.device_count()
        print(f"Available GPUs: {device_count}")
        
        if use_multi_gpu and device_count > 1:
            print(f"Using {device_count} GPUs for training")
            # Adjust batch size for multi-GPU training
            # The effective batch size will be per_device_train_batch_size * num_gpus
            print(f"Effective batch size: {per_device_train_batch_size * device_count}")
        elif device_count == 1:
            print("Using single GPU for training")
        else:
            print("Using CPU for training")
        
        # Enable gradient checkpointing to reduce memory usage during fine-tuning
        model.gradient_checkpointing_enable()
       
        # Prepare the model for training 
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r = lora_r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type,
        )
        model = get_peft_model(model, peft_config)

        # Print information about the percentage of trainable parameters
        self.print_trainable_parameters(model)
       
        args = TrainingArguments(
                    output_dir = output_dir,
                    num_train_epochs=train_epochs,
                    per_device_train_batch_size = per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps = 8,
                    learning_rate = learning_rate, 
                    logging_steps=10,
                    fp16 = True,
                    weight_decay=0.001,
                    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
                    max_steps=-1,
                    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
                    group_by_length=False,
                    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
                    # report_to="wandb",                  # report metrics to wandb
                    eval_strategy="epoch",              # save checkpoint every epoch
                    save_strategy="epoch",
                    gradient_checkpointing=True,              # use gradient checkpointing to save memory
                    optim="paged_adamw_32bit",
                    remove_unused_columns=False,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_accuracy",
                    label_names = ['labels'],
                    save_total_limit=2,
                    ) 
        
        # Create a partial function to pass tokenizer to collate_fn
        from functools import partial
        data_collator = partial(collate_fn, tokenizer=tokenizer)
        
        trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
            )
       
        model.config.use_cache = False
        do_train = True

        # Launch training and log metrics
        print("Training...")
        if do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)

