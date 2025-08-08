from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from peft import AutoPeftModelForSequenceClassification, PeftModel
from transformers import AutoTokenizer
from .model_finetuner import collate_fn


class Evaluater:
    def __init__(self) -> None:
        print('Evaluating the model...')
    def merge_model(self, finetuned_model_dir:Path, labels, label2id, id2label):
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        compute_dtype = getattr(torch, "float16")   
        
        # Load model on single device for evaluation
        model =  AutoPeftModelForSequenceClassification.from_pretrained(
                        str(finetuned_model_dir)+'/',
                        torch_dtype=compute_dtype,
                        device_map={"": "cuda:0"},  # Force everything to cuda:0
                        num_labels = len(labels),
                    )
        
        model.config.id2label = id2label
        model.config.label2id = label2id

        model = model.merge_and_unload()
        
        # Ensure the entire model is on cuda:0
        model.config.pad_token_id = tokenizer.pad_token_id
       
        if 'mistral' in str(finetuned_model_dir):
            model.config.sliding_window = 4096
        
        return model, tokenizer

    def predict(self, test, model, tokenizer, id2label, output_dir, batch_size=32):
        eval_file = output_dir / "eval_pred.csv"
        print('eval_file', eval_file)
        if eval_file.exists():
            eval_file.unlink()
        model = model.to('cuda:0')
        # Check model's data type by examining the first parameter
        model_dtype = next(model.parameters()).dtype
        
        # Process data in batches
        all_results = []
        num_batches = (len(test) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test))
            
            # Get batch samples - handle HuggingFace Dataset indexing properly
            batch_indices = list(range(start_idx, end_idx))
            batch_samples = test.select(batch_indices)
            
            # Convert to list of dictionaries for collate_fn
            batch_samples_list = []
            for i in range(len(batch_samples)):
                batch_samples_list.append(batch_samples[i])
            
            # Debug: print the type and structure of the first sample

            # Use collate_fn to prepare batch inputs
            inputs = collate_fn(batch_samples_list, device='cuda:0', tokenizer=tokenizer)
            
            with torch.no_grad():
                logits = model(**inputs, return_dict=True).logits
            
            # Process predictions for the entire batch
            predicted_class_ids = logits.argmax(dim=1).cpu().numpy()
            
            # Create results for this batch
            for i, sample_idx in enumerate(range(start_idx, end_idx)):
                sample = test[sample_idx]
                pred = id2label[int(predicted_class_ids[i])]
                
                result = {
                    'doc_name': sample['doc_name'],
                    'node_ix_src': sample['node_ix_src'], 
                    'node_ix_tgt': sample['node_ix_tgt'],
                    'true': id2label[sample['label']],
                    'pred': pred,
                }
                all_results.append(result)
        
        # Save all results at once
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(eval_file, index=False)
            

    def evaluate(self, test, model=None, tokenizer=None, model_dir=None, output_dir=None,  do_predict = True, 
                 labels=None, label2id=None, id2label=None, 
                 emb_type=None, input_type=None, response_key = None, batch_size=32):
        """
        Evaluate the model on the test set
        :param test: Test set
        :param model: Hugging Face model, the fine-tuned model
        :param tokenizer: Model tokenizer
        :param model_dir: Directory containing the fine-tuned model
        :param output_dir: Directory to save the evaluation results
        :param do_predict: Whether to predict the labels
        :param labels: List of labels
        :param label2id: Dictionary mapping labels to ids
        :param id2label: Dictionary mapping ids to labels
        :param emb_type: transformation function, None for SeqC
        :param input_type: Type of input text
        :param response_key: Response key, None for SeqC
        :param batch_size: Batch size for prediction
        """
        # load the model
        if model is None or tokenizer is None:
            model, tokenizer = self.merge_model(model_dir, labels, label2id, id2label)

        start_time = pd.Timestamp.now()
        if output_dir is None:
                output_dir = Path(model_dir)
        if do_predict:
            self.predict(test, model, tokenizer, id2label, output_dir, batch_size=batch_size)
        end_time = pd.Timestamp.now()
        inference_time = end_time - start_time
        inference_time = inference_time.total_seconds()

       
        df = pd.read_csv(output_dir / "eval_pred.csv")
        none_nr = len(df[df['pred'] == 'none'])
        assert none_nr == 0, f'None labels found in the predictions: {none_nr}' 
        total_nr = len(df)

        eff = round((total_nr / int(inference_time)), 1)
        with open (output_dir / "inference_time.json", 'w') as f:
            json.dump({'inference_time':int(inference_time), 'inference_efficieny':eff}, f, indent=4)

        df = df[df['pred'] != 'none']
        y_pred = df["pred"]
        y_true = df["true"]
        print(df)
        
        # Map labels to ids
        map_func = lambda label: label2id[label]
        y_true = np.vectorize(map_func)(y_true)
        y_pred = np.vectorize(map_func)(y_pred)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        print(f'Accuracy: {accuracy:.3f}')
        
        # Generate classification report
        class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels, output_dict=True, zero_division=0)
        print('\nClassification Report:')
        class_report['none_nr'] = none_nr
        class_report['AIR'] = round(((total_nr - none_nr) / total_nr)*100, 1)
        print(class_report)

        eval_file = output_dir / "eval_report.json"
        if eval_file.exists():
            eval_file.unlink()
        with open(str(eval_file), 'w') as f:
            json.dump(class_report, f, indent=4)
        