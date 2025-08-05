import os
import shutil
import pandas as pd
from datasets import Dataset
from datasets import enable_caching
from finetuning_llm_seqc.model_loader import ModelLoader
from finetuning_llm_seqc.data_preprocessor import DataPreprocessor
from finetuning_llm_seqc.model_finetuner import ModelFinetuner
from finetuning_llm_seqc.evaluater import Evaluater
import wandb
from accelerate import PartialState


def create_model_dir(task_name, method, model_path, lora_r, lora_alpha, lora_dropout, learning_rate, 
                     per_device_train_batch_size, train_epochs, train_type, test_type,
                     max_length, emb_type, input_type, recreate_dir=True):
    # create model dir
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, task_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, method)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_name = os.path.basename(model_path).split('.')[0] if '.' in os.path.basename(model_path) else os.path.basename(model_path)
    model_folder_name = model_name + '_' + f'_lora-r{lora_r}-a{lora_alpha}-d{lora_dropout}_lr{learning_rate}'
    model_folder_name += f'_bs{per_device_train_batch_size}_ep{train_epochs}_{train_type}_{test_type}_ml{max_length}_{emb_type}_{input_type}'
    output_dir = os.path.join(output_dir, model_folder_name)

    os.makedirs(output_dir, exist_ok=recreate_dir)
    return output_dir
        

def main():
    # Enable dataset caching for better performance
    enable_caching()
    
    ############################################################################

    # wandb.init(project="Re3-Sci", entity="re3-sci", name="finetune_EIC_SeqC")
    # basic settings
    # <settings>
    task_name ='edit_intent_classification'
    method = 'finetuning_llm_seqc' # select an approach from ['finetuning_llm_gen','finetuning_llm_seqc', 'finetuning_llm_snet', 'finetuning_llm_xnet']
    train_type ='train' # name of the training data in data/Re3-Sci/tasks/edit_intent_classification
    val_type = 'val' # name of the validation data in data/Re3-Sci/tasks/edit_intent_classification
    test_type = 'test' # name of the test data in data/Re3-Sci/tasks/edit_intent_classification
    # </settings>
    print('========== Basic settings: ==========')
    print(f'task_name: {task_name}')
    print(f'method: {method}')
    print(f'train_type: {train_type}')
    print(f'val_type: {val_type}')
    print(f'test_type: {test_type}')
    ############################################################################
    # load task data
    def load_task_data(task_name, train_type, val_type, test_type):
        """Load task data from CSV files"""
        base_path = f"data/Re3-Sci/tasks/{task_name}"
        
        train_df = pd.read_csv(f"{base_path}/{train_type}.csv")
        val_df = pd.read_csv(f"{base_path}/{val_type}.csv")
        test_df = pd.read_csv(f"{base_path}/{test_type}.csv")
        
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df)
        test_ds = Dataset.from_pandas(test_df)
        
        # Get unique labels
        all_labels = set(train_df['label'].unique()) | set(val_df['label'].unique()) | set(test_df['label'].unique())
        labels = sorted(list(all_labels))
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        
        return train_ds, val_ds, test_ds, labels, label2id, id2label
    
    train_ds, val_ds, test_ds, labels, label2id, id2label = load_task_data(task_name, train_type, val_type, test_type)
    print('========== 1. Task data loaded: ==========')
    print(f'train_ds: {train_ds}')
    print(f'val_ds: {val_ds}')
    print(f'test_ds: {test_ds}')
    print(f'labels: {labels}')
    print(f'label2id: {label2id}')
    print(f'id2label: {id2label}')
    ############################################################################
    # load model from path
    # <settings>
    model_path = 'meta-llama/Llama-3.2-1B'  # 请修改为您的模型路径
    use_custom_llama = False  # 设置为 True 以使用自定义 Llama 模型
    use_multi_gpu = True  # 是否使用多GPU训练
    if use_multi_gpu:
        device_map="DDP" # for DDP and running with `accelerate launch test_sft.py`
        device_string = PartialState().process_index
        device_map={'':device_string}
    else:   
        device_map="auto"
    emb_type = None # transformation function for xnet and snet approaches, select from [''diff', diffABS', 'n-diffABS', 'n-o', 'n-diffABS-o'], None for SeqC and Gen
    #input type for the model, select from ['text_nl_on', 'text_st_on', 'inst_text_st_on', 'inst_text_nl_on'] 
    #for natural language input, structured input, instruction + structured input,  instruction + natural language input, respectively
    input_type='text_st_on'
    
    # Custom Llama model arguments (from BackwardSupportedArguments)
    model_args = {
        'architecture': 'INPLACE',  # Type of architecture: NONE, INPLACE, EXTEND, INTER, EXTRA
        'mask_type': 'MASK0',  # Type of sink mask: MASK0, BACK
        'num_unsink_layers': 0,  # Number of layers to change to unsink attention
        'num_bidir_layers': 0,  # Number of layers to change to bidirectional attention
        'unsink_layers': None,  # Manually set specific layers to change to unsink attention
        'bidir_layers': None,  # Manually set specific layers to change to bidirectional attention
        'res_connect': 3,  # Use ResConnect in Model
        'freeze_type': None,  # Freeze type: all, backbone, default, false/none
        'num_unfreeze_layers': 0,  # Unfreeze layers starting from the last layer
        'model_init': True,  # Whether to initialize extended layers using forward params
        'num_classifier_layers': 1,  # Layers of classifiers
    }
    # </settings>
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model_from_path(
        model_path,
        device_map=device_map, 
        labels=labels, 
        label2id=label2id, 
        id2label=id2label, 
        emb_type=emb_type, 
        input_type=input_type,
        use_custom_llama=use_custom_llama,
        **model_args
    )
    print('========== 2. Model loaded: ==========')
    print(f'model: {model}')
    

    ############################################################################
    # preprocess dataset
    # <settings>
    max_length = 1024
    # </settings>
    data_preprocessor = DataPreprocessor()
    if method in ['finetuning_llm_seqc', 'finetuning_llm_snet', 'finetuning_llm_xnet']:
        train_ds = data_preprocessor.preprocess_data(train_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
        val_ds = data_preprocessor.preprocess_data(val_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
        test_ds = data_preprocessor.preprocess_data(test_ds, label2id, tokenizer, max_length=max_length, input_type=input_type)
        response_key = None
    elif method in ['finetuning_llm_gen']:
        train_ds,_ = data_preprocessor.preprocess_data(train_ds, max_length=max_length, input_type=input_type, is_train=True)
        val_ds,_ = data_preprocessor.preprocess_data(val_ds, max_length=max_length, input_type=input_type, is_train=False)
        test_ds, response_key = data_preprocessor.preprocess_data(test_ds, max_length=max_length, input_type=input_type, is_train=False)
    print('========== 3. Dataset preprocessed: ==========')
    print('train_ds: ', train_ds)
    print('val_ds: ', val_ds)
    print('test_ds: ', test_ds)
    print('response_key: ', response_key)


    ############################################################################
    # fine-tune model
    # <settings>
    lora_r = 8 # LoRA rank parameter
    lora_alpha = 8 # Alpha parameter for LoRA scaling
    lora_dropout = 0.1 # Dropout probability for LoRA layers
    learning_rate = 2e-4 # Learning rate
    per_device_train_batch_size = 16# Batch size per GPU for training 
    train_epochs = 1 # Number of epochs to train
    recreate_dir = True # Create a directory for the model
    do_train = False # Whether to train the model
    # </settings>
    # create model dir to save the fine-tuned model
    output_dir = create_model_dir(task_name, method, model_path, lora_r, lora_alpha, lora_dropout, learning_rate, 
                     per_device_train_batch_size, train_epochs, train_type, test_type,
                     max_length, emb_type, input_type, recreate_dir=recreate_dir)
    print('========== 4. Model dir created: ==========')
    print('output_dir: ', output_dir)
    # fine-tune\
    if do_train:
        model_finetuner = ModelFinetuner()
        model_finetuner.fine_tune(model, tokenizer, train_ds = train_ds , val_ds = val_ds,  lora_r = lora_r, lora_alpha = lora_alpha, lora_dropout = lora_dropout,
                                    learning_rate = learning_rate, per_device_train_batch_size = per_device_train_batch_size, train_epochs = train_epochs, output_dir = output_dir)
        print('========== 5. Model fine-tuned: ==========')
        print('output_dir: ', output_dir)

    ############################################################################
    # evaluate the fine-tuned model
    evaluater = Evaluater()
    evaluater.evaluate(test_ds, model, model_dir=output_dir, labels=labels, label2id=label2id, id2label=id2label, emb_type=emb_type, input_type=input_type, response_key=response_key)
    print('========== 6. Model evaluated ==========')
    print('output_dir: ', output_dir)
    print('========== DONE ==========')
    


if __name__ == "__main__":
    main()
