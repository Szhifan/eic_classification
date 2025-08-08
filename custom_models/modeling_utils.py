from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
import torch
from torch import nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import AutoModel, BitsAndBytesConfig
import re
import os
from torch import Tensor
import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

DECODER_MODEL_TYPES = tuple(['gpt2', 'llama', 'mistral', 'qwen2', 'phi3', 'olmo'])
ARCHITECTURES = tuple(['NONE', 'INPLACE', 'EXTEND', 'INTER', 'EXTRA'])


class Pooler:
    def __init__(self, pool_type, include_prompt=False):
        self.pool_type = pool_type
        self.include_prompt = include_prompt or self.pool_type in ("cls", "last")

    def __call__(
        self, 
        last_hidden_states: Tensor,
        attention_mask: Tensor,
        prompt_length: int = None,
    ) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1)
        batch_size = last_hidden_states.shape[0]
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        device = last_hidden_states.device
        
        if not self.include_prompt and prompt_length is not None:
            if left_padding:
                prompt_mask = torch.ones_like(attention_mask)
                range_tensor = torch.arange(attention_mask.size(1), 0, -1, device=device).unsqueeze(0)
                prompt_mask = (range_tensor > (sequence_lengths-prompt_length).unsqueeze(1))
                attention_mask[prompt_mask] = 0
            else:
                attention_mask[:, :prompt_length] = 0
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
            attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            emb = s / d
        elif self.pool_type == "cls":
            emb = last_hidden[:, 0]
        elif self.pool_type == "last":
            if left_padding:
                emb = last_hidden[:, -1]
            else:
                emb = last_hidden[torch.arange(batch_size, device=device), sequence_lengths-1]
        else:
            raise ValueError(f"pool_type {self.pool_type} not supported")

        return emb
@dataclass
class BackwardSupportedArguments:
    architecture: str = field(
        default="None",
        metadata={"help": "Type of architecture to use. Options: " + ", ".join(ARCHITECTURES)}
    )
    mask_type: str = field(
        default="MASK0", 
        metadata={"help": "Type of sink mask to use. Options: MASK0, BACK"}
    )
    num_unsink_layers: int = field(
        default=0, 
        metadata={"help": "Number of layers to change to unsink attention."}
    )
    num_bidir_layers: int = field(
        default=0,
        metadata={"help": "Number of layers to change to bidirectional attention."}
    )
    unsink_layers: Optional[str] = field(
        default=None, 
        metadata={"help": "Manually set specific layers to change to unsink attention."}
    )
    bidir_layers: Optional[str] = field(
        default=None, 
        metadata={"help": "Manually set specific layers to change to bidirectional attention."}
    )
    res_connect: Optional[int] = field(
        default=None,
        metadata={"help": "Use ResConnect in Model."}
    )
    freeze_type: Optional[str] = field(
        default=None,
        metadata={"help": "Options:\
            all: freeze all parameters in the model.\
            backbone: freeze the backbone of the model, only train output related parameters(unfreeze_layer + norm).\
            default: freeze the backbone of the model, excluding layers changed to unsink or noncausal attention. \
            false\\none: no to freeze the model."
        }
    )
    num_unfreeze_layers: int = field(
        default=0,
        metadata={"help": "Unfreeze layers starting from the last layer of the frozen layers."}
    )
    model_init: bool = field(
        default=True,
        metadata={"help": "Whether to initialize extended layers using forward params."}
    )
    num_classifier_layers: int = field(
        default=1, metadata={"help": "Layers of classifiers."}
    )
    gguf_file: Optional[str] = field(
        default=None, metadata={"help": "Whether to load model from a specific gguf file."}
    )
    peft_file: Optional[str] = field(
        default=None, metadata={"help": "Peft file path"}
    )
    pool_type: str = field(
        default="avg", 
        metadata={"help": "Pooling type for the model output. Options: avg, weightedavg, cls, last"}
    )
    
    # Quantization parameters
    use_quantization: bool = field(
        default=True, metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_latent_attention: bool = field(
        default=False, metadata={"help": "Whether to use late attention"}
    )

    def __post_init__(self):
        if self.unsink_layers is None or self.unsink_layers.lower() in ("none", "false", "f", "no", "n", "[]", "{}", "()"):
            self.unsink_layers = []
        else:
            sep = "," if "," in self.unsink_layers else " "
            self.unsink_layers = [int(item) for item in self.unsink_layers.strip("[]").strip("{}").strip("()").split(sep)]

        if self.bidir_layers is None or self.bidir_layers.lower() in ("none", "false", "f", "no", "n", "[]", "{}", "()"):
            self.bidir_layers = []
        else:
            sep = "," if "," in self.bidir_layers else " "
            self.bidir_layers = [int(item) for item in self.bidir_layers.strip("[]").strip("{}").strip("()").split(sep)]

        self.architecture = self.architecture.upper()
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Invalid Model Architecture. Options: " + ", ".join(ARCHITECTURES))
        self.mask_type = self.mask_type.upper()
        
        if self.architecture == "NONE" :
            self.num_unsink_layers = 0
            self.num_bidir_layers = 0
            self.unsink_layers = []
            self.bidir_layers = []
            
        if (self.num_unsink_layers or self.num_bidir_layers) and (self.unsink_layers or self.bidir_layers):
            raise ValueError("Cannot set both unsink/bidir_layers and num_unsink/bidir_layers.")
        if self.freeze_type is not None:
            if self.freeze_type.lower() in ("none", "false", "f", "no", "n"):
                self.freeze_type = False
            elif self.freeze_type.lower() in ("true", "t", "yes", "y"):
                self.freeze_type = "default"
        if self.pool_type not in ("avg", "weightedavg", "cls", "last"):
            self.pool_type = "avg"
            print(f"Invalid pool_type {self.pool_type}, using default 'avg' pooling.")
        if self.gguf_file is not None and self.gguf_file.lower() in ("none", "false", "f", "no", "n"):
            self.gguf_file = None

        if self.peft_file is not None and self.peft_file.lower() in ("none", "false", "f", "no", "n"):
            self.peft_file = None
            
        # Handle quantization parameters
        if not torch.cuda.is_available():
            self.use_quantization = False

    def to_dict(self):
        return asdict(self)
    
    def get_quantization_config(self):
        """Create BitsAndBytesConfig from quantization parameters"""
        if not self.use_quantization:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["classifier", "latent_attention", "pooler", "score"],  # Skip quantization for custom layers
        )

    def get_suffix(self, training_args=None):
        suffix = []

        if self.gguf_file is not None:
            match = re.search(r"(\w+).gguf", self.gguf_file)
            if match:
                suffix.append(match.group(1).lower())

        suffix.append(self.architecture)

        if self.architecture != "NONE":
            if self.num_unsink_layers > 0:
                suffix.append(f"{self.mask_type.lower()}_{self.num_unsink_layers}")
            elif self.unsink_layers:
                suffix.append(f"{self.mask_type.lower()}_{self.unsink_layers}")

            if self.num_bidir_layers > 0:
                suffix.append(f"bidir_{self.num_bidir_layers}")
            elif self.bidir_layers:
                suffix.append(f"bidir_{self.bidir_layers}")
        
            if self.model_init:
                suffix.append("initialized")

        if self.num_classifier_layers > 0:
            suffix.append(f"classifier_{self.num_classifier_layers}")

        if self.freeze_type:
            freeze_type = "freeze_" + self.freeze_type
            if self.num_unfreeze_layers > 0:
                freeze_type += f"-{self.num_unfreeze_layers}"
            suffix.append(freeze_type)

        if training_args is not None:
            suffix.append(f"_lr_{format(training_args.learning_rate, '.0e')}")

            suffix = [s for s in suffix if s not in training_args.output_dir]

        return "_".join([""] + suffix)

def get_custom_model(model_args, config, MODEL_TYPE=AutoModel):
    is_keyword_present = any(keyword in model_args.model_name_or_path.lower() for keyword in DECODER_MODEL_TYPES)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    attn_implementation = "flash_attention_2" if \
        os.environ.get('TINY_FLASH_ATTN', False) \
        and is_keyword_present else None
    
    # Get quantization config if available
    quantization_config = None
    if hasattr(model_args, 'get_quantization_config'):
        quantization_config = model_args.get_quantization_config()
        if quantization_config:
            logger.info("Using 4-bit quantization for custom model")
    
    if model_args.model_name_or_path is None:
        config._attn_implementation = attn_implementation
        model = MODEL_TYPE.from_config(
            config, 
            trust_remote_code=model_args.trust_remote_code, 
            torch_dtype=torch_dtype,
            quantization_config=quantization_config
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        return model

    model_kwargs = {
        'gguf_file': getattr(config, "gguf_file", None),
        'from_tf': bool(".ckpt" in model_args.model_name_or_path),
        'config': config,
        'cache_dir': getattr(model_args, "cache_dir", None),
        'revision': getattr(model_args, "model_revision", None),
        'token': getattr(model_args, "token", None),
        'trust_remote_code': getattr(model_args, "trust_remote_code", False),
        'torch_dtype': torch_dtype,
    }
    
    # Add quantization config if available
    if quantization_config is not None:
        model_kwargs['quantization_config'] = quantization_config
        logger.info(f"Loading model with quantization config: {quantization_config}")

    model = MODEL_TYPE.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
        # ignore_mismatched_sizes=getattr(model_args, "ignore_mismatched_sizes", False),
        # low_cpu_mem_usage=getattr(model_args, "low_cpu_mem_usage", False),
        # attn_implementation=attn_implementation,
    )
    
    if getattr(config, "model_init", False): 
        if hasattr(model.base_model, "model_init"):
            model.base_model.model_init()

    if getattr(config, "num_classifier_layers", 1) == 2:
        classifier = nn.Sequential(
            nn.Linear(config.hidden_size,config.hidden_size), 
            nn.Dropout(), nn.GELU(), 
            nn.Linear(config.hidden_size, config.num_labels)
        )
        if hasattr(model, "score"):
            model.score = classifier
        else:
            model.classifier = classifier
            
    logger.info("All parameters:")
    logger.info(sum(p.numel() for p in model.parameters()))
    
    # Ensure custom layers use float32 precision to avoid quantization issues
    _ensure_custom_layers_precision(model)
    
    if getattr(config, "freeze_type", False):
        if hasattr(model.base_model, "freeze_model"):
            model.base_model.freeze_model(config)
        else:
            for param in model.base_model.parameters():
                param.requires_grad = False

        logger.info("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
    return model

def _ensure_custom_layers_precision(model):
    """
    Ensure custom layers (classifier, latent_attention, pooler) use float32 precision
    to avoid quantization issues with bitsandbytes
    """
    custom_layer_names = ["classifier", "latent_attention", "pooler", "score"]
    
    for name in custom_layer_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            if layer is not None:
                layer = layer.float()
                setattr(model, name, layer)
                for param in layer.parameters():
                    param.requires_grad_(True)
                logger.info(f"Set {name} layer to float32 precision")
    
    # Also check for nested custom layers in submodules
    for module_name, module in model.named_modules():
        if any(custom_name in module_name for custom_name in custom_layer_names):
            module = module.float()
            for param in module.parameters():
                param.requires_grad_(True)
            logger.info(f"Set custom module {module_name} to float32 precision")

def get_noncausal_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

def get_noncausal_attention_mask_0(self, attention_mask, input_shape, device=None, dtype=None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    assert attention_mask[:, 0].sum() == attention_mask.shape[0]
    assert self.config._attn_implementation != "flash_attention_2"
    # attention_mask[:, 0] = 0
    
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    
    extended_attention_mask = extended_attention_mask.repeat(1, 1, extended_attention_mask.shape[-1], 1)
    extended_attention_mask[:, :, 1:, 0] = 0

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

    return extended_attention_mask

def get_backward_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    output_attentions: bool,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask.flip(dims=(-1,))
        return None
    
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1]

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)

    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.flip(dims=(-2,-1))

    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    return causal_mask

def flip_tensor(tensor, flag=True):
    if flag:
        return tensor.flip(dims=(1,))
    else:
        return tensor
    
def use_res_connect(layers, connect_layers, index):
    return connect_layers is not None and index >= layers and (index - layers) % connect_layers == 0

class PlaceHolder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        pass

    def load_state_dict(self, state_dict, strict = True, assign = False):
        pass
class LatentAttention(nn.Module):
    """
    Latent Attention module where the query is the last hidden representation of an LLM,
    whereas key and value are learnable parameters.
    """
    def __init__(self, hidden_dim, num_latent_vectors=512, dropout_prob=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_latent_vectors = num_latent_vectors
        
        # Learnable key and value parameters
        self.key = nn.Parameter(torch.randn(num_latent_vectors, hidden_dim))
        self.value = nn.Parameter(torch.randn(num_latent_vectors, hidden_dim))
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Multi-head attention with 1 head
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.key)
        nn.init.xavier_uniform_(self.value)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Hidden representation from LLM [batch_size, seq_len, hidden_dim]
        Returns:
            context_vector: Attended output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use hidden states as queries
        query = hidden_states  # [batch_size, seq_len, hidden_dim]
        
        # Expand key and value for batch processing
        key = self.key.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_latent, hidden_dim]
        value = self.value.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_latent, hidden_dim]
        
        # Apply multi-head attention
        context, _ = self.attention(query, key, value)  # [batch_size, seq_len, hidden_dim]
        
        # Apply layer normalization and residual connection
        context = self.layer_norm(context + hidden_states)
        context = self.dropout(context)
        
        return context