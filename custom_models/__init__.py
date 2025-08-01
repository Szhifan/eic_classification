from .modeling_llama import LlamaModel
from .modeling_utils import *

from transformers import AutoModelForTokenClassification, OlmoConfig
import transformers.models

from .modeling_flash_attention_utils import _flash_attention_forward
transformers.models.llama.modeling_llama._flash_attention_forward = _flash_attention_forward

transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
