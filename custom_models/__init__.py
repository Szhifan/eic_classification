from .modeling_llama import LlamaModel, LlamaForSequenceClassification
from .modeling_utils import *

from transformers import AutoModelForTokenClassification, OlmoConfig
import transformers.models


transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
transformers.models.llama.modeling_llama.LlamaForSequenceClassification = LlamaForSequenceClassification