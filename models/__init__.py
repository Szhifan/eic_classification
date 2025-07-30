from .modeling_llama import LlamaModel, LlamaForSequenceClassification
from .modeling_utils import *
import transformers.models


transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
