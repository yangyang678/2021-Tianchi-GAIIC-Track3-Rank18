import torch
from model.modeling_nezha import NeZhaModel
from model.configuration_nezha import NeZhaConfig
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.roberta.configuration_roberta import RobertaConfig

base_config = NeZhaConfig.from_json_file('./nezha-cn-base/config.json')
base_model = NeZhaModel.from_pretrained(
    './nezha-cn-base/pytorch_model.bin', config=base_config)
torch.save(base_model.embeddings.word_embeddings.state_dict(), 'embeddingNeZhaBase.pth')

wwm_config = NeZhaConfig.from_json_file('./nezha-cn-wwm/bert_config.json')
wwm_model = NeZhaModel.from_pretrained(
    './nezha-cn-wwm/pytorch_model.bin', config=wwm_config)
torch.save(wwm_model.embeddings.word_embeddings.state_dict(), 'embeddingNeZhaWWM.pth')

roberta_config = RobertaConfig.from_json_file('./roberta-cn-wwm/bert_config.json')
roberta_model = BertModel.from_pretrained(
    './roberta-cn-wwm/pytorch_model.bin', config=roberta_config)
torch.save(roberta_model.embeddings.word_embeddings.state_dict(), 'embeddingRoberta.pth')

roberta_uer_config = RobertaConfig.from_json_file('./roberta-cn-uer/config.json')
roberta_uer_model = BertModel.from_pretrained(
    './roberta-cn-uer/pytorch_model.bin', config=roberta_uer_config)
torch.save(roberta_uer_model.embeddings.word_embeddings.state_dict(), 'embeddingRobertaUer.pth')

