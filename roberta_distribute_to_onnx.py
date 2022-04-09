import torch
from roberta_distribute_finetune import Model
import argparse
from transformers.models.roberta.configuration_roberta import RobertaConfig
from onnxruntime_tools import optimizer


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--maxlen", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--lr", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--epoch", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--struc', default='cls', type=str,
                        choices=['cls', 'bilstm', 'bigru', 'idcnn'])
    parser.add_argument("--model_save_path", default='./finetune_model/', type=str,
                        help="Path to save finetuned model")
    parser.add_argument("--pre_model_path", default='./pretrain_model/', type=str)
    #parser.add_argument("--bert_config", default='./roberta-cn-wwm/bert_config.json', type=str,
    #                    help="Path to save finetuned model")
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lookahead", action='store_true',
                        help="Whether to use lookahead.")
    parser.add_argument("--dropout_num", default=1, type=int)
    parser.add_argument("--AveragePooling", default=True, type=bool)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lstm_dim", default=256, type=int)
    parser.add_argument("--gru_dim", default=256, type=int)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)
    parser.add_argument("--model_path", default='./finetune_model/best_model.pth', type=str)
    parser.add_argument("--filepath", default='best_model.onnx', type=str)
    parser.add_argument("--model_name", default='hgf', type=str)

    args = parser.parse_args()

    if args.model_name == 'hgf':
        bert_config = './roberta-cn-wwm/bert_config.json'
    elif args.model_name == 'uer':
        bert_config = './roberta-cn-uer/config.json'

    Robertaconfig = RobertaConfig.from_json_file(bert_config)
    Robertaconfig.vocab_size = 21128
    model = Model(args=args, config=Robertaconfig)
    torch_model = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(torch_model)
    model = model.to(args.device)
    model.eval()

    Batch_size = 1
    seg_length = 67
    #filepath = 'best_model2.onnx'
    dummy_input0 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    dummy_input1 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    dummy_input2 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    inputs = {
                'input_ids': dummy_input0.to(args.device),
                'attention_mask': dummy_input1.to(args.device),
                'token_type_ids': dummy_input2.to(args.device)
            }
    torch.onnx.export(model, inputs, args.filepath, opset_version=11)
    optimized_model = optimizer.optimize_model(args.filepath, model_type='bert', num_heads=12, hidden_size=768)
    optimized_model.save_model_to_file(args.filepath)


if __name__ == '__main__':
    main()
    