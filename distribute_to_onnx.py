import torch
from nezha_base_distribute_finetune import Model
import argparse
from model.configuration_nezha import NeZhaConfig

#model_path = './finetune_model/best_model.pth'


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
    # parser.add_argument("--bert_config", default='./dataset/base_config/config.json', type=str,
    #                     help="Path to save finetuned model")
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
    parser.add_argument("--model_name", default='nezha_base', type=str)

    args = parser.parse_args()

    if args.model_name == 'nezha_base':
        bert_config = './nezha-cn-base/config.json'
    elif args.model_name == 'nezha_wwm':
        bert_config = './nezha-cn-wwm/bert_config.json'
    NeZhaconfig = NeZhaConfig.from_json_file(bert_config)
    NeZhaconfig.vocab_size = 21128
    model = Model(args=args, config=NeZhaconfig)
    torch_model = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(torch_model)
    model = model.to(args.device)
    model.eval()

    Batch_size = 1
    seg_length = 67
    #filepath = 'best_model.onnx'
    dummy_input0 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    dummy_input1 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    dummy_input2 = torch.Tensor([[1 for _ in range(seg_length)]]).long()
    inputs = {
                'input_ids': dummy_input0.to(args.device),
                'attention_mask': dummy_input1.to(args.device),
                'token_type_ids': dummy_input2.to(args.device)
            }
    torch.onnx.export(model, inputs, args.filepath, opset_version=11)


if __name__ == '__main__':
    main()
