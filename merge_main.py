import time
import requests
import numpy as np
import onnxruntime
import onnx
import pickle
import random
import torch
from flask import Flask, request
import os
import sys


# 允许使用类似Flask的别的服务方式
app = Flask(__name__)

with open('./dataset/keep_tokens/tokens_dict_3.pkl', 'rb') as l:
    vocab = pickle.load(l)


def truncate_sequences(maxlen, index, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, input_numpy):
        """
        input_feed={self.input_name: input_numpy}
        :param input_name:
        :param input_numpy:
        :return:
        """
        input_feed = {}
        for i, name in enumerate(input_name):
            input_feed[name] = input_numpy[i]
        return input_feed

    def forward(self, input_numpy):
        # 输入数据的类型必须与模型一致
        input_feed = self.get_input_feed(self.input_name, input_numpy)
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores


#正式测试，batch_size固定为1
@app.route("/tccapi", methods=['GET', 'POST'])
def tccapi():
    data = request.get_data()
    if (data == b"exit"):
        print("received exit command, exit now")
        os._exit(0)

    input_list = request.form.getlist("input")
    index_list = request.form.getlist("index")
    
    response_batch = {}
    response_batch["results"] = []
    for i in range(len(index_list)):
        index_str = index_list[i]

        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split("\t")
            query_A = elems[0].strip()
            query_B = elems[1].strip()
            token_ids, attention_mask, segment_ids = make_input_fn(query_A, query_B)
            predict1 = infer(model1, token_ids, attention_mask, segment_ids)
            predict2 = infer(model2, token_ids, attention_mask, segment_ids)
            predict3 = infer(model3, token_ids, attention_mask, segment_ids)
            predict4 = infer(model4, token_ids, attention_mask, segment_ids)
            predict = (predict1 + predict2 + predict3 + predict4) / 4
            response["predict"] = predict
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
        response_batch["results"].append(response)
    
    return response_batch


def make_input_fn(query_A, query_B):
    t1 = [int(i) for i in query_A.split(' ')]
    t2 = [int(i) for i in query_B.split(' ')]
    truncate_sequences(64, -1, t1, t2)
    text1_ids = [vocab.get(t, 1) for t in t1]
    text2_ids = [vocab.get(t, 1) for t in t2]

    token_ids = ([2] + text1_ids + [3] + text2_ids + [3])
    segment_ids = [0] * len(token_ids)

    padding = [0 for _ in range(64 + 3 - len(token_ids))]
    attention_mask = len(token_ids) * [1] + len(padding) * [0]
    token_ids.extend(padding), segment_ids.extend(padding)
    token_ids = np.array([token_ids])
    attention_mask = np.array([attention_mask])
    segment_ids = np.array([segment_ids])

    return token_ids, attention_mask, segment_ids


# 此处示例，需要根据模型类型重写
def infer(model, token_ids, attention_mask, segment_ids):
    predict = [0]
    outputs = model.forward([token_ids, attention_mask, segment_ids])
    predict = outputs[0][:, 1] / (outputs[0].sum(axis=1) + 1e-8)

    return float(predict[0])


# 此处示例，需要根据模型类型重写
def init_model():
    model1 = ONNXModel('./best_model1.onnx')
    model2 = ONNXModel('./best_model2.onnx')
    model3 = ONNXModel('./best_model3.onnx')
    model4 = ONNXModel('./best_model4.onnx')
    return model1, model2, model3, model4


if __name__ == "__main__":

    model1, model2, model3, model4 = init_model()
    app.run(host="0.0.0.0", port=8080)

