#! -*- coding: utf-8 -*-
# 用Seq2Seq做小学数学应用题
# 数据集为ape210k：https://github.com/Chenny0808/ape210k
# Base版准确率为70%+，Large版准确率为73%+
# 实测环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.8
# 介绍链接：https://kexue.fm/archives/7809

from __future__ import division
import json, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from sympy import Integer

# import tensorflow as tf
# from keras.callbacks import ModelCheckpoint

import os
import argparse

# 选择gpu
os.environ["CUDA_VISIBLE_DEVICES"]='1'


def is_equal(a, b):
    """比较两个结果是否相等
    """
    a = round(float(a), 6)
    b = round(float(b), 6)
    return a == b


def load_data(filename):
    """读取训练数据，并做一些标准化，保证equation是可以eval的
    参考：https://kexue.fm/archives/7809
    """
    D = []
    for l in open(filename):
        l = json.loads(l)
        question, equation, answer = l['original_text'], l['equation'], l['ans']
        question = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+(\\2))', question)
        equation = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+(\\2))', equation)
        answer = re.sub('(\d+)\((\d+/\d+)\)', '(\\1+(\\2))', answer)
        equation = re.sub('(\d+)\(', '\\1+(', equation)
        answer = re.sub('(\d+)\(', '\\1+(', answer)
        question = re.sub('\((\d+/\d+)\)', '\\1', question)
        equation = re.sub('([\.\d]+)%', '(\\1/100)', equation)
        answer = re.sub('([\.\d]+)%', '(\\1/100)', answer)
        equation = equation.replace(':', '/').replace('%', '/100')
        answer = answer.replace(':', '/').replace('%', '/100')
        if equation[:2] == 'x=':
            equation = equation[2:]
        try:
            if is_equal(eval(equation), eval(answer)):
                D.append((question, equation, answer))
        except:
            continue
    return D


# 数据生成batch
class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (question, equation, answer) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                question, equation, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []

# 计算二院交叉熵
class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# ape模型seq2seq预测、生成公式
class AutoSolve(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids).replace(' ', '')


# 评估与保存
class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, mymodel):
        self.best_acc = 0.
        self.mymodel = mymodel

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型

        # 保存断点模型
        model.save_weights('./weights/{}_{}_model.weights'.format(self.mymodel,epoch))  # 保存模型

        # 保存最佳模型
        if metrics['acc'] >= self.best_acc:
            self.best_acc = metrics['acc']
            model.save_weights('./best_{}_model.weights'.format(self.mymodel))

        metrics['best_acc'] = self.best_acc
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total, right = 0.0, 0.0
        for question, equation, answer in tqdm(data):
            total += 1
            pred_equation = autosolve.generate(question, topk)
            try:
                right += int(is_equal(eval(pred_equation), eval(answer)))
            except:
                pass
        return {'acc': right / total}


def predict(in_file, out_file, topk=1):
    """输出预测结果到文件
    该函数主要为比赛 https://www.datafountain.cn/competitions/467 所写，
    主要是读取该比赛的测试集，然后预测equation，并且根据不同的问题输出不同格式的答案，
    out_file可以直接提交到线上评测，线上准确率可以达到38%+。
    """
    fw = open(out_file, 'w', encoding='utf-8')
    raw_data = pd.read_csv(in_file, header=None, encoding='utf-8')
    for i, question in tqdm(raw_data.values):
        pred_equation = autosolve.generate(question, topk) # 生成公式
        if '.' not in pred_equation:
            pred_equation = re.sub('([\d]+)', 'Integer(\\1)', pred_equation)
        try:
            pred_answer = eval(pred_equation)
        except:
            pred_answer = np.random.choice(21) + 1
        if '.' in pred_equation:
            if u'百分之几' in question:
                pred_answer = pred_answer * 100
            pred_answer = round(pred_answer, 2)
            if int(pred_answer) == pred_answer:
                pred_answer = int(pred_answer)
            if (
                re.findall(u'多少[辆|人|个|只|箱|包本|束|头|盒|张]', question) or
                re.findall(u'几[辆|人|个|只|箱|包|本|束|头|盒|张]', question)
            ):
                if re.findall(u'至少|最少', question):
                    pred_answer = np.ceil(pred_answer)
                elif re.findall(u'至多|最多', question):
                    pred_answer = np.floor(pred_answer)
                else:
                    pred_answer = np.ceil(pred_answer)
                pred_answer = int(pred_answer)
            pred_answer = str(pred_answer)
            if u'百分之几' in question:
                pred_answer = pred_answer + '%'
        else:
            pred_answer = str(pred_answer)
            if '/' in pred_answer:
                if re.findall('\d+/\d+', question):
                    a, b = pred_answer.split('/')
                    a, b = int(a), int(b)
                    if a > b:
                        pred_answer = '%s_%s/%s' % (a // b, a % b, b)
                else:
                    if re.findall(u'至少|最少', question):
                        pred_answer = np.ceil(eval(pred_answer))
                    elif re.findall(u'至多|最多', question):
                        pred_answer = np.floor(eval(pred_answer))
                    else:
                        pred_answer = np.ceil(eval(pred_answer))
                    pred_answer = str(int(pred_answer))
        fw.write(str(i) + ',' + pred_answer + '\n')
        fw.flush()
    fw.close()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert", type=str, help="选择bert 或者 roberta.")
    parser.add_argument("--do_train", action='store_true',help="Whether to run training.")
    parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--maxlen", default=160, type=int,  help="Total batch size for training.")
    parser.add_argument("--epochs", default=100, type=int,  help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--output_dir", default='outputs_bert', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    # parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    args = parser.parse_args()
    print('here:',args)

    # 基本参数
    maxlen = args.maxlen
    # epochs = args.epochs


    # 加载预训练模型
    if args.bert_model == 'bert':
        config_path = './weights/chinese_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = './weights/chinese_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = './weights/chinese_L-12_H-768_A-12/vocab.txt'
    elif args.bert_model == 'roberta':
        config_path = './weights/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = './weights/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = './weights/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

    # 加载数据集
    train_data = load_data('./datasets/train.ape.json')
    valid_data = load_data('./datasets/valid.ape.json')
    test_data = load_data('./datasets/test.ape.json')

    # 加载并精简词表，建立分词器
    token_dict, keep_tokens = load_vocab(
        dict_path=dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    tokenizer = Tokenizer(token_dict, do_lower_case=True)


    # 建模
    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    )
    output = CrossEntropy(2)(model.inputs + model.outputs)

    model = Model(model.inputs, output)
    model.compile(optimizer=Adam(args.learning_rate))  # 用足够小的学习率
    model.summary()

    # seq2seq预测公式
    autosolve = AutoSolve(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

    # 训练模式
    if args.do_train:

        evaluator = Evaluator(args.bert_model)
        train_generator = data_generator(train_data, args.batch_size)

        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=args.epochs,
            callbacks=[evaluator]
        )


    # 加载最佳权重模型
    model.load_weights('./best_{}_model.weights'.format(args.bert_model))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 预测赛道数据集（.csv格式）
    if args.bert_model == 'bert':
        predict('./datasets/test.csv', os.path.join(args.output_dir, 'bert_outputs.csv') )
    else:
        predict('./datasets/test.csv', os.path.join(args.output_dir, 'roberta_outputs.csv') )



