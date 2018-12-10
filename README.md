# BERT-CRF-NER
Tensorflow solution of NER task Using CRF model with Google BERT Fine-tuning
Mainly adapted from BERT-BiLSTM-CRF:https://github.com/macanv/BERT-BiLSTM-CRF-NER

使用谷歌的BERT模型在CRF模型上进行预训练用于中文命名实体识别的Tensorflow代码'

The Chinese training data includes msra-ne(24 named entities),msra_ne1(3 named entities),pku98-gold(6 named entities)
  
The CoNLL-2003 data($PATH/NERdata/ori/) come from:https://github.com/kyzhouhzau/BERT-NER 
  
The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py  


Try to implement NER work based on google's BERT code and CRF.


## How to train

#### 1.using config param in terminal

```
  python3 bert_lstm_ner.py   \
                  --task_name="NER"  \ 
                  --do_train=True   \
                  --do_eval=True   \
                  --do_predict=True
                  --data_dir=NERdata   \
                  --vocab_file=checkpoint/vocab.txt  \ 
                  --bert_config_file=checkpoint/bert_config.json \  
                  --init_checkpoint=checkpoint/bert_model.ckpt   \
                  --max_seq_length=128   \
                  --train_batch_size=32   \
                  --learning_rate=2e-5   \
                  --num_train_epochs=3.0   \
                  --output_dir=./output/result_dir/ 
 ```       


## result:
all params using default
#### Tested on msra-ner 24 entities:
![](/picture1.png)

#### In pku98-gold 6 entities data set:（msra-ne和pku98标注标准不一，结果存疑）
![](/picture2.png)

## What I add:
#### Automatically read the label from trainning set, no need to hard-encode the label. Rewritten in Class NerProcessor.
```
    def get_labels(self, data_dir):  # to be improved
        return self._get_labels(
            self._read_data(os.path.join(data_dir, "train.txt"))[0]
        )

    def get_predict_labels(self, data_dir):  # to be improved
        return self._get_labels(
            self._read_data(os.path.join(data_dir, "pku98-gold.txt"))[0]
        )

    def _get_labels(self, lines):
        labels = set()
        for line in lines:
            for l in line[0].split():
                labels.add(l)
        labels = list(labels)
        labels.insert(0, "[CLS]")
        labels.insert(1, "[SEP]")
        labels.insert(2, "X")
        return labels
```

#### Make sure the label2id can be reused in predict data set, because once we rebuild the label2id of predict data set, the id might not the same as trainning set. And the number of named entities might much smaller than the original training set, wo no need to predict so much label like wo do it in trainning set.Rewritten in Function convert_single_example(...).
```angular2html
    label_map = {}
    if os.path.exists(os.path.join(FLAGS.output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
            label_map = pickle.load(rf)
    else:
        # 1表示从1开始对label进行index化
        for (i, label) in enumerate(label_list, 1):
            label_map[label] = i
        # 保存label->index 的map
        # print("the label map is", label_map)
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)
    eval_list_dir = os.path.join(FLAGS.output_dir, 'eval_ids_list.txt')
    if not os.path.exists(eval_list_dir):
        eval_list = [ label_map[i] for i in predict_label_list if i!='O' and i!='X' and i!="[CLS]" and i!="[SEP]"]
        file=open(eval_list_dir, 'w')
        for i in eval_list:
            file.write(str(i) + '\n')
        file.close()
        print("Get the eval list in eval_tag_list.txt")
```

#### Rewritten Class: CRF layer in crf_layer.py
```
import tensorflow as tf
from tensorflow.contrib import crf


class CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type,num_layers, droupout_rate,
                 initializers,num_labels, seq_length, labels, lengths, is_training):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.droupout_rate = droupout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths

        self.is_training = is_training

    def add_crf_layer(self):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate set 0.5 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate)
        # project
        logits = self.project_layer(self.embedded_chars)
        # crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return ((loss, logits, trans, pred_ids))


    def project_layer(self, embedded_chars, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        hidden_state = self.embedded_chars.get_shape()[-1]
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[hidden_state, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                embeddeding = tf.reshape(self.embedded_chars,[-1, hidden_state])
                pred = tf.nn.xw_plus_b(embeddeding, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])


    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans
```

##Notice:
micro f1 is different from macro f1.http://sofasofa.io/forum_main_post.php?postid=1001112
如果这个数据集中各个类的分布不平衡的话，更建议使用mirco-F1，因为macro没有考虑到各个类别的样本大小。

## reference: 
+ The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://github.com/kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

+ [https://github.com/zjy-ucas/ChineseNER](https://github.com/zjy-ucas/ChineseNER)

> Any problem please email me(ma_cancan@163.com)
