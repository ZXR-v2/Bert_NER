# BERT-CRF-NER
Tensorflow solution of NER task Using CRF model with Google BERT Fine-tuning
Mainly adapted from BERT-BiLSTM-CRF:https://github.com/macanv/BERT-BiLSTM-CRF-NER

使用谷歌的BERT模型在CRF模型上进行预训练用于中文命名实体识别的Tensorflow代码'

The Chinese training data includes msra-ne(24 named entities),msra_ne1(3 named entities),pku98-gold(6 named entities)
  
The CoNLL-2003 data($PATH/NERdata/ori/) come from:https://github.com/kyzhouhzau/BERT-NER 
  
The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py  


Try to implement NER work based on google's BERT code and CRF.


## How to train

#### 1.using config param in terminal. Or we can change parameters in the end of bert_crf_test.py.

```
  python3 bert_crf_test.py   \
                  --task_name="NER"  \ 
                  --do_train=True   \
                  --do_eval=True   \
                  --do_predict=False  \
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
#### Rewrite tf.Estimators in low tensorflow API.
```angular2html
def training_phase(bert_config, processor, tokenizer, label_list, predict_label_list):
    """
    :param processor:
    :param tokenizer:
    :param label_list:
    :param predict_label_list:
    :return:
    """
    """
     建training_data和dev_data的联合pipeline，返回train_init_op和dev_init_op来初始化
     """
    (input_ids, input_mask, segment_ids, label_ids, train_init_op, dev_init_op,
     num_train_steps, handle) = build_train_dev_data_pipeline(processor, tokenizer, label_list, predict_label_list)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    tf.logging.info("=========== Train and evaluate set are loaded ============")
    """
    建模型
    """
    (total_loss, logits, trans, pred_ids) = create_model(bert_config=bert_config, is_training=True,
                                                         input_ids=input_ids,
                                                         input_mask=input_mask, segment_ids=segment_ids,
                                                         labels=label_ids,
                                                         num_labels=len(label_list) + 1, use_one_hot_embeddings=False)
    tf.summary.scalar("total_loss", total_loss)
    tf.logging.info("================= Model is built ====================")
    """
    以下开始加载BERT，即用BERT的参数去初始化我新模型的权重，
    init_from_checkpoint即按assignment_map对应的变量来加载 
    """
    init_checkpoint = FLAGS.init_checkpoint
    tvars = tf.trainable_variables()
    # 加载BERT模型, 用Bert的参数来初始化模型
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                  init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    train_op = optimization.create_optimizer(
        total_loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
    tf.logging.info("================BERT are loaded to initiate and train_op is built===========")
    """
    设置三种评价指标, 每个指标包含两个element(scalar float Tensor, update_op)
    """
    (precision, recall, f1) = eval_phase(label_ids, pred_ids,len(label_list) + 1)
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    running_vars_initializer = tf.variables_initializer(var_list=running_vars) #初始化precision、recall、f1这些计算节点
    prec_scalar, prec_op = precision
    recall_scalar, recall_op = recall
    f1_scalar, f1_op = f1
    tf.logging.info("=================eval metrics are loaded=========================")
    """
    设置Savar为最多保存五个model
    """
    saver = tf.train.Saver(max_to_keep=5)

    merged = tf.summary.merge_all()

    tf.logging.info("==================Entering Session Running=========================")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #除了CRF层，其它层都被initialized了
        train_writer = tf.summary.FileWriter(FLAGS.output_dir , sess.graph)
        dev_writer = tf.summary.FileWriter(FLAGS.output_dir + '/eval')
        train_iterator_handle = sess.run(train_init_op.string_handle())
        dev_iterator_handle = sess.run(dev_init_op.string_handle())
        for step in range(num_train_steps):
            if step % 100 == 0:
                tf.logging.info("===============evaluate at %d step=============="%step)
                sess.run(running_vars_initializer)
                sess.run(dev_init_op.initializer)
                # while True:
                while True:
                    try:
                        # print(sess.run([label_ids, pred_ids], feed_dict={handle: dev_iterator_handle}))
                        summary,_,_,_ = sess.run([merged, prec_op, recall_op, f1_op], feed_dict={handle: dev_iterator_handle})
                    except tf.errors.OutOfRangeError:
                        break
                dev_writer.add_summary(summary, step)
                _precision, _recall, _f1 = sess.run([prec_scalar, recall_scalar, f1_scalar])
                print("At step {}, the precision is {:.2f}%,the recall is {:.2f}%,the f1 is {:.2f}%".format(step, _precision*100, _recall*100, _f1*100))
            else:
                if step % 1000 == 999:
                    tf.logging.info("===============save model at %d step==============" % step)
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ , _total_loss= sess.run([merged, train_op, total_loss],
                                          feed_dict={handle: train_iterator_handle},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
                    train_writer.add_summary(summary, step)
                    tf.logging.info("========== the total loss is %.5f ===============" %(_total_loss))
                    print('Adding run metadata for', step)
                    save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"), global_step=step)
                    print("Model saved in path: %s" % save_path)
                else:
                    # print(sess.run([pred_ids, label_ids], feed_dict={handle: train_iterator_handle}))
                    summary, _ = sess.run([merged, train_op], feed_dict={handle: train_iterator_handle})
                    train_writer.add_summary(summary, step)
        train_writer.close()
        dev_writer.close()

def predict_phase(bert_config, processor, tokenizer, label_list, predict_label_list):
    test_batch_size = 1000

    predict_examples, predict_examples_number = processor.get_test_examples(FLAGS.data_dir, test_batch_size)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(predict_examples, label_list,
                                             FLAGS.max_seq_length, tokenizer,
                                             predict_file, mode="test", predict_label_list=predict_label_list)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    tf.logging.info(" Character Number = %d", predict_examples_number)

    predict_dataset = get_file_based_dataset(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        batch_size=FLAGS.predict_batch_size)

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, predict_dataset.output_types, predict_dataset.output_shapes)
    predict_init_op = predict_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    # predict_iterator = predict_dataset.make_one_shot_iterator()
    # next_element = predict_iterator.get_next()

    tf.logging.info("*** Features ***")
    for name in sorted(next_element.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, next_element[name].shape))
    input_ids = next_element["input_ids"]
    input_mask = next_element["input_mask"]
    segment_ids = next_element["segment_ids"]
    label_ids = next_element["label_ids"]
    tf.logging.info("=========== Predict set are loaded ============")

    (total_loss, logits, trans, pred_ids) = create_model(bert_config=bert_config, is_training=False, input_ids=input_ids,
                          input_mask=input_mask, segment_ids=segment_ids, labels=label_ids,
                          num_labels=len(label_list) + 1, use_one_hot_embeddings=False)

    tf.logging.info("================= Model is built ====================")

    saver = tf.train.Saver()

    merged = tf.summary.merge_all()

    predict_result = []

    with tf.Session() as sess:
        global_step = 0
        saver.restore(sess, './output/msra_24_crf_result_dir/model.ckpt-26999')
        # sess.run(tf.global_variables_initializer())
        predict_writer = tf.summary.FileWriter(FLAGS.output_dir+'/predict' , sess.graph)
        predict_init_handle = sess.run(predict_init_op.string_handle())
        while True:
            try:
                tf.logging.info("======================= the %d step starts ==================="%global_step)
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, label, result= sess.run([merged, label_ids, pred_ids], feed_dict={handle:predict_init_handle},
                                                 options=run_options, run_metadata=run_metadata)
                predict_writer.add_summary(summary, global_step=global_step)
                predict_result.append(result)
                print("the label is", label)
                print("the result is", result)
                # result_to_pair(predict_examples, result)
                global_step += 1
            except tf.errors.OutOfRangeError:
                break
        predict_writer.close()

        # Create the Timeline object, and write it to a json
        # tl = timeline.Timeline(run_metadata.step_stats)
        # ctf = tl.generate_chrome_trace_format()
        # with open('timeline.json', 'w') as f:
        #     f.write(ctf)
```
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
#### Build an feedable data pipeline.
```angular2html
def build_train_dev_data_pipeline(processor, tokenizer, label_list, predict_label_list):

    """
    这里的代码参考至https://blog.csdn.net/briblue/article/details/80962728#commentBox，
    以及https://isaacchanghau.github.io/post/tensorflow_dataset_api/，对Dataset API的解释非常通透
    :param processor:
    :param tokenizer:
    :param label_list:
    :param predict_label_list:
    :return:
    """
    """
    加载训练集为Dataset对象
    """
    train_examples, train_words_num = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

    filed_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer,
        train_file, predict_label_list=predict_label_list)

    tf.logging.info("***** Running trainning*****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_dataset = get_file_based_dataset(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        batch_size=FLAGS.train_batch_size,
        is_training=True)

    """
    加载验证集为Dataset对象
    """
    dev_examples, dev_words_num = processor.get_dev_examples(FLAGS.data_dir, batch_num=20000)
    dev_file = os.path.join(FLAGS.output_dir, "dev.tf_record")

    filed_based_convert_examples_to_features(
        dev_examples, label_list, FLAGS.max_seq_length, tokenizer,
        dev_file, predict_label_list=predict_label_list)

    tf.logging.info("***** Running dev set*****")
    tf.logging.info("  Num examples = %d", len(dev_examples))
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    dev_dataset = get_file_based_dataset(
        input_file=dev_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False, #记住把验证集这里改成False，不然验证集会repeat而且shuffle
        batch_size=FLAGS.eval_batch_size)

    """
    为train和dev的数据集构造feedable迭代器，注意这里迭代器的四种形式：
    1、 单次 Iterator ，它最简单，但无法重用，无法处理数据集参数化的要求。 
    2、 可以初始化的 Iterator ，它可以满足 Dataset 重复加载数据，满足了参数化要求。 
    3、可重新初始化的 Iterator，它可以对接不同的 Dataset，也就是可以从不同的 Dataset 中读取数据。 
    4、可馈送的 Iterator，它可以通过 feeding 的方式，让程序在运行时候选择正确的 Iterator,它和可重新初始化的 Iterator 不同的地方就是它的数据在不同的 Iterator 切换时，可以做到不重头开始读取数据
    """
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = train_dataset.make_one_shot_iterator()
    dev_init_op = dev_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    tf.logging.info("*** Features ***")
    for name in sorted(next_element.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, next_element[name].shape))
    input_ids = next_element["input_ids"]
    input_mask = next_element["input_mask"]
    segment_ids = next_element["segment_ids"]
    label_ids = next_element["label_ids"]

    return (input_ids, input_mask, segment_ids, label_ids, train_init_op, dev_init_op,
            num_train_steps, handle)

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
