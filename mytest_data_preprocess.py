import os
import codecs

output_dir = "/home/public/BERT-BiLSTM-CRF-NER"
# output_dir = "C:\\Users\\Lenovo\\PycharmProjects\\BERT-BiLSTM-CRF-NER-master"
preprocess_data_file = os.path.join(output_dir, "preprocess_mytest.txt")
result_dir = os.path.join(output_dir, "NERdata")
result_file = os.path.join(result_dir, "mytest1.txt")

def text_to_line(writer):
    with codecs.open(preprocess_data_file) as f:
        words = []
        for line in f:
            new_line = ''
            words = list(line.strip())
            for word in words:
                if word == ' ':
                    word = '，'
                new_line += word + ' ' + 'O' + '\n'
            writer.write(new_line + '\n')

with codecs.open(result_file, 'w', encoding='utf-8') as writer:
    text_to_line(writer)
