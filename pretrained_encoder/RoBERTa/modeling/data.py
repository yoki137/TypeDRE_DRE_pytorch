import csv
import json
import math
import os
import pickle
import random
from collections import defaultdict
from itertools import permutations
import logging

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

import logging

from models.BERTs.BERT import get_cuda

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of datasets."""

    def __init__(self, entity, input_ids, input_mask, segment_ids, mention_mask, label_id):
        self.entity = entity
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.mention_mask = mention_mask
        self.label_id = label_id


class DataProcessor(object):
    """Base class for datasets converters for sequence classification datasets sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this datasets set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


def tokenize(text, tokenizer, max_seq_length):
    D = ['[S1]', '[S2]', '[S3]', '[S4]', '[S5]', '[S6]', '[S7]', '[S8]', '[S9]', '[ENTITY]', '[/ENTITY]']
    textraw = text.split('\n')
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in D:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]

    mention_mask = np.zeros([max_seq_length])
    for i in range(len(text)):
        if i== max_seq_length-1:
            break
        if '[ENTITY]' in text[i]:
            mention_mask[i+1] = 1
    if sum(mention_mask)>0:
        mention_mask = mention_mask/sum(mention_mask)
    return text, mention_mask


def _truncate_seq_tuple(tokens_a, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a datasets file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        tokens_a, mention_mask = tokenize(example.text_a, tokenizer, max_seq_length)

        _truncate_seq_tuple(tokens_a, max_seq_length - 2)

        tokens = []
        segment_ids = []
        tokens.append("<s>")
        segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("</s>")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("entity: %s" % (example.text_b))
            logger.info("tokens: %s" % " ".join(
                [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "mention_mask: %s" % " ".join([str(x) for x in mention_mask]))
            logger.info("label_id: %s" % " ".join([str(x) for x in label_id]))

        features[-1].append(
            InputFeatures(
                entity=example.text_b,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                mention_mask=mention_mask,
                label_id=label_id))
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features



class bertsProcessor(DataProcessor):  # bert_s
    def __init__(self, src_file):
        random.seed(666)
        self.D = [[], [], []]

        id2spktokenS = ['pad', '[S1]', '[S2]', '[S3]', '[S4]', '[S5]', '[S6]', '[S7]', '[S8]', '[S9]']
        type2id= {"PER": 0, "GPE": 1, "ORG": 2, "STRING": 3, "VALUE": 4}
        for sid in range(3):
            with open(src_file + ["/train.json", "/dev.json", "/test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                entities = []
                labels = []
                for j in range(len(data[i][1])):
                    x = data[i][1][j]["x"]
                    x_type = data[i][1][j]["x_type"]
                    if "Speaker" not in x and x not in entities:
                        entities.append(x)
                        labels.append(type2id[x_type])
                    y = data[i][1][j]["y"]
                    y_type = data[i][1][j]["y_type"]
                    if "Speaker" not in y and y not in entities:
                        entities.append(y)
                        labels.append(type2id[y_type])
                new_d = '\n'.join(data[i][0]).lower()
                # for s in range(1, 10):
                #     new_d = new_d.replace("speaker "+str(s), id2spktokenS[s])
                for j in range(len(entities)):
                    tid = []
                    for k in range(5):
                        if k == labels[j]:
                            tid += [1]
                        else:
                            tid += [0]
                    d = [new_d.replace(entities[j].lower(), "[ENTITY]"+entities[j].lower()+"[/ENTITY]"), str(i)+":"+entities[j], tid]
                    self.D[sid] += [d]

        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[2], "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            guid = "%s-%s" % (set_type, i)
            text_a = data[i][0]
            text_b = data[i][1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b,  label=data[i][2]))

        return examples

class Dataset(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer):

        super(Dataset, self).__init__()

        self.data = None
        self.input_max_length = max_seq_length

        print('Reading datasets from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['datasets']
            print('load preprocessed datasets from {}.'.format(save_file))

        else:
            self.data = []

            bertsProcessor_class = bertsProcessor(src_file)
            if "train" in save_file:
                examples = bertsProcessor_class.get_train_examples(save_file)
            elif "dev" in save_file:
                examples = bertsProcessor_class.get_dev_examples(save_file)
            elif "test" in save_file:
                examples = bertsProcessor_class.get_test_examples(save_file)
            else:
                print("error")

            features = convert_examples_to_features(examples, max_seq_length, tokenizer)

            for f in features:
                self.data.append({
                    'entity': f[0].entity,
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'mention_mask': np.array(f[0].mention_mask),
                    'label_ids': np.array(f[0].label_id)
                })
            if "train" not in save_file:
                self.entity_dict = []
                for f in features:
                    did, entity = int(f[0].entity.split(":")[0]), f[0].entity.split(":")[1]
                    while did > len(self.entity_dict)-1:
                        self.entity_dict.append({})
                    self.entity_dict[did][entity] = ""
                with open(file=save_file.replace(".pkl", "_entity_dict.json"), mode="w", encoding="utf8") as f:
                    json.dump(self.entity_dict, f)
                print('save entity dictionary to {}.'.format(src_file + "/entity_dict.json"))
            # save datasets
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'datasets': self.data}, fw)
            print('finish reading {} and save preprocessed datasets to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


class Dataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, relation_num=5, max_length=512):
        super(Dataloader, self).__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.relation_num = relation_num

        self.order = list(range(self.length))

    def __iter__(self):
        # shuffle
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)]

        # begin
        input_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        segment_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        input_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        mention_mask = torch.FloatTensor(self.batch_size,self.max_length).cpu()
        label_ids = torch.Tensor(self.batch_size, self.relation_num).cpu()

        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch)
            entities = [[] for i in range(self.batch_size)]
            for mapping in [input_ids, segment_ids, input_mask, mention_mask, label_ids]:
                if mapping is not None:
                    mapping.zero_()


            for i, example in enumerate(minibatch):
                entity, mini_input_ids, mini_segment_ids, mini_input_mask, mini_mention_mask, mini_label_ids = \
                    example['entity'], example['input_ids'], example['segment_ids'], example['input_mask'], example['mention_mask'], example['label_ids']

                word_num = mini_input_ids.shape[0]
                relation_num = mini_label_ids.shape[0]

                entities[i].append(entity)
                input_ids[i, :word_num].copy_(torch.from_numpy(mini_input_ids))
                segment_ids[i, :word_num].copy_(torch.from_numpy(mini_segment_ids))
                input_mask[i, :word_num].copy_(torch.from_numpy(mini_input_mask))
                mention_mask[i, :word_num].copy_(torch.from_numpy(mini_mention_mask))
                label_ids[i, :relation_num].copy_(torch.from_numpy(mini_label_ids))

            context_word_mask = input_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()

            yield {
                   'entities': entities,
                   'input_ids': get_cuda(input_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'segment_ids': get_cuda(segment_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'input_masks': get_cuda(input_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'mention_mask':get_cuda(mention_mask[:cur_bsz,:batch_max_length].contiguous()),
                   'label_ids': get_cuda(label_ids[:cur_bsz, :self.relation_num].contiguous())
                   }
