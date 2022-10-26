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

from pretrained_encoder.BERT.modeling.BERT import get_cuda

head_markers = ['[SPK]', '[PER]', '[GPE]', '[ORG]', '[STRING]', '[VALUE]']
type_markers = ['[SUBJ-SPK]', '[/SUBJ-SPK]', '[OBJ-SPK]', '[/OBJ-SPK]', '[SUBJ-PER]', '[/SUBJ-PER]', '[OBJ-PER]', '[/OBJ-PER]', \
     '[SUBJ-GPE]', '[/SUBJ-GPE]', '[OBJ-GPE]', '[/OBJ-GPE]', '[SUBJ-ORG]', '[/SUBJ-ORG]', '[OBJ-ORG]', '[/OBJ-ORG]', \
     '[SUBJ-STRING]', '[/SUBJ-STRING]', '[OBJ-STRING]', '[/OBJ-STRING]', '[SUBJ-VALUE]', '[/SUBJ-VALUE]', '[OBJ-VALUE]',
     '[/OBJ-VALUE]']

id2spktokenS = ['pad', '[S1]', '[S2]', '[S3]', '[S4]', '[S5]', '[S6]', '[S7]', '[S8]', '[S9]']
id2spktokenE = ['pad', '[/S1]', '[/S2]', '[/S3]', '[/S4]', '[/S5]', '[/S6]', '[/S7]', '[/S8]', '[/S9]']

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
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
        self.text_c = text_c
        self.label = label


class InputFeatures(object):
    """A single set of features of datasets."""

    def __init__(self, input_ids, input_mask, segment_ids, subj_mask, obj_mask, pair_entitytype_id, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.subj_mask = subj_mask
        self.obj_mask = obj_mask
        self.pair_entitytype_id = pair_entitytype_id
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


def tokenize(text, x, y, tokenizer, max_length, offset):
    textraw = text.split('\n')
    for delimiter in type_markers:
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
        if t in type_markers:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]

    subj_mask = np.zeros([max_length])
    obj_mask = np.zeros([max_length])
    for i in range(len(text)):
        if i == max_length - 4 -offset:
            break
        if x in text[i]:
            subj_mask[i+1] = 1
        if y in text[i]:
            obj_mask[i+1] = 1

    return text, subj_mask, obj_mask


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

def convert_examples_to_features_BERT(examples, max_seq_length, tokenizer):
    """Loads a datasets file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        tokens_a, subj_mask, obj_mask = tokenize(example.text_a, example.text_b, example.text_c, tokenizer, max_seq_length, 0)

        subj_token = example.text_b
        obj_token = example.text_c

        subj_tokid , obj_tokid = -1, -1
        for i in range(len(head_markers)):
            if head_markers[i] in subj_token:
                subj_tokid = i
            if head_markers[i] in obj_token:
                obj_tokid = i

        pair_entitytype_id = np.zeros([2, 6], dtype=int)
        pair_entitytype_id[0][subj_tokid] = 1
        pair_entitytype_id[1][obj_tokid] = 1

        _truncate_seq_tuple(tokens_a, max_seq_length-5)

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        tokens.append(subj_token)
        segment_ids.append(1)
        subj_mask[len(tokens)-1] = -1

        tokens.append(obj_token)
        segment_ids.append(1)
        obj_mask[len(tokens)-1] = -1

        tokens.append("[SEP]")
        segment_ids.append(1)

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
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("subj_mask: %s" % " ".join([str(x) for x in subj_mask]))
            logger.info("obj_mask: %s" % " ".join([str(x) for x in obj_mask]))
            logger.info("subj_tokentype_id: %s" % " ".join([str(x) for x in pair_entitytype_id[0]]))
            logger.info("obj_tokentype_id: %s" % " ".join([str(x) for x in pair_entitytype_id[1]]))
            logger.info("label_id: %s" % " ".join([str(x) for x in label_id]))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                subj_mask=subj_mask,
                obj_mask=obj_mask,
                pair_entitytype_id=pair_entitytype_id,
                label_id=label_id))
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features


def convert_examples_to_features_RoBERTa(examples, max_seq_length, tokenizer):
    """Loads a datasets file into a list of `InputBatch`s."""

    print("#examples", len(examples))

    features = [[]]
    for (ex_index, example) in enumerate(examples):
        tokens_a, subj_mask, obj_mask = tokenize(example.text_a, example.text_b, example.text_c, tokenizer, max_seq_length, 1)

        subj_token = example.text_b
        obj_token = example.text_c

        subj_tokid , obj_tokid = -1, -1
        for i in range(len(head_markers)):
            if head_markers[i] in subj_token:
                subj_tokid = i
            if head_markers[i] in obj_token:
                obj_tokid = i

        pair_entitytype_id = np.zeros([2, 6],dtype=int)
        pair_entitytype_id[0][subj_tokid] = 1
        pair_entitytype_id[1][obj_tokid] = 1

        _truncate_seq_tuple(tokens_a, max_seq_length - 6)


        tokens = []
        segment_ids = []
        tokens.append("<s>")
        segment_ids.append(0)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)

        tokens.append('</s>')
        segment_ids.append(0)
        tokens.append('</s>')
        segment_ids.append(0)

        tokens.append(subj_token)
        segment_ids.append(1)
        subj_mask[len(tokens) - 1] = -1
        tokens.append(obj_token)
        segment_ids.append(1)
        obj_mask[len(tokens) - 1] = -1

        tokens.append('</s>')
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == len(input_mask)
        assert len(input_mask) == len(segment_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(1)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("subj_mask: %s" % " ".join([str(x) for x in subj_mask]))
            logger.info("obj_mask: %s" % " ".join([str(x) for x in obj_mask]))
            logger.info("subj_tokentype_id: %s" % " ".join([str(x) for x in pair_entitytype_id[0]]))
            logger.info("obj_tokentype_id: %s" % " ".join([str(x) for x in pair_entitytype_id[1]]))
            logger.info("label_id: %s" % " ".join([str(x) for x in label_id]))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                subj_mask=subj_mask,
                obj_mask=obj_mask,
                pair_entitytype_id=pair_entitytype_id,
                label_id=label_id))
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]
    print('#features', len(features))
    return features

def is_speaker(a):
    a = a.split()
    return len(a) == 2 and a[0] == "speaker" and a[1].isdigit()

class REProcessor(DataProcessor):
    def __init__(self, src_file, n_class):

        self.D = [[], [], []]
        random.seed(666)

        for sid in range(3):
            path = src_file + ["/train.json", "/dev.json", "/test.json"][sid]
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            if sid == 0:
                random.shuffle(data)
            for i in range(len(data)):
                # insert start token and end token of turn
                max_spkid = 0
                d = [i for i in data[i][0]]
                for j in range(len(data[i][0])):
                    turn = d[j].split(':')
                    turn[1] = ''.join(turn[1:]).lower()
                    spk_ids = [int(n) for n in turn[0].replace('Speaker', '').split(',')]
                    if len(spk_ids) > 1:
                        for spkid in spk_ids:
                            if spkid > max_spkid:
                                max_spkid = spkid
                            turn[1] = id2spktokenS[spkid] + turn[1] + id2spktokenE[spkid]
                        d[j] = turn[1]
                    else:
                        spkid = spk_ids[0]
                        if spkid > max_spkid:
                            max_spkid = spkid
                        d[j] = id2spktokenS[spkid] + turn[1] + id2spktokenE[spkid]

                for j in range(len(data[i][1])):
                    new_d = ' '.join(d)
                    rid = []
                    for k in range(n_class):
                        if k + 1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]
                    x = data[i][1][j]["x"].lower()
                    y = data[i][1][j]["y"].lower()
                    x_type = data[i][1][j]["x_type"]
                    y_type = data[i][1][j]["y_type"]

                    if is_speaker(x):
                        tokenS, tokenE = id2spktokenS[int(x.split(' ')[1])], id2spktokenE[int(x.split(' ')[1])]
                        new_d = new_d.replace(tokenS, "[SUBJ-SPK]").replace(tokenE, "[/SUBJ-SPK]")
                        x = "[SUBJ-SPK]"
                    else:
                        S, E= "[SUBJ-"+x_type+"]", "[/SUBJ-"+x_type+"]"
                        new_d = new_d.replace(x, S+x+E)
                        x = S
                    if is_speaker(y):
                        tokenS, tokenE = id2spktokenS[int(y.split(' ')[1])], id2spktokenE[int(y.split(' ')[1])]
                        new_d = new_d.replace(tokenS, "[OBJ-SPK]").replace(tokenE, "[/OBJ-SPK]")
                        y = "[OBJ-SPK]"
                    else:
                        S, E= "[OBJ-"+y_type+"]", "[/OBJ-"+y_type+"]"
                        new_d = new_d.replace(y, S+y+E)
                        y = S
                    for e in range(1, len(id2spktokenE)):
                        new_d = new_d.replace(id2spktokenE[e], '')
                    for s in range(1, len(id2spktokenS)):
                        new_d = new_d.replace(id2spktokenS[s], 'speaker ' + str(s) + ":")
                    new_d = [new_d, x, y, rid]
                    self.D[sid] += [new_d]
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
            text_c = data[i][2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))

        return examples

class REProcessor4F1c(DataProcessor):
    def __init__(self, src_file, n_class):

        self.D = [[], [], []]
        random.seed(666)
        for sid in range(1, 3):
            path = src_file + ["/dev.json", "/test.json"][sid - 1]
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf8") as f:
                data = json.load(f)
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    rid = []
                    for k in range(n_class):
                        if k + 1 in data[i][1][j]["rid"]:
                            rid += [1]
                        else:
                            rid += [0]

                    for l in range(1, len(data[i][0]) + 1):
                        d = [i for i in data[i][0][:l]]
                        for m in range(l):
                            turn = d[m].split(':')
                            turn[1] = ''.join(turn[1:]).lower()
                            spk_ids = [int(n) for n in turn[0].replace('Speaker', '').split(',')]
                            if len(spk_ids) > 1:
                                for spkid in spk_ids:
                                    turn[1] = id2spktokenS[spkid] + turn[1] + id2spktokenE[spkid]
                                d[m] = turn[1]
                            else:
                                spkid = spk_ids[0]
                                d[m] = id2spktokenS[spkid] + turn[1] + id2spktokenE[spkid]

                        d = ' '.join(d)
                        x = data[i][1][j]["x"].lower()
                        y = data[i][1][j]["y"].lower()
                        x_type = data[i][1][j]["x_type"]
                        y_type = data[i][1][j]["y_type"]

                        if is_speaker(x):
                            tokenS, tokenE = id2spktokenS[int(x.split(' ')[1])], id2spktokenE[int(x.split(' ')[1])]
                            d = d.replace(tokenS, "[SUBJ-SPK]").replace(tokenE, "[/SUBJ-SPK]")
                            x = "[SUBJ-SPK]"
                        else:
                            S, E = "[SUBJ-" + x_type + "]", "[/SUBJ-" + x_type + "]"
                            d = d.replace(x, S + x + E)
                            x = S
                        if is_speaker(y):
                            tokenS, tokenE = id2spktokenS[int(y.split(' ')[1])], id2spktokenE[int(y.split(' ')[1])]
                            d = d.replace(tokenS, "[OBJ-SPK]").replace(tokenE, "[/OBJ-SPK]")
                            y = "[OBJ-SPK]"
                        else:
                            S, E = "[OBJ-" + y_type + "]", "[/OBJ-" + y_type + "]"
                            d = d.replace(y, S + y + E)
                            y = S

                        for e in range(1, len(id2spktokenE)):
                            d = d.replace(id2spktokenE[e], '')
                        for s in range(1, len(id2spktokenS)):
                            d = d.replace(id2spktokenS[s], 'speaker '+str(s)+":")
                        d = [d, x, y, rid]
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
            text_c = data[i][2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, text_c=text_c, label=data[i][3]))

        return examples

class REDataset(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type):

        super(REDataset, self).__init__()

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

            processor = REProcessor(src_file, n_class)
            if "train" in save_file:
                examples = processor.get_train_examples(save_file)
            elif "dev" in save_file:
                examples = processor.get_dev_examples(save_file)
            elif "test" in save_file:
                examples = processor.get_test_examples(save_file)
            else:
                print("error")

            if encoder_type == "BERT":
                features = convert_examples_to_features_BERT(examples, max_seq_length, tokenizer)
            else:
                features = convert_examples_to_features_RoBERTa(examples, max_seq_length, tokenizer)

            for f in features:

                self.data.append({
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'subj_mask': np.array(f[0].subj_mask),
                    'obj_mask': np.array(f[0].obj_mask),
                    'pair_entitytype_id': np.array(f[0].pair_entitytype_id),
                    'label_ids': np.array(f[0].label_id),
                })
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

class REDataset4F1c(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type):

        super(REDataset4F1c, self).__init__()

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

            processor = REProcessor4F1c(src_file, n_class)
            if "dev" in save_file:
                examples = processor.get_dev_examples(save_file)
            elif "test" in save_file:
                examples = processor.get_test_examples(save_file)
            else:
                print("error")

            if encoder_type == "BERT":
                features = convert_examples_to_features_BERT(examples, max_seq_length, tokenizer)
            else:
                features = convert_examples_to_features_RoBERTa(examples, max_seq_length, tokenizer)

            for f in features:
                self.data.append({
                    'input_ids': np.array(f[0].input_ids),
                    'segment_ids': np.array(f[0].segment_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'subj_mask': np.array(f[0].subj_mask),
                    'obj_mask': np.array(f[0].obj_mask),
                    'pair_entitytype_id': np.array(f[0].pair_entitytype_id),
                    'label_ids': np.array(f[0].label_id),
                })
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

class REDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, max_length=512):
        super(REDataLoader, self).__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length

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
        obj_mask = torch.Tensor(self.batch_size, self.max_length).cpu()
        subj_mask = torch.Tensor(self.batch_size, self.max_length).cpu()
        pair_entitytype_id = torch.Tensor(self.batch_size, 2, 6).cpu()
        label_ids = torch.Tensor(self.batch_size, 36).cpu()

        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch)

            for mapping in [input_ids, segment_ids, input_mask, obj_mask, subj_mask, label_ids]:
                if mapping is not None:
                    mapping.zero_()


            for i, example in enumerate(minibatch):
                mini_input_ids, mini_segment_ids, mini_input_mask, mini_subj_mask, mini_obj_mask, mini_pair_entitytype_id, mini_label_ids = \
                    example['input_ids'], example['segment_ids'], example['input_mask'], example['subj_mask'], example['obj_mask'], example['pair_entitytype_id'], example['label_ids']

                word_num = mini_input_ids.shape[0]
                relation_num = mini_label_ids.shape[0]

                input_ids[i, :word_num].copy_(torch.from_numpy(mini_input_ids))
                segment_ids[i, :word_num].copy_(torch.from_numpy(mini_segment_ids))
                input_mask[i, :word_num].copy_(torch.from_numpy(mini_input_mask))
                subj_mask[i, :word_num].copy_(torch.from_numpy(mini_subj_mask))
                obj_mask[i, :word_num].copy_(torch.from_numpy(mini_obj_mask))
                pair_entitytype_id[i, :2, :6].copy_(torch.from_numpy(mini_pair_entitytype_id))
                label_ids[i, :relation_num].copy_(torch.from_numpy(mini_label_ids))

            context_word_mask = input_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()

            yield {'input_ids': get_cuda(input_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'segment_ids': get_cuda(segment_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'input_masks': get_cuda(input_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'subj_mask':get_cuda(subj_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'obj_mask': get_cuda(obj_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'pair_entitytype_id': get_cuda(pair_entitytype_id[:cur_bsz, :2, :6].contiguous()),
                   'label_ids': get_cuda(label_ids[:cur_bsz, :36].contiguous())
                   }
