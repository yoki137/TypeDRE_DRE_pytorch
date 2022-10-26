import logging
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange

from evaluate import caculate
from model.relation_extraction.RelationExtraction_RoBERTa import RelationExtraction_RoBERTa
from pretrained_encoder.BERT.modeling import tokenization
from pretrained_encoder.BERT.modeling.BERT import BertConfig
from model.relation_extraction.RelationExtraction_BERT import RelationExtraction_BERT
from pretrained_encoder.RoBERTa.modeling.configuration_roberta import RobertaConfig
from pretrained_encoder.RoBERTa.modeling.tokenization_roberta import RobertaTokenizer
from model.relation_extraction.relation_extraction_data import REDataset, REDataset4F1c, REDataLoader
from optimization import BERTAdam



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

n_class = 36

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def accuracy(out, labels):
    out = out.reshape(-1)
    out = 1 / (1 + np.exp(-out))
    return np.sum((out > 0.5) == (labels > 0.5)) / 36


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan


def f1_eval(logits, labels_all):
    def getpred(result, T1=0.5, T2=0.4):
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret += [r]
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0

        for i in range(len(data)):
            for id in data[i]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1

            for id in devp[i]:
                if id != 36:
                    all_sys += 1

        precision = 1 if all_sys == 0 else correct_sys / all_sys
        recall = 0 if correct_gt == 0 else correct_sys / correct_gt
        f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return f_1

    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))

    labels = []
    for la in labels_all:
        label = []
        for i in range(36):
            if la[i] == 1:
                label += [i]
        if len(label) == 0:
            label = [36]
        labels += [label]
    assert (len(labels) == len(logits))

    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2 / 100.)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2 / 100.

    return bestf_1, bestT2


def get_logits4eval(model, dataloader, savefile, device):
    model.eval()
    logits_all = []

    desc = "Iteration"
    if "dev" in savefile:
        desc = "DEV"
    elif "test" in savefile:
        desc = "TEST"
    for batch in tqdm(dataloader, desc=desc):
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        input_masks = batch['input_masks'].to(device)
        label_ids = batch['label_ids'].to(device)
        obj_mask = batch['obj_mask'].to(device)
        subj_mask = batch['subj_mask'].to(device)
        pair_entitytype_id = batch['pair_entitytype_id'].to(device)

        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, obj_mask=obj_mask, subj_mask=subj_mask, pair_entitytype_id=pair_entitytype_id, labels=label_ids)

        logits = logits.detach().cpu().numpy()
        for i in range(len(logits)):
            logits_all += [logits[i]]

    with open(savefile, "w") as f:
        for i in range(len(logits_all)):
            for j in range(len(logits_all[i])):
                f.write(str(logits_all[i][j]))
                if j == len(logits_all[i]) - 1:
                    f.write("\n")
                else:
                    f.write(" ")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="dataset/DialogRE",
                        type=str,
                        required=False,
                        help="The input datasets dir. Should contain the .json for the task.")
    parser.add_argument("--typed_data_dir",
                        default="dataset/DialogRE_typed",
                        type=str,
                        required=False,
                        help="The input datasets dir that contains .json file of the test set generated by entity typing module")
    parser.add_argument("--encoder_type",
                        default="BERT",
                        type=str,
                        required=False,
                        help="The type of pre-trained model.")
    parser.add_argument("--config_file",
                        default="pretrained_encoder/BERT/bert_config.json",
                        type=str,
                        required=False,
                        help="The config json file corresponding to the pre-trained model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--merges_file",
                        default="pretrained-encoder/RoBERTa/merges.txt",
                        type=str,
                        help="The merges file that the RoBERTa model was trained on.")
    parser.add_argument("--vocab_file",
                        default="pretrained_encoder/BERT/vocab.txt",
                        type=str,
                        required=False,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default="pretrained_encoder/BERT/pytorch_model.bin",
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained model).")
    parser.add_argument("--do_lower_case",
                        default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set and the test set using the ground truth type of entities.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on test set using the predicted types from entity typing module.")
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=random.randint(0, 9999),
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume the training.")
    parser.add_argument("--f1eval",
                        default=True,
                        action='store_true',
                        help="Whether to use f1 for dev evaluation during training.")
    parser.add_argument('--layer_num',
                        type=int,
                        default=2,
                        help="The number of layer in Multi-head Attention.")
    parser.add_argument('--window_size',
                        type=int,
                        default=7,
                        help="The window size of mask in Multi-head Attention.")

    args = parser.parse_args()

    logger.info("****************************************************************************************")
    logger.info('*********    ' + "Relation Extraction".rjust(44, ' '))
    logger.info("****************************************************************************************")
    for k, v in vars(args).items():
        logger.info('*********    '+k.rjust(32, ' ') + ' = ' + str(v))
    logger.info("****************************************************************************************")


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    if args.encoder_type not in ["BERT", "RoBERTa"]:
        raise ValueError("Encoder type must BERT or RoBERTa.")

    if args.encoder_type == "BERT":
        config = BertConfig.from_json_file(args.config_file)
    else:
        config = RobertaConfig.from_json_file(args.config_file)

    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, config.max_position_embeddings))

    if os.path.exists(args.output_dir) and 'model.pt' in os.listdir(args.output_dir):
        if args.do_train and not args.resume:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.encoder_type == "BERT":
        tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    else:
        tokenizer = RobertaTokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file)
        special_tokens_dict = {
            'additional_special_tokens': ['[SUBJ-SPK]', '[/SUBJ-SPK]', '[OBJ-SPK]', '[/OBJ-SPK]', '[SUBJ-PER]', '[/SUBJ-PER]', '[OBJ-PER]', '[/OBJ-PER]',
                                          '[SUBJ-GPE]', '[/SUBJ-GPE]', '[OBJ-GPE]', '[/OBJ-GPE]', '[SUBJ-ORG]', '[/SUBJ-ORG]', '[OBJ-ORG]', '[/OBJ-ORG]',
                                          '[SUBJ-STRING]', '[/SUBJ-STRING]', '[OBJ-STRING]', '[/OBJ-STRING]', '[SUBJ-VALUE]', '[/SUBJ-VALUE]', '[OBJ-VALUE]', '[/OBJ-VALUE]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    train_set = None
    num_train_steps = None
    if args.do_train:
        train_set = REDataset(src_file=args.data_dir, save_file=args.data_dir + "/train_" + args.encoder_type + ".pkl",
                            max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                            encoder_type=args.encoder_type)
        num_train_steps = int(
            len(train_set) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        train_loader = REDataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True, max_length=args.max_seq_length)

    if args.encoder_type == "BERT":
        model = RelationExtraction_BERT(config, args.window_size, args.layer_num)
    else:
        model = RelationExtraction_RoBERTa(config, args.layer_num, args.window_size)

    if args.init_checkpoint is not None:
        if args.encoder_type == "BERT":
            model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'), strict=False)
            model.roberta.resize_token_embeddings(len(tokenizer))

    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]

    optimizer = BERTAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    train_record = None

    if args.do_train:
        dev_set = REDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_" + args.encoder_type + ".pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class, encoder_type=args.encoder_type)
        dev_loader = REDataLoader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False, max_length=args.max_seq_length)

        best_metric = 0
        train_record = []

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        logger.info("  Seed = %d", args.seed)

        epoch_trange = trange(int(args.num_train_epochs), leave=False, desc="Epoch")
        for epoch in epoch_trange:
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_loader, leave=False, desc="Iteration")):
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                obj_mask = batch['obj_mask'].to(device)
                subj_mask = batch['subj_mask'].to(device)
                pair_entitytype_id = batch['pair_entitytype_id'].to(device)
                label_ids = batch['label_ids'].to(device)

                loss, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, subj_mask=subj_mask, obj_mask=obj_mask, pair_entitytype_id=pair_entitytype_id, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            logits_all = []
            labels_all = []
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                label_ids = batch['label_ids'].to(device)
                obj_mask = batch['obj_mask'].to(device)
                subj_mask = batch['subj_mask'].to(device)
                pair_entitytype_id = batch['pair_entitytype_id'].to(device)

                with torch.no_grad():
                    tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, subj_mask=subj_mask, obj_mask=obj_mask, pair_entitytype_id=pair_entitytype_id, labels=label_ids)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                for i in range(len(logits)):
                    logits_all += [logits[i]]
                for i in range(len(label_ids)):
                    labels_all.append(label_ids[i])

                tmp_eval_accuracy = accuracy(logits, label_ids.reshape(-1))

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            if args.do_train:
                result = {'eval_loss': eval_loss,
                          'global_step': global_step,
                          'loss': tr_loss / nb_tr_steps}
            else:
                result = {'eval_loss': eval_loss}

            if args.f1eval:
                eval_f1, eval_T2 = f1_eval(logits_all, labels_all)
                result["f1"] = eval_f1
                result["T2"] = eval_T2

            epoch_trange.clear()
            logger.info("*************** Epoch " + str(epoch).rjust(3, " ") + " ***************")
            for key in result.keys():
                logger.info("  %s = %s", key, str(result[key]))

            record = {"epoch": epoch, "loss": result['loss'], "eval_loss": result['eval_loss'], "f1": result['f1']}
            train_record.append(record)

            if args.f1eval:
                if eval_f1 >= best_metric:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    best_metric = eval_f1
            else:
                if eval_accuracy >= best_metric:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    best_metric = eval_accuracy
            logger.info("***** BEST METRIC NOW ：" + str(best_metric) + " *****")

        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        os.remove(os.path.join(args.output_dir, "model_best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if 'model_best.pt' in os.listdir(args.output_dir) and 'model.pt' in os.listdir(args.output_dir):
        os.remove(os.path.join(args.output_dir, "model_best.pt"))

    if args.do_eval:
        # for f1
        dev_set = REDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_" + args.encoder_type + ".pkl",
                          max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                          encoder_type=args.encoder_type)
        dev_loader = REDataLoader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False, max_length=args.max_seq_length)

        test_set = REDataset(src_file=args.data_dir, save_file=args.data_dir + "/test_" + args.encoder_type + ".pkl",
                             max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                             encoder_type=args.encoder_type)
        test_loader = REDataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False, max_length=args.max_seq_length)

        # for f1c
        devc_set = REDataset4F1c(src_file=args.data_dir,
                               save_file=args.data_dir + "/devc_" + args.encoder_type + ".pkl",
                               max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                               encoder_type=args.encoder_type)
        devc_loader = REDataLoader(dataset=devc_set, batch_size=args.eval_batch_size, shuffle=False, max_length=args.max_seq_length)
        testc_set = REDataset4F1c(src_file=args.data_dir,
                                save_file=args.data_dir + "/testc_" + args.encoder_type + ".pkl",
                                max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                                encoder_type=args.encoder_type)
        testc_loader = REDataLoader(dataset=testc_set, batch_size=args.eval_batch_size, shuffle=False, max_length=args.max_seq_length)

        get_logits4eval(model, dev_loader, os.path.join(args.output_dir, "logits_dev.txt"), device)
        get_logits4eval(model, test_loader, os.path.join(args.output_dir, "logits_test.txt"), device)
        get_logits4eval(model, devc_loader, os.path.join(args.output_dir, "logits_devc.txt"), device)
        get_logits4eval(model, testc_loader, os.path.join(args.output_dir, "logits_testc.txt"), device)

    if args.do_test:
        # typed test set
        if not os.path.exists(args.typed_data_dir + "/test.json"):
            raise ValueError("Missing test.json from entity typing module.")
        if not os.path.exists(os.path.join(args.output_dir, "logits_dev.txt")):
            dev_set = REDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_" + args.encoder_type + ".pkl",
                                max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                                encoder_type=args.encoder_type)
            dev_loader = REDataLoader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False,
                                      max_length=args.max_seq_length)
            get_logits4eval(model, dev_loader, os.path.join(args.output_dir, "logits_dev.txt"), device)
        test_set = REDataset(src_file=args.typed_data_dir,
                             save_file=args.typed_data_dir + "/test_typed_" + args.encoder_type + ".pkl",
                             max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                             encoder_type=args.encoder_type)
        test_loader = REDataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False,
                                   max_length=args.max_seq_length)
        testc_set = REDataset4F1c(src_file=args.typed_data_dir,
                                  save_file=args.typed_data_dir + "/testc_typed_" + args.encoder_type + ".pkl",
                                  max_seq_length=args.max_seq_length, tokenizer=tokenizer, n_class=n_class,
                                  encoder_type=args.encoder_type)
        testc_loader = REDataLoader(dataset=testc_set, batch_size=args.eval_batch_size, shuffle=False,
                                    max_length=args.max_seq_length)
        get_logits4eval(model, test_loader, os.path.join(args.output_dir, "logits_test_typed.txt"), device)
        get_logits4eval(model, testc_loader, os.path.join(args.output_dir, "logits_testc_typed.txt"), device)

    if args.do_eval or args.do_test:
        caculate(args, train_record, model)

if __name__ == "__main__":
    main()
