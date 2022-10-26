import json
import logging
import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange

from model.entity_typing.entity_typing_data import ETDataset, ETDataLoader
from model.entity_typing.EntityTyping import EntityTyping
from pretrained_encoder.RoBERTa.modeling.configuration_roberta import RobertaConfig
from pretrained_encoder.RoBERTa.modeling.tokenization_roberta import RobertaTokenizer
from optimization import BERTAdam



os.environ['CUDA_VISIBLE_DEVICES'] = '1'

n_class = 5

# Entity marker
EntityMarkers = ['[ENTITY]', '[/ENTITY]']
# Type
id2type = ["PER", "GPE", "ORG", "STRING", "VALUE"]



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def accuracy(out, labels):
    out = out.argmax(axis=1)
    labels = labels.argmax(axis=1)
    return (out == labels).sum()


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

def get_logits4eval(model, dataloader, device, dataset, data_dir=None, output_dir=None):
    model.eval()
    examples = 0
    eval_accuracy = 0
    dict = None
    if data_dir is not None and output_dir is not None:
        with open(data_dir, "r", encoding="utf8") as f:
            dict = json.load(f)
    for batch in tqdm(dataloader, desc="Iteration"):
        entities = batch['entities']
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        input_masks = batch['input_masks'].to(device)
        mention_mask = batch['mention_mask'].to(device)
        label_ids = batch['label_ids'].to(device)

        with torch.no_grad():
            tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, mention_mask=mention_mask, labels=label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        examples += input_ids.size(0)
        if dict is not None:
            ids = logits.argmax(axis=1)
            for i in range(len(ids)):
                id, entity = int(entities[i][0].split(":")[0]), entities[i][0].split(":")[1]
                dict[id][entity] = id2type[ids[i]]

    if dict is not None:
        with open(file=os.path.join(output_dir, "test_entity_dict.json"), mode="w", encoding="utf8") as f:
            json.dump(dict, f)

    eval_accuracy = eval_accuracy / examples

    logger.info(dataset+":"+str(eval_accuracy))
    return eval_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="dataset/DialogRE",
                        type=str,
                        required=False,
                        help="The input datasets dir. Should contain the .json for the task.")
    parser.add_argument("--config_file",
                        default="pretrained_encoder/RoBERTa/config.json",
                        type=str,
                        required=False,
                        help="The config json file corresponding to the pre-trained model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file",
                        default="pretrained_encoder/RoBERTa/vocab.json",
                        type=str,
                        required=False,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument("--merges_file",
                        default="pretrained_encoder/RoBERTa/merges.txt",
                        type=str,
                        help="The merges file that the RoBERTa model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default="pretrained_encoder/RoBERTa/pytorch_model.bin",
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
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=30.0,
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
                        default=6,
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

    args = parser.parse_args()

    logger.info("****************************************************************************************")
    logger.info('*********    ' + "Entity Typing".rjust(40, ' '))
    logger.info("****************************************************************************************")
    for k, v in vars(args).items():
        logger.info('*********    ' + k.rjust(32, ' ') + ' = ' + str(v))
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

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

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


    tokenizer = RobertaTokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file)
    special_tokens_dict = {
        'additional_special_tokens': EntityMarkers}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    train_set = None
    num_train_steps = None
    if args.do_train:
        train_set = ETDataset(src_file=args.data_dir, save_file=args.data_dir + "/train_entity_typing.pkl",
                            max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        num_train_steps = int(
            len(train_set) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        train_loader = ETDataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True, relation_num=n_class, max_length=args.max_seq_length)

    model = EntityTyping(config)

    if args.init_checkpoint is not None:
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

    no_decay = ['bias', 'gamma', 'beta']
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


    if args.do_train:
        dev_set = ETDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_entity_typing.pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        dev_loader = ETDataLoader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False,relation_num=n_class, max_length=args.max_seq_length)

        best_metric = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        epoch_trange = trange(int(args.num_train_epochs), leave=False, desc="Epoch")
        for e in epoch_trange:
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_loader, leave=False, desc="Iteration")):
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                input_mask = batch['input_masks'].to(device)
                mention_mask = batch['mention_mask'].to(device)
                label_ids = batch['label_ids'].to(device)

                loss, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, mention_mask=mention_mask, labels=label_ids)
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
                mention_mask = batch['mention_mask'].to(device)
                label_ids = batch['label_ids'].to(device)

                with torch.no_grad():
                    tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, mention_mask=mention_mask, labels=label_ids)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                for i in range(len(logits)):
                    logits_all += [logits[i]]
                for i in range(len(label_ids)):
                    labels_all.append(label_ids[i])

                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            if args.do_train:
                result = {'eval_loss': eval_loss,
                          'global_step': global_step,
                          'loss': tr_loss / nb_tr_steps,
                          'eval_accuracy':eval_accuracy }
            else:
                result = {'eval_loss': eval_loss}

            epoch_trange.clear()
            logger.info("*************** Epoch "+str(e).rjust(3, " ")+" ***************")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))

            if eval_accuracy >= best_metric:
                torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                best_metric = eval_accuracy

            logger.info("***** BEST METRIC NOW ：" + str(best_metric) + " *****")

        model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_best.pt")))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
        os.remove(os.path.join(args.output_dir, "model_best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))

    if args.do_eval:
        # for f1
        dev_set = ETDataset(src_file=args.data_dir, save_file=args.data_dir + "/dev_entity_typing.pkl",
                          max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        dev_loader = ETDataLoader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False, relation_num=n_class, max_length=args.max_seq_length)
        test_set = ETDataset(src_file=args.data_dir, save_file=args.data_dir + "/test_entity_typing.pkl", max_seq_length=args.max_seq_length, tokenizer=tokenizer)
        test_loader = ETDataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False,
                                     relation_num=n_class, max_length=args.max_seq_length)

        dev_result = get_logits4eval(model, dev_loader, device, "DEV")
        test_restult = get_logits4eval(model, test_loader, device, "TEST", args.data_dir+"/test_entity_dict.json", args.output_dir)
        writer = open(os.path.join(args.output_dir,"result.txt"), "w")

        writer.write("****************************************************************************************\n")
        writer.write("Dev ACC:"+str(dev_result)+"\n")
        writer.write("TEST ACC:" + str(test_restult))
        writer.write(str(dev_result)+" " + str(test_restult))
        writer.write("\n****************************************************************************************\n")
        for k, v in vars(args).items():
            writer.write('*********    ' + k.rjust(32, ' ') + ' = ' + str(v) + "\n")
        writer.write("****************************************************************************************\n")
        writer.write(str(model))
        writer.close()


if __name__ == "__main__":
    main()
