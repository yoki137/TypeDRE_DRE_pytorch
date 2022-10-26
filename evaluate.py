import json
import logging
import time
import datetime
import numpy as np
import os

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def getresult(fn):
    result = []
    with open(fn, "r") as f:
        l = f.readline()
        while l:
            l = l.strip().split()
            for i in range(len(l)):
                l[i] = float(l[i])
            result += [l]
            l = f.readline()
    result = np.asarray(result)
    return list(1 / (1 + np.exp(-result)))

def getpredict(result, T1 = 0.5, T2 = 0.4):
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
        result[i] = r
    return result

def evaluate(devp, data):
    index = 0
    correct_sys, all_sys = 0, 0
    correct_gt = 0
    
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            for id in data[i][1][j]["rid"]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[index]:
                        correct_sys += 1
            for id in devp[index]:
                if id != 36:
                    all_sys += 1
            index += 1

    precision = correct_sys/all_sys if all_sys != 0 else 1
    recall = correct_sys/correct_gt if correct_gt != 0 else 0
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1


def evaluate_f1c(devp, data):
    index = 0
    precisions = []
    recalls = []
    
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            correct_sys, all_sys = 0, 0
            correct_gt = 0
            
            x = data[i][1][j]["x"].lower().strip()
            y = data[i][1][j]["y"].lower().strip()
            t = {}
            for k in range(len(data[i][1][j]["rid"])):
                if data[i][1][j]["rid"][k] != 36:
                    t[data[i][1][j]["rid"][k]] = data[i][1][j]["t"][k].lower().strip()

            l = set(data[i][1][j]["rid"]) - set([36])

            ex, ey = False, False
            et = {}
            for r in range(36):
                et[r] = r not in l

            for k in range(len(data[i][0])):
                o = set(devp[index]) - set([36])
                e = set()
                if x in data[i][0][k].lower():
                    ex = True
                if y in data[i][0][k].lower():
                    ey = True
                if k == len(data[i][0])-1:
                    ex = ey = True
                    for r in range(36):
                        et[r] = True
                for r in range(36):
                    if r in t:
                        if t[r] != "" and t[r] in data[i][0][k].lower():
                            et[r] = True
                    if ex and ey and et[r]:
                        e.add(r)
                correct_sys += len(o & l & e)
                all_sys += len(o & e)
                correct_gt += len(l & e)
                index += 1

            precisions += [correct_sys/all_sys if all_sys != 0 else 1]
            recalls += [correct_sys/correct_gt if correct_gt != 0 else 0]

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f_1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0

    return precision, recall, f_1


def caculate(args, train_record=None, model=None):
    path = args.output_dir

    result_path = os.path.join(path, "result.txt")
    dev = os.path.join(args.data_dir, "dev.json")
    test = os.path.join(args.data_dir, "test.json")
    f1dev = os.path.join(path, "logits_dev.txt")

    if args.do_eval:
        f1test = os.path.join(path, "logits_test.txt")
        f1cdev = os.path.join(path, "logits_devc.txt")
        f1ctest = os.path.join(path, "logits_testc.txt")

    if args.do_test:
        f1test_typed = os.path.join(path, "logits_test_typed.txt")
        f1ctest_typed = os.path.join(path, "logits_testc_typed.txt")

    with open(dev, "r", encoding='utf8') as f:
        datadev = json.load(f)
    with open(test, "r", encoding='utf8') as f:
        datatest = json.load(f)
    for i in range(len(datadev)):
        for j in range(len(datadev[i][1])):
            for k in range(len(datadev[i][1][j]["rid"])):
                datadev[i][1][j]["rid"][k] -= 1
    for i in range(len(datatest)):
        for j in range(len(datatest[i][1])):
            for k in range(len(datatest[i][1][j]["rid"])):
                datatest[i][1][j]["rid"][k] -= 1

    bestT2 = bestf_1 = 0
    for T2 in range(51):
        dev = getresult(f1dev)
        devp = getpredict(dev, T2=T2/100.)
        precision, recall, f_1 = evaluate(devp, datadev)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2/100.

    writer = open(result_path, "w")

    logger.info("best T2: {}".format(bestT2))
    writer.write("best T2: %s\n" % (bestT2))
    writer.write("****************************************************************************************\n")
    writer.write("evaluate results:\n")
    result = []

    if args.do_eval:
        dev = getresult(f1dev)
        devp = getpredict(dev, T2=bestT2)
        precision, recall, f_1 = evaluate(devp, datadev)
        result += [precision, recall, f_1]
        logger.info("dev (P={}, R={}, F1={})".format(precision, recall, f_1))
        writer.write("dev (P=%s, R=%s, F1=%s)\n" % (precision, recall, f_1))


        dev = getresult(f1cdev)
        devp = getpredict(dev, T2=bestT2)
        precision, recall, f_1c = evaluate_f1c(devp, datadev)
        result += [precision, recall, f_1c]
        logger.info("dev (P_c={}, R_c={}, F1_c={})".format(precision, recall, f_1c))
        writer.write("dev (P_c=%s, R_c=%s, F1_c=%s)\n" % (precision, recall, f_1c))


        test = getresult(f1test)
        testp = getpredict(test, T2=bestT2)
        precision, recall, f_1 = evaluate(testp, datatest)
        result += [precision, recall, f_1]
        logger.info("test (P={}, R={}, F1={})".format(precision, recall, f_1))
        writer.write("test (P=%s, R=%s, F1=%s)\n" % (precision, recall, f_1))

        test = getresult(f1ctest)
        testp = getpredict(test, T2=bestT2)
        precision, recall, f_1c = evaluate_f1c(testp, datatest)
        result += [precision, recall, f_1c]
        logger.info("test (P_c={}, R_c={}, F1_c={})".format(precision, recall, f_1c))
        writer.write("test (P_c=%s, R_c=%s, F1_c=%s)\n" % (precision, recall, f_1c))

    if args.do_test:
        test = getresult(f1test_typed)
        testp = getpredict(test, T2=bestT2)
        precision, recall, f_1 = evaluate(testp, datatest)
        result += [precision, recall, f_1]
        logger.info("test_typed (P={}, R={}, F1={})".format(precision, recall, f_1))
        writer.write("test_typed (P=%s, R=%s, F1=%s)\n" % (precision, recall, f_1))

        test = getresult(f1ctest_typed)
        testp = getpredict(test, T2=bestT2)
        precision, recall, f_1c = evaluate_f1c(testp, datatest)
        result += [precision, recall, f_1c]
        logger.info("test_typed (P_c={}, R_c={}, F1_c={})".format(precision, recall, f_1c))
        writer.write("test_typed (P_c=%s, R_c=%s, F1_c=%s)\n" % (precision, recall, f_1c))

    result_txt = [str(f) for f in result]
    writer.write(' '.join(result_txt))

    writer.write("****************************************************************************************\n")

    for k, v in vars(args).items():
        writer.write('*********    ' + k.rjust(32, ' ') + ' = ' + str(v) + "\n")

    writer.write("****************************************************************************************\n")

    if model is not None:
        writer.write(str(model))
        writer.write("****************************************************************************************\n")

    if train_record is not None:
        writer.write("\n****************************************************************************************\n")
        writer.write("training record:\n")
        writer.write("Epoch".rjust(4, ' ') + "Loss".rjust(32, ' ') + "Eval Loss".rjust(32, ' ') + "F1".rjust(32, ' ') + "\n")
        for record in train_record:
            writer.write(str(record["epoch"]).rjust(4, ' ') + str(record["loss"]).rjust(32, ' ') + str(record["eval_loss"]).rjust(32, ' ') + str(record["f1"]).rjust(32, ' ') + "\n")

    writer.close()

    writer = open("result-" + datetime.datetime.now().strftime("%m%d") + ".txt", "a")

    result_txt = [str(f*100) for f in result]

    writer.write("\nModel:" + args.output_dir+"\nSeed:" + str(args.seed)+"\n")
    writer.write(' '.join(result_txt)+"\n")
    writer.close()


