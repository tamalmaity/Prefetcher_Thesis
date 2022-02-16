import numpy as np
import os
import argparse
from collections import Counter
from datetime import datetime
#import tensorflow as tf
import json
#from get_available_gpu import mask_unused_gpus
from scipy.stats import entropy
import random
from mha import MultiHeadedAttention
from models import HierarchicalLSTM
import csv
import lzma
import glob
import pythia

import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

np.set_printoptions(precision=3)
#tf.random.set_seed(0)
tf.set_random_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_batches(pc_in, page_in, offset_in, page_out, offset_out, batch_size, num_steps, pc_localization):
    n_batches = len(pc_in)//batch_size
    for ii in range(num_steps, len(pc_in), batch_size):
        batch_pc_in = []
        batch_page_in = []
        batch_offset_in = []
        batch_page_out = []
        batch_offset_out = []
        if ii+batch_size >= len(pc_in):
            break
        for jj in range(batch_size):
            batch_pc_in.append(pc_in[ii+jj-num_steps+1:ii+jj+1])
            batch_page_in.append(page_in[ii+jj-num_steps+1:ii+jj+1])
            batch_offset_in.append(offset_in[ii+jj-num_steps+1:ii+jj+1])
            if pc_localization:
                batch_page_out.append(page_out[ii+jj])
                batch_offset_out.append(offset_out[ii+jj])
            else:
                batch_page_out.append(page_in[ii+jj+1])
                batch_offset_out.append(offset_in[ii+jj+1])

        yield batch_pc_in, batch_page_in, batch_offset_in, batch_page_out, batch_offset_out


def get_output_pc_localization(pc_in, page_in, offset_in):
    last_pc_index = {}
    pc_correlated_index = {}
    page_out = np.zeros_like(page_in)
    offset_out = np.zeros_like(offset_in)
    for i in range(len(pc_in)):
        pc = pc_in[i]
        if pc not in last_pc_index:
            last_pc_index[pc] = i
        else:
            last_index = last_pc_index[pc]
            page_out[last_index] = page_in[i]
            offset_out[last_index] = offset_in[i]
            last_pc_index[pc] = i
    return page_out, offset_out


def process_line(line):
    split = line.split(' ')
    #split = line.strip().split(', ')
    return int(split[1][2:], 16), int(split[2][2:], 16)
    #return int(split[0]), int(split[1]), int(split[2], 16), int(split[3], 16), split[4] == '1'

name1 = "/ip_addr_105_output_1.txt"
name2 = "/ip_addr_105_output_2.txt"

write_file1 = open("/home/tamalmaity/mlsysarch/PYTHIA" + name1, "w")
write_file1.write("curr_pc, curr_page, next_page, page_pred, curr_offset, next_offset, offset_pred" + "\n")


write_file2 = open("/home/tamalmaity/mlsysarch/PYTHIA" + name2, "w")
write_file2.write("curr_pc, curr_page, next_page, curr_offset, next_offset" + "\n")


def build_and_train_network(benchmark, args):
    directory = benchmark

    unique_pcs = {'oov': 0}
    unique_pages = {'oov': 0}
    rev_unique_pcs = {0 : 'oov'}
    rev_unique_pages = {0 : 'oov'}
    pc_in = []
    page_in = []
    offset_in = []
    #addr_in = []
    '''
    with open(benchmark) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            #row = row.split(',')
            #print(row)
            pc, page, offset = int(row[2], 16), int(row[3], 16)>>12, (int(row[3], 16)>>6)&0x3f #since 4KB page and since 64B blocks (offset can only start with these multiples)
            
            '''
    load_trace = "/home/tamalmaity/data/LoadTraces/"
    workloads = ["spec06"] #, ["gap", "spec06", "spec17"]
    for ext in workloads: 
        #path = load_trace+ext
        path = "/home/tamalmaity/mlsysarch/IP_ADDR/ip_addr_105.out.xz"
        count=0
        #for file in glob.iglob(path+'/*astar*.txt.xz'):
        for file in glob.iglob(path):
        #for cnt in range(3):
            #file = path+"/623.xalancbmk-s" + str(3+cnt) + ".txt.xz"
            count+= 1
            print(count, file)
            with lzma.open(file, mode='rt') as f:
                for line in f:
                    pline = process_line(line)
                    #pc, page,offset = pline[2], pline[3]>>12, (pline[3]>>6)&0x3f
                    addr = pline[1]
                    pc, page, offset = pline[0], pline[1]>>12, (pline[1]>>6)&0x3f
                    if pc not in unique_pcs:
                        value = len(unique_pcs)
                        unique_pcs[pc] = value # dict for storing all unique pcs
                        rev_unique_pcs[value] = pc
                    if page not in unique_pages:
                        value = len(unique_pages)
                        unique_pages[page] = value
                        rev_unique_pages[value] = page

                    pc_in.append(unique_pcs[pc]) #list of PCs according to dict value like 1,2,2,3,4,1,... (to start from 1 -> oov)
                    page_in.append(unique_pages[page])
                    offset_in.append(offset) #can only be from 0 to 63
                    #addr_in.append(addr)

    pc_in = np.array(pc_in)
    page_in = np.array(page_in)
    offset_in = np.array(offset_in)
    #addr_in = np.array(addr_in)

    pc_localization = args.pc_localization
    if pc_localization:
        page_out, offset_out = get_output_pc_localization(pc_in, page_in, offset_in)
    else:
        page_out, offset_out = np.zeros_like(page_in), np.zeros_like(offset_in)

    train_split = 0.8
    test_split = 0.8
    dataset_all = 1.0
    train_split = int(pc_in.shape[0]*train_split)
    test_split = int(pc_in.shape[0]*test_split)
    dataset_all = int(pc_in.shape[0]*dataset_all)

    train_pc_in = pc_in[:train_split]
    test_pc_in = pc_in[test_split:dataset_all]
    train_page_in = page_in[:train_split]
    test_page_in = page_in[test_split:dataset_all]
    train_offset_in = offset_in[:train_split]
    test_offset_in = offset_in[test_split:dataset_all]   #till here 80-20 for training and testing
    #test_addr_in = addr_in[test_split:dataset_all]
    # only for pc localization
    train_page_out = page_out[:train_split]
    test_page_out = page_out[test_split:dataset_all]
    train_offset_out = offset_out[:train_split]
    test_offset_out = offset_out[test_split:dataset_all]

    pc_vocab_size = int(max(pc_in)+1)
    page_vocab_size = page_out_vocab_size = int(max(page_in)+1)

    # all
    epoch = 1 #500
    batch_size = args.batch_size
    num_layers = args.lstm_layer
    num_steps = args.step_size
    lstm_size = args.lstm_size
    pc_embed_size = args.pc_embed_size
    page_embed_size = args.page_embed_size
    offset_embed_size = args.offset_embed_size
    offset_embed_size_internal = args.offset_embed_size_internal
    keep_prob = args.keep_ratio
    decay_rate = args.learning_rate_decay
    use_pc_history = args.use_pc_history
    complete_loss = args.complete_loss
    complete_embedding = args.complete_embedding

    learning_rate = args.learning_rate
    learning_rate_adjust = decay_rate != 1

    print('\n')
    print('Global prediction, baseline STMS')
    print('Dataset stats...')
    print('Benchmark: {}'.format(directory))
    print('Dataset size: {}'.format(len(page_in)))
    print('Split point, train: {}'.format(train_split))
    print('Split point, test: {}'.format(test_split))
    print('Batch: {}'.format(batch_size))
    print('Epoch: {}'.format(epoch))

    print('\n')
    print('Hypers...')
    print('Number of steps: {}'.format(num_steps))
    print('PC vocab size: {}'.format(pc_vocab_size))
    print('Page vocab size: {}'.format(page_vocab_size))
    print('Page out vocab size: {}'.format(page_out_vocab_size))
    print('Learning rate: {}'.format(learning_rate))
    print('Learning rate adjust: {}'.format(learning_rate_adjust))
    print('Learning rate decay: {}'.format(decay_rate))
    print('PC embed size: {}'.format(pc_embed_size))
    print('Page embed size: {}'.format(page_embed_size))
    print('Offset embed size: {}'.format(offset_embed_size))
    print('LSTM size: {}'.format(lstm_size))
    print('Number of layers: {}'.format(num_layers))
    print('Keep ratio: {}'.format(keep_prob))

    print('\n')

    correct_offset = []

    #mask_unused_gpus()

    lstm = HierarchicalLSTM(pc_vocab_size, page_vocab_size, page_out_vocab_size, batch_size, num_steps, learning_rate, pc_embed_size, page_embed_size, offset_embed_size, offset_embed_size_internal, lstm_size, num_layers)

    losses = []
    accs = [0]
    baseline_dict = {}
    baseline_up = 0
    baseline_down = 0
    baseline_acc = 0
    last_change = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epoch):
            print("Epoch: {}/{}...".format(e+1, epoch))
            start_time = datetime.now()
            train_accs = []
            train_page_accs = []
            train_offset_accs = []
            train_loss = []
            train_page_loss = []
            train_offset_loss = []
            train_learning_rate = []

            #print("here1")
            if learning_rate_adjust and e-last_change >= 5:
                if accs[-1] - accs[-4] < 0.005:
                    learning_rate /= decay_rate
                    last_change = e
                    if learning_rate < 1e-4:
                        learning_rate = 1e-4
                    print('\tLearning rate dacayed to {:.7f}'.format(learning_rate))

            ## PYTHIA ##
            counter = 0
            offset_jump_dict = {}
            scooby_actions = []

            #print("here2..")
            for ii, (batch_pc_in, batch_page_in, batch_offset_in, batch_page_out, batch_offset_out) in enumerate(get_batches(train_pc_in, train_page_in, train_offset_in, train_page_out, train_offset_out, batch_size, num_steps, pc_localization), 1):
                #if (ii%1000==0):
                    #print("Loop 1:", ii )
                feed = {lstm.pc_in: batch_pc_in,
                        lstm.page_out_init: batch_page_out,
                        lstm.offset_out_init: batch_offset_out,
                        lstm.pl_page_in: batch_page_in,
                        lstm.pl_offset_in: batch_offset_in,
                        lstm.keep_prob: keep_prob,
                        lstm.learning_rate: learning_rate}
                _, batch_loss, batch_acc, batch_page_acc, batch_offset_acc, batch_learning_rate = sess.run([lstm.optimizer1, lstm.loss, lstm.overall_accuracy, lstm.page_accuracy, lstm.offset_accuracy, lstm.optimizer._lr], feed_dict = feed)
                train_loss.append(batch_loss)
                train_accs.append(batch_acc)
                train_page_accs.append(batch_page_acc)
                train_offset_accs.append(batch_offset_acc)
                train_learning_rate.append(batch_learning_rate)

                for j in range (len(batch_page_in)-1):
                    pred_from_page = batch_page_in[j][-1]
                    actual_page = batch_page_in[j+1][-1]

                    pred_from_offset = batch_offset_in[j][-1]
                    actual_offset = batch_offset_in[j+1][-1]
                    jump = actual_offset - pred_from_offset
                    if jump in offset_jump_dict:
                        offset_jump_dict[jump] += 1
                    else:
                        offset_jump_dict[jump] = 1


                    '''
                    tmp = []
                    test_pc = batch_pc_in[j][0]
                    test_page = rev_unique_pages[pred_from_page]
                    test_offset = pred_from_offset
                    test_addr = ( test_page << 12) + ( test_offset << 6)
                    reinforcement.invoke_prefetcher(test_pc, test_addr, 0, 0, tmp)

                    counter += 1

                    if (counter%1000000 == 0):
                        print("Reached" + str(counter))
                    '''



                if e == 0:
                    count = 0
                    for (page_in, offset_in, page_out, offset_out) in zip(batch_page_in, batch_offset_in, batch_page_out, batch_offset_out):
                        count+= 1
                        #if (count%1000==0):
                            #print ("Loop 2:", count)
                        baseline_dict[(page_in[-1], offset_in[-1])] = (page_out, offset_out)



            ### PYTHIA ###
            rev_sort_offset_jump_dict = sorted(offset_jump_dict.items(), key = lambda x : x[1], reverse = 1)
            for keys in sorted(rev_sort_offset_jump_dict):
                scooby_actions.append(keys[0])
                if len(scooby_actions) == 15:
                    break

            print(scooby_actions)
            print('\n\n')

            ####



            #print("here3")
            time_elapsed = datetime.now() - start_time
            print("\tTrain Loss: {:.3f}...".format(np.mean(train_loss)))
            print("\tTrain Accuracy: {:.3f}...".format(np.mean(train_accs)))
            print("\tTrain Page Accuracy: {:.3f}...".format(np.mean(train_page_accs)))
            print("\tTrain Offset Accuracy: {:.3f}...".format(np.mean(train_offset_accs)))
            print("\tTrain Learning Rate: {:.7f}...".format(np.mean(train_learning_rate)))
            print('\tTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

            start_time = datetime.now()
            test_accs = []
            test_page_accs = []
            test_offset_accs = []
            test_loss = []
            test_page_loss = []
            test_offset_loss = []
            test_page_preds = []
            test_offset_preds = []
            test_baseline_page_preds = []
            test_attns = []



            # PYTHIA #
            reinforcement = pythia.Scooby(scooby_actions)
            pref_addr = []
            all_test_cnt = 0
            correct_predictions = 0



            for ii, (batch_pc_in, batch_page_in, batch_offset_in, batch_page_out, batch_offset_out) in enumerate(get_batches(test_pc_in, test_page_in, test_offset_in, test_page_out, test_offset_out, batch_size, num_steps, pc_localization), 1):
                #if (ii%1000==0):
                    #print("Loop 3 :", ii)
                feed = {lstm.pc_in: batch_pc_in,
                        lstm.page_out_init: batch_page_out,
                        lstm.offset_out_init: batch_offset_out,
                        lstm.pl_page_in: batch_page_in,
                        lstm.pl_offset_in: batch_offset_in,
                        lstm.keep_prob: 1.0}
                batch_loss, batch_acc, batch_page_acc, batch_offset_acc, batch_page_preds, batch_offset_preds, attns = sess.run([lstm.loss, lstm.overall_accuracy, lstm.page_accuracy, lstm.offset_accuracy, lstm.page_preds, lstm.offset_preds, lstm.attns], feed_dict = feed)
                test_loss.append(batch_loss)
                test_accs.append(batch_acc)
                test_page_accs.append(batch_page_acc)
                test_offset_accs.append(batch_offset_acc)
                test_page_preds.append(batch_page_preds)
                test_offset_preds.append(batch_offset_preds)
                test_attns.append([entropy(a) for a in attns])

                for j in range (len(batch_page_in)-1):
                    pred_from_page = batch_page_in[j][-1]
                    actual_page = batch_page_in[j+1][-1]
                    pred_page = batch_page_preds[j+1] ####changes

                    pred_from_offset = batch_offset_in[j][-1]
                    actual_offset = batch_offset_in[j+1][-1]
                    pred_offset = batch_offset_preds[j+1] ####changes

                    # PYTHIA ########################
                    tmp = []
                    test_pc = batch_pc_in[j][0]
                    test_page = rev_unique_pages[pred_from_page]
                    test_offset = pred_from_offset
                    test_addr = ( test_page << 12) + ( test_offset << 6)
                    reinforcement.invoke_prefetcher(test_pc, test_addr, 0, 0, tmp)

                    pref_page = pref_offset = 0
                    if not tmp:
                        pref_page = pref_offset = -1
                        tmp.append(-1)
                    else:
                        pref_page = tmp[-1]>>12
                        pref_offset = (tmp[-1]>>6)&0x3f

                    pref_addr.append(tmp)

                    if pref_page == actual_page and pref_offset == actual_offset:
                        correct_predictions += 1
                    counter += 1
                    all_test_cnt += 1

                    write_file2.write(str(test_pc) + " " + str(test_page) + " " + str(pref_page) + " " + str(test_offset) + " " + str(pref_offset) + "\n")


                    if (counter%1000000 == 0):
                        print("Reached" + str(counter))
                    ###############################

                    write_file1.write(str(batch_pc_in[j][0]) + " " + str(rev_unique_pages[pred_from_page]) + " " + str(rev_unique_pages[actual_page]) + " " + str(rev_unique_pages[pred_page]) + " " + str(pred_from_offset) + " " + str(actual_offset) + " " + str(pred_offset) + "\n")

                if e == 0:
                    count = 0
                    for (page_in, offset_in, page_out, offset_out) in zip(batch_page_in, batch_offset_in, batch_page_out, batch_offset_out):
                        count+= 1
                        #if (count%1000==0):
                            #print ("Loop 4:", count)
                        if (page_in[-1], offset_in[-1]) in baseline_dict:
                            tgt = baseline_dict[(page_in[-1], offset_in[-1])]
                            if tgt == (page_out, offset_out):
                                baseline_up += 1
                        baseline_down += 1
                    baseline_acc = float(baseline_up) / float(baseline_down)

            time_elapsed = datetime.now() - start_time
            acc = np.mean(test_accs)
            accs.append(acc)
            #print("here4")
            print("\tBaseline accuracy: {:.3f}...".format(baseline_acc))
            print("\tTest Loss: {:.3f}...".format(np.mean(test_loss)))
            print("\tTest Accuracy: {:.3f}...".format(np.mean(test_accs)))
            print("\tTest Page Accuracy: {:.3f}...".format(np.mean(test_page_accs)))
            print("\tTest Offset Accuracy: {:.3f}...".format(np.mean(test_offset_accs)))
            print("\tTest Attention Entropy: {:.3f}...".format(np.mean(np.concatenate(test_attns))))
            print('\tTime elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
            print('\tBest acc: {:.3f}'.format(max(accs)))

            print("Accuracy of Pythia is : {} \n\n".format(correct_predictions/all_test_cnt))

            print(reinforcement.stats)



    ############## PYTHIA ###############
    '''
    pref_addr = []
    size = len(pref_addr)
    reinforcement = pythia.Scooby()
    cnt = 0
    for (test_pc, test_addr) in zip(test_pc_in, test_addr_in):
        reinforcement.invoke_prefetcher(test_pc, test_addr, 0, 0, pref_addr)
        ###########
        if size != len(pref_addr):
            print(test_addr, end = " ")
            for i in range(len(pref_addr)-size):
                if i+1!=len(pref_addr)-size:
                    print(pref_addr[size+i], end = ",")
                else:
                    print(pref_addr[size+i])
            size = len(pref_addr)
        else:
            print(test_addr, '-')
        ##########
        if size == len(pref_addr):
            pref_addr.append(-1)
        
        size += 1
        cnt += 1

        test_page = test_addr>>12
        test_offset = (test_addr>>6)&0x3f
        pref_page = pref_offset = 0
        if pref_addr[-1] == -1:
            pref_page = pref_offset = -1
        else:
            pref_page = pref_addr[-1]>>12
            pref_offset = (pref_addr[-1]>>6)&0x3f

        write_file2.write(str(test_pc) + " " + str(test_page) + " " + str(pref_page) + " " + str(test_offset) + " " + str(pref_offset) + "\n")

        if (cnt%50000 == 0):
            print("Reached" + str(cnt))

    correct_predictions = 0
    for i in range(len(test_addr_in)-1):
        if test_addr_in[i+1] == pref_addr[i]:
            correct_predictions += 1

    print("Accuracy of Pythia is : {}".format(correct_predictions/(dataset_all - test_split)))
    '''

    


def main():
    parser = argparse.ArgumentParser(description='LSTM attention.')
    parser.add_argument("--benchmark", help="benchmark name", type=str) # specify the directory of input data
    parser.add_argument("--trace_length", help="trace length", type=str) #not used
    parser.add_argument("--page_embed_size", help="page embedding size", type=int) #128
    parser.add_argument("--pc_embed_size", help="pc embedding size", type=int) #64
    parser.add_argument("--offset_embed_size", help="offset embedding size", type=int) #128*100
    parser.add_argument("--offset_embed_size_internal", help="offset embedding size internal", type=int) #128
    parser.add_argument("--lstm_size", help="lstm size", type=int) #256
    parser.add_argument("--keep_ratio", help="keep ratio", type=float) #dropout ratio = 0.8
    parser.add_argument("--step_size", help="step size", type=int) #16 -> seq length
    parser.add_argument("--learning_rate", help="learning rate", type=float) #0.001
    parser.add_argument("--learning_rate_decay", help="learning rate decay", type=float) #2
    parser.add_argument("--complete_loss", help="complete loss", type=int) #not used
    parser.add_argument("--complete_embedding", help="complete embedding", type=int) #not used
    parser.add_argument("--batch_size", help="batch size", type=int) #512
    parser.add_argument("--lstm_layer", help="lstm layer", type=int) #1
    parser.add_argument("--use_pc_history", help="use pc history", type=int) #use_pc_seq = 1
    parser.add_argument("--pc_localization", help="pc localization or global", type=int) # 0: predict global stream 1: predict PC localized stream , here 0

    args = parser.parse_args()

    build_and_train_network(args.benchmark, args)


if __name__ == '__main__':
    main()
