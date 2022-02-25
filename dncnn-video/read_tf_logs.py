import gc
import os
from glob import glob
import sys
import pdb
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

import utils

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument('--log_dirPath_parent', dest='log_dirPath_parent')
parser.add_argument('--plot_dirPath', dest='plot_dirPath')
args = parser.parse_args()

# pdb.set_trace()

def get_varLists(log_fName):
    varLists = dict()
    # pdb.set_trace()
    for event in tf.train.summary_iterator(log_fName):
        for value in event.summary.value:
            if value.tag not in varLists:
                varLists[value.tag] = []

            varLists[value.tag].append(value.simple_value)


    return varLists

def concat_varLists_multiple(varLists_multiple):
    
    num_varLists = len(varLists_multiple) 
    varLists_concat = dict()
    for varName in varLists_multiple[0].keys():
        varLists_concat[varName] = []
        for i in range(num_varLists):
            varLists_concat[varName] = varLists_concat[varName] + varLists_multiple[i][varName]

    return varLists_concat


def read_varLists_from_logDir(log_dirPath):

    log_fList = glob(log_dirPath+'*')
    log_fList.sort(key=os.path.getmtime)
    print("\n".join(log_fList))


    varLists_multiple = []
    for i in range(len(log_fList)):
        log_fName = log_fList[i]
        
        varLists = get_varLists(log_fName)
        varLists_multiple.append(varLists)

        print("Id: {}, log_fName: {} ".format(i, log_fName))
        # print_varLists(varLists)


    varLists_concat = concat_varLists_multiple(varLists_multiple)

    return varLists_concat


def print_varLists(varLists):
    for varName in varLists.keys():
        print(varName+' :')
        print(varLists[varName])

def save_allPlots_auto(varLists, plot_dirPath):  

    utils.makeDirInPath(plot_dirPath)

    for varName in varLists.keys():
        plt.plot(varLists[varName])
        plt.title(varName)
        # plt.xlabel('Batches')
        # plt.ylabel('Loss')
        plt.savefig(plot_dirPath + varName + '_plot.png')
        plt.close()

def save_select_plots(varLists, plot_dirPath):  

    utils.makeDirInPath(plot_dirPath)

    num_epochs = len(varLists['eval_psnr'])
    num_batches = len(varLists['train_psnr'])
    
    epoch_list = np.asarray(range(num_epochs))
    batch_list = np.asarray(range(num_batches))
    batch_list_normalized = num_epochs*batch_list/num_batches

    # plt.plot( varLists['epoch'], varLists['train_psnr'], label='train_psnr')
    # , lw=1, marker='.')
    plt.plot( batch_list_normalized, varLists['train_psnr'], label='train_psnr')
    plt.plot( epoch_list, varLists['eval_psnr'], label='eval_psnr', lw=2, marker='o')
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid()
    plt.legend(loc='lower right')

    plt.savefig(plot_dirPath + 'psnr_plot.png')
    plt.close()


if __name__ == '__main__':

    varLists_concat_train = read_varLists_from_logDir(args.log_dirPath_parent+'train/')
    varLists_concat_eval = read_varLists_from_logDir(args.log_dirPath_parent+'eval/')

    varLists_concat_all = varLists_concat_train
    varLists_concat_all.update(varLists_concat_eval)

    save_allPlots_auto(varLists_concat_all, args.plot_dirPath)

    save_select_plots(varLists_concat_all, args.plot_dirPath)

