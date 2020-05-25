# implement the Training main function for the C-Decision scheme with Binary Feedback

import matplotlib.pyplot as plt
from BS_brain import Agent
from Environment import *
import pickle
import os
import random
import numpy as np
import tensorflow as tf
from Sim_Config import RL_Config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # use GPU 0 to run this code


def main():
    """
    Train the agent
    """

    # number of different trainings settings
    train_num = [1, 2, 3, 4, 5,
                 6, 7, 8, 9, 10]
    # number of binary feedback values
    num_feedback_set = [1, 2, 3,  6,  9,
                        15, 18, 27, 36, 45]
    # discount factor
    gamma_set = [0.05]
    # mini-batch size
    batch_set = [512]

    # weight for the sum rate of V2V in the reward
    v2v_weight = 1
    # weight for the sum rate of V2I in the reward
    v2i_weight = 0.1

    num_train_settings = 5

    # start training
    for train_loop in range(num_train_settings):

        # set the current random seed for training
        train_seed_sequence = 1001
        random.seed(train_seed_sequence)
        np.random.seed(train_seed_sequence)
        tf.set_random_seed(train_seed_sequence)

        # set values for current simulation
        curr_RL_Config = RL_Config()

        train_show_tra = '-----Start the Number -- ' + str(train_num[train_loop]) + ' -- training -----!'
        print(train_show_tra)

        # set key parameters for this train
        num_feedback = num_feedback_set[train_loop]
        gamma = gamma_set[0]
        batch_size = batch_set[0]
        curr_RL_Config.set_key_value(num_feedback, gamma, batch_size, v2v_weight, v2i_weight)

        # start the Environment
        Env = start_env()

        # run the training process
        [Train_Loss, Train_Q_mean, Train_Q_max_mean] = run_train(Env, curr_RL_Config)

        # save the train results
        save_flag = save_train_results(Train_Loss, Train_Q_mean, Train_Q_max_mean, curr_RL_Config)
        if save_flag:
            print('RL Training is finished!')


def start_env():
    # start the environment simulator
    """
    Generate the Environment
    """
    up_lanes = [3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]
    left_lanes = [3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)

    Env.new_random_game(Env.n_Veh)

    return Env


def run_train(Env, curr_RL_Config):
    # run the training process
    """
    Run the Training Process
    """
    # parameters to construct a BS Agent object
    Num_neighbor = Env.n_Neighbor
    Num_d2d = Env.n_Veh
    Num_CH = Env.n_RB

    Num_D2D_feedback = curr_RL_Config.Num_Feedback

    BS_Agent = Agent(Num_d2d, Num_CH, Num_neighbor, Num_D2D_feedback, Env, curr_RL_Config)

    Num_Episodes = curr_RL_Config.Num_Episodes
    Num_Train_Step = curr_RL_Config.Num_Train_Steps
    # get the train loss
    [Train_Loss, Train_Q_mean, Train_Q_max_mean] = BS_Agent.train(Num_Episodes, Num_Train_Step)

    return [Train_Loss, Train_Q_mean, Train_Q_max_mean]


def save_train_results(Train_Loss, Train_Q_mean, Train_Q_max_mean, curr_rl_config):
    # plot and save the training results
    """
    Save and Plot the Training results
    """
    # get the current training parameter values from curr_rl_config
    Batch_Size = curr_rl_config.Batch_Size
    Num_Train_Step = curr_rl_config.Num_Train_Steps
    Num_Episodes = curr_rl_config.Num_Episodes
    Num_D2D_feedback = curr_rl_config.Num_Feedback
    GAMMA = curr_rl_config.Gamma
    V2I_Weight = curr_rl_config.v2i_weight

    # check the saving process
    save_flag = False

    # Train_Loss size: [num_episodes x num_train_steps]
    Train_Loss_per_Episode = np.sum(Train_Loss, axis=1)/Num_Train_Step
    # compute the Target Q value
    Train_Q_mean_per_Episode = np.sum(Train_Q_mean, axis=1) / Num_Train_Step
    Train_Q_max_mean_per_Episode = np.sum(Train_Q_max_mean, axis=1) / Num_Train_Step

    # save results in their corresponding simulation parameter settings
    curr_sim_set = 'Train-Result' + '-Feedback-' + str(Num_D2D_feedback) + '-BatchSize-' + str(Batch_Size) \
                   + '-Gamma-' + str(GAMMA) + '-V2Iweight-' + str(V2I_Weight)
    folder = os.getcwd() + '\\' + curr_sim_set + '\\'
    if not os.path.exists(folder):
        os.makedirs(folder)
        print('Create the new folder in train main ', folder)

    curr_Result_Dir = folder

    # plot the results
    x = range(Num_Episodes)
    y = Train_Loss_per_Episode
    plt.figure()
    plt.plot(x, y, color='red', label='C-Decision')
    plt.xlabel("Number of Episodes")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.title("RL Training Loss")
    plt.legend()
    # save results to the file
    Curr_OS = os.name
    if Curr_OS == 'nt':
        print('Current OS is Windows！')
        Fig_Dir = curr_Result_Dir

    Fig_Name = 'Training-LOSS-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
               + '-Batch-' + str(Batch_Size) + '.png'
    Fig_Para = Fig_Dir + Fig_Name
    plt.savefig(Fig_Para, dpi=600)
    # save another format if necessary
    Fig_Name1 = 'Training-LOSS-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
                + '-Batch-' + str(Batch_Size) + '.eps'
    Fig_Para1 = Fig_Dir + Fig_Name1
    plt.savefig(Fig_Para1)

    # plot the Q max mean results of Target value
    x = range(Num_Episodes)
    y = Train_Q_mean_per_Episode
    y1 = Train_Q_max_mean_per_Episode
    plt.figure()
    plt.plot(x, y, color='red', label='Q mean')
    plt.plot(x, y1, color='blue', label='Q-max mean')
    plt.xlabel("Number of Episodes")
    plt.ylabel("Return per Episode")
    plt.grid(True)
    plt.title("Q Function in Training")
    plt.legend()

    # save the figure
    Fig_Name = 'Train-Q-Function-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
               + '-Batch-' + str(Batch_Size) + '.png'
    Fig_Para = Fig_Dir + Fig_Name
    plt.savefig(Fig_Para, dpi=600)
    Fig_Name1 = 'Train-Q-Function-plot' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
               + '-Batch-' + str(Batch_Size) + '.eps'
    Fig_Para1 = Fig_Dir + Fig_Name1
    plt.savefig(Fig_Para1, dpi=600)

    # save the results to file
    if Curr_OS == 'nt':
        # print('Current OS is Windows！')
        Data_Dir = curr_Result_Dir
    Data_Name = 'Training-Result' + '-Episode-' + str(Num_Episodes) + '-Step-' + str(Num_Train_Step) \
                + '-Batch-' + str(Batch_Size) + '.pkl'
    Data_Para = Data_Dir + Data_Name
    # open data file
    file_to_open = open(Data_Para, 'wb')
    # write train results to data file
    pickle.dump((Train_Loss_per_Episode, Train_Loss,
                 Train_Q_mean_per_Episode, Train_Q_mean,
                 Train_Q_max_mean_per_Episode, Train_Q_max_mean), file_to_open)
    file_to_open.close()

    save_flag = True

    return save_flag


if __name__ == '__main__':
    main()
