# implement the Testing main function for the C-Decision scheme with Binary Feedback

import matplotlib.pyplot as plt
from BS_brain import Agent
from Environment import *
import pickle
from Sim_Config import RL_Config
import random
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # use GPU 0 to run this code


def main():

    """
    Test the trained agent
    """

    # number of different testings
    test_num = [1, 2, 3, 4, 5,
                6, 7, 8, 9, 10]
    # number of binary feedback values
    num_feedback_set = [1,  2,  3,  6,  9,
                        15, 18, 27, 36, 45]
    # discount factor
    gamma_set = [0.05]
    # size of mini-batch
    batch_set = [512]

    # weight for sum rate of V2V in the reward
    v2v_weight = 1
    # weight for sum rate of V2I in the reward
    v2i_weight = 0.1

    num_test_settings = 4

    # parameter setting for testing
    num_test_episodes = 2000
    num_test_steps = 1000
    opt_flag = False    # whether run optimal scheme
    robust_flag = True  # whether run robust test

    if robust_flag:
        # robust parameters setting
        # feedback_interval_set = [1, 1, 1, 1, 1, 1]
        feedback_interval_set = [1, 2, 5, 10, 20, 50]
        # input_noise_level_set = [0.05, 0.1, 0.2, 2, 10, 100]
        input_noise_level_set = [0, 0, 0, 0, 0, 0]
        feedback_noise_level_set = [0, 0, 0, 0, 0, 0]
        # feedback_noise_level_set = [0.05, 0.1, 0.2, 2, 10, 100]

    # start testing
    for test_loop in range(num_test_settings):

        # set the current random seed for training
        test_seed_sequence = 1
        random.seed(test_seed_sequence)
        np.random.seed(test_seed_sequence)
        tf.set_random_seed(test_seed_sequence)  # random seed for tensor flow

        # set values for current simulation
        curr_RL_Config = RL_Config()

        train_show_tra = '----- Start the Number -- ' + str(test_num[test_loop]) + ' -- Testing -----!'
        print(train_show_tra)

        # set key parameters for the trained model
        num_feedback = num_feedback_set[test_loop]
        gamma = gamma_set[0]
        batch_size = batch_set[0]
        curr_RL_Config.set_key_value(num_feedback, gamma, batch_size, v2v_weight, v2i_weight)

        # display the parameters settings for current trained model
        curr_RL_Config.display()

        # start the Environment
        Env = start_env()

        # load the trained model
        BS_Agent = load_trained_model(Env, curr_RL_Config)

        if robust_flag:
            # set key values for testing
            feedback_interval = feedback_interval_set[test_loop]
            input_noise_level = input_noise_level_set[test_loop]
            feedback_noise_level = feedback_noise_level_set[test_loop]

            curr_RL_Config.set_robust_test_values(num_test_episodes, num_test_steps, opt_flag, v2v_weight, v2i_weight,
                                                  feedback_interval, input_noise_level, feedback_noise_level)

            # run the testing process and save the testing results
            save_flag = robust_run_test(curr_RL_Config, BS_Agent, test_seed_sequence)

            # track the testing process
            if save_flag:
                print('RL ROBUST Testing is finished!')

        else:
            # set key parameters for this testing
            curr_RL_Config.set_test_values(num_test_episodes, num_test_steps, opt_flag, v2v_weight, v2i_weight)

            # run the testing process and save the testing results
            save_flag = run_test(curr_RL_Config, BS_Agent, test_seed_sequence)

            # track the testing process
            if save_flag:
                print('RL Testing is finished!')


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


def load_trained_model(Env, curr_RL_Config):
    # load the trained C-Decision model for testing
    """
    Load the trained DNN-RL model
    """
    # parameters to construct a BS Agent object
    Num_neighbor = Env.n_Neighbor
    Num_d2d = Env.n_Veh
    Num_D2D_feedback = curr_RL_Config.Num_Feedback
    Num_CH = Env.n_RB
    # construct a BS agent
    BS_Agent = Agent(Num_d2d, Num_CH, Num_neighbor, Num_D2D_feedback, Env, curr_RL_Config)

    # load the Trained model weights
    # Training Parameters
    BATCH_SIZE = curr_RL_Config.Batch_Size
    num_episodes = curr_RL_Config.Num_Episodes
    num_train_steps = curr_RL_Config.Num_Train_Steps
    GAMMA = curr_RL_Config.Gamma
    V2I_Weight = curr_RL_Config.v2i_weight

    # load the trained results according to their corresponding simulation parameter settings
    curr_sim_set = 'Train-Result' + '-Feedback-' + str(Num_D2D_feedback) + '-BatchSize-' + str(BATCH_SIZE) \
                   + '-Gamma-' + str(GAMMA) + '-V2Iweight-' + str(V2I_Weight)
    folder = os.getcwd() + '\\' + curr_sim_set + '\\'
    if not os.path.exists(folder):
        os.makedirs(folder)
        print('Create the new folder in Testing main ', folder)

    model_dir = folder

    model_name = 'Q-Network_model_weights' + '-Episode-' + str(num_episodes) \
                 + '-Step-' + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
    model_para = model_dir + model_name

    # save the Target Network's weights in case we need it
    target_model_name = 'Target-Network_model_weights' + '-Episode-' + str(num_episodes) + '-Step-' \
                        + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
    target_model_para = model_dir + target_model_name

    # load Q-Function Network weights
    BS_Agent.brain.model.load_weights(model_para)
    # load Target Network weights
    BS_Agent.brain.target_model.load_weights(target_model_para)

    # for debugging
    print('Load the trained model successfully under this setting!')

    # return the agent with trained model
    return BS_Agent


def run_test(curr_RL_Config, BS_Agent, test_seed_sequence):
    # run the test according to current settings via the trained model
    """
    Run the Test
    """
    save_flag = False      # check the saving process
    # get current testing setting
    Num_Run_Episodes = curr_RL_Config.Num_Run_Episodes
    Num_Test_Step = curr_RL_Config.Num_Test_Steps
    Opt_Flag = curr_RL_Config.Opt_Flag
    Num_D2D_feedback = curr_RL_Config.Num_Feedback
    Batch_Size = curr_RL_Config.Batch_Size
    GAMMA = curr_RL_Config.Gamma
    V2I_Weight = curr_RL_Config.v2i_weight
    V2V_Weight = curr_RL_Config.v2v_weight

    # for tracking of the test
    print("-----Current Testing Parameters Settings are: ")
    print('     Number of feedback: ', Num_D2D_feedback)
    print('     Discount Factor Gamma: ', GAMMA)
    print('     Optimal Scheme Flag: ', Opt_Flag)
    print('     Batch Size: ', Batch_Size)
    print('     Testing Episodes: ', Num_Run_Episodes)
    print('     Testing Steps per Episode: ', Num_Test_Step)
    print('     Testing Seed: ', test_seed_sequence)
    print('     V2V Rate weight: ', V2V_Weight)
    print('     V2I Rate weight: ', V2I_Weight)

    if Opt_Flag:

        print('To Run Test DNN-RL with Optimal Scheme!')

        # Run with Implementing Optimal Scheme
        [Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
        Per_V2B_Interference,
        RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
        RA_Per_V2B_Interference,
        Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
        Opt_Per_V2B_Interference]  \
            = BS_Agent.run(Num_Run_Episodes, Num_Test_Step, Opt_Flag)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Opt-Run-Result' + '-Feedback-' + str(Num_D2D_feedback) + '-BatchSize-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) + '-Seed-' + str(test_seed_sequence) \
                       + '-V2Iweight-' + str(V2I_Weight)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main with Opt-scheme', folder)

        print('------> Testing Results are: ')
        # to better evaluate the RL performance
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        print('      The indexes of episodes, where RL is worse than RA  are ', LessThanRA_Index)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        print('      The return differences of episodes, where RL is worse than RA  are ', LessThanRA)
        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)

        ave_Opt_Expect_Return = np.sum(Opt_Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of Opt Scheme is ', ave_Opt_Expect_Return)
        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of RA scheme is ', ave_RA_Return)

        print('*******> Testing Results for V2V link are: ')
        ave_Opt_Per_V2V_Rate = np.sum(Opt_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of Optimal scheme is ', ave_Opt_Per_V2V_Rate)
        ave_Per_V2V_Rate = np.sum(Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode  of RL scheme is ', ave_Per_V2V_Rate)
        ave_RA_Per_V2V_Rate = np.sum(RA_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of RA scheme is ', ave_RA_Per_V2V_Rate)

        print('*******> Testing Results for V2I link are: ')
        ave_Opt_Per_V2I_Rate = np.sum(Opt_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode of Optimal scheme is ', ave_Opt_Per_V2I_Rate)
        ave_Per_V2I_Rate = np.sum(Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode  of RL scheme is ', ave_Per_V2I_Rate)
        ave_RA_Per_V2I_Rate = np.sum(RA_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode of RA scheme is ', ave_RA_Per_V2I_Rate)

        Interfernece_Normalizer = Num_Run_Episodes * Num_Test_Step
        ave_Opt_Per_V2B_Interference = np.sum(Opt_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of Optimal scheme is ', ave_Opt_Per_V2B_Interference)
        ave_Per_V2B_Interference = np.sum(Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of RL scheme is ', ave_Per_V2B_Interference)
        RA_ave_Per_V2B_Interference = np.sum(RA_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of RA scheme is ', RA_ave_Per_V2B_Interference)

        # plot the results
        # here we just show some examples
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
        y2 = Opt_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.plot(x, y2, color='blue', label='Optimal Scheme')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Curr_OS = os.name
        if Curr_OS == 'nt':
            # print('Current OS is Windows！')
            Fig_Dir = folder
        Fig_Name = 'Opt-D2DRLplot' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-D2DRLplot' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = range(Num_Run_Episodes)
        y = Expect_Return / Opt_Expect_Return
        y1 = RA_Expect_Return / Opt_Expect_Return
        y2 = Opt_Expect_Return / Opt_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.plot(x, y2, color='blue', label='Optimal Scheme')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Opt-Norm' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-Norm' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results to compare Random Action(RA) and C-Decision
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Opt-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Opt-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # save the results to file
        if Curr_OS == 'nt':
            print('Save Test Results, Current OS is Windows！')
            Data_Dir = folder
        Data_Name = 'Opt-Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write variables to data file
        pickle.dump((Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
                     Per_V2B_Interference,
                    RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
                    RA_Per_V2B_Interference,
                    Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
                    Opt_Per_V2B_Interference), file_to_open)

        # close data file
        file_to_open.close()

        # save some key testing results
        Data_Name1 = 'Ave-Opt-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(
            Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'wb')
        pickle.dump((ave_Opt_Expect_Return, ave_Opt_Per_V2I_Rate,
                     ave_Opt_Per_V2B_Interference, ave_Opt_Per_V2V_Rate,
                     ave_Expected_Return, ave_Per_V2I_Rate,
                     ave_Per_V2B_Interference, ave_Per_V2V_Rate,
                     ave_RA_Return, ave_RA_Per_V2I_Rate,
                     RA_ave_Per_V2B_Interference, ave_RA_Per_V2V_Rate,
                     BetterThanRA_Num, LessThanRA, LessThanRA_Index), file_to_open)
        file_to_open.close()

        print('The Optimal Testing is finished!')
        save_flag = True

    else:
        print('To Run Test C-Decision without Implementing Optimal Scheme!')

        [Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
         Per_V2B_Interference,
         RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
         RA_Per_V2B_Interference] \
            = BS_Agent.run(Num_Run_Episodes, Num_Test_Step, Opt_Flag)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Run-Result' + '-Feedback-' + str(Num_D2D_feedback) + '-BatchSize-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) + '-Seed-' + str(test_seed_sequence) \
                       + '-V2Iweight-' + str(V2I_Weight)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main without Opt-Scheme ', folder)

        print('------> Testing Results for V2V link are: ')

        # to better evaluate the RL performance
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        print('      The indexes of episodes, where RL is worse than RA  are ', LessThanRA_Index)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        print('      The return differences of episodes, where RL is worse than RA  are ', LessThanRA)
        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)

        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The average return of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The average return of RA scheme is ', ave_RA_Return)

        print('*******> Testing Results for V2V link are: ')
        ave_Per_V2V_Rate = np.sum(Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode  of RL scheme is ', ave_Per_V2V_Rate)
        ave_RA_Per_V2V_Rate = np.sum(RA_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of RA scheme is ', ave_RA_Per_V2V_Rate)

        print('*******> Testing Results for V2I link are: ')
        ave_Per_V2I_Rate = np.sum(Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate of RL scheme is ', ave_Per_V2I_Rate)
        ave_RA_Per_V2I_Rate = np.sum(RA_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate of RA scheme is ', ave_RA_Per_V2I_Rate)

        print('$$$$$$$> Testing Results for V2B Interference control are: ')
        Interfernece_Normalizer = Num_Run_Episodes * Num_Test_Step
        ave_Per_V2B_Interference = np.sum(Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference of RL scheme is ', ave_Per_V2B_Interference)
        RA_ave_Per_V2B_Interference = np.sum(RA_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference of RA scheme is ', RA_ave_Per_V2B_Interference)

        # plot the results
        Curr_OS = os.name
        if Curr_OS == 'nt':
            Fig_Dir = folder
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()

        Fig_Name = 'Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = range(Num_Run_Episodes)
        y = Expect_Return / Expect_Return
        y1 = RA_Expect_Return / Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # save the results to file
        if Curr_OS == 'nt':
            print('Save testing results！ Current OS is Windows！')
            Data_Dir = folder

        Data_Name = 'Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write testing results to data file
        pickle.dump((Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
                    Per_V2B_Interference,
                    RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
                    RA_Per_V2B_Interference), file_to_open)
        # close data file
        file_to_open.close()

        # save some key testing results
        Data_Name1 = 'Ave-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'wb')
        # write key testing results to data file
        pickle.dump((ave_Expected_Return, ave_Per_V2I_Rate,
                     ave_Per_V2B_Interference, ave_Per_V2V_Rate,
                     ave_RA_Return, ave_RA_Per_V2I_Rate,
                     RA_ave_Per_V2B_Interference, ave_RA_Per_V2V_Rate,
                     BetterThanRA_Num, LessThanRA, LessThanRA_Index), file_to_open)
        file_to_open.close()

        save_flag = True

    return save_flag


def robust_run_test(curr_RL_Config, BS_Agent, test_seed_sequence):
    # run robust test according to current settings via the trained model
    """
    Run the Robust Test
    """
    save_flag = False      # check the saving process
    Num_Run_Episodes = curr_RL_Config.Num_Run_Episodes
    Num_Test_Step = curr_RL_Config.Num_Test_Steps
    Opt_Flag = curr_RL_Config.Opt_Flag
    Num_D2D_feedback = curr_RL_Config.Num_Feedback
    Batch_Size = curr_RL_Config.Batch_Size
    GAMMA = curr_RL_Config.Gamma
    V2I_Weight = curr_RL_Config.v2i_weight
    V2V_Weight = curr_RL_Config.v2v_weight
    # robust related parameters
    Feedback_Interval = curr_RL_Config.feedback_interval
    Input_Noise_Level = curr_RL_Config.input_noise_level
    Feedback_Noise_Level = curr_RL_Config.feedback_noise_level

    # for tracking of the test
    print("-----Current ROBUST Testing Parameters Settings are: ")
    print('     Number of feedback: ', Num_D2D_feedback)
    print('     Discount Factor Gamma: ', GAMMA)
    print('     Optimal Scheme Flag: ', Opt_Flag)
    print('     Batch Size: ', Batch_Size)
    print('     Testing Episodes: ', Num_Run_Episodes)
    print('     Testing Steps per Episode: ', Num_Test_Step)
    print('     Testing Seed: ', test_seed_sequence)
    print('     V2V Rate weight: ', V2V_Weight)
    print('     V2I Rate weight: ', V2I_Weight)
    print('     Feedback Interval: ', Feedback_Interval)
    print('     Inputs Noise Level: ', Input_Noise_Level)
    print('     Feedback Noise Level: ', Feedback_Noise_Level)

    if Opt_Flag:

        print('To Run ROBUST test C-Decision with Optimal Scheme!')

        # Run with Implementing Optimal Scheme
        [Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
        Per_V2B_Interference,
        RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
        RA_Per_V2B_Interference,
        Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
        Opt_Per_V2B_Interference]  \
            = BS_Agent.robust_run(Num_Run_Episodes, Num_Test_Step, Opt_Flag, Feedback_Interval,
                                  Input_Noise_Level, Feedback_Noise_Level)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Robust-Opt-Run-Result' + '-FB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) + '-V2Iweight-' + str(V2I_Weight) \
                       + '-FBInter-' + str(Feedback_Interval) + '-FBNoise-' + str(Feedback_Noise_Level) \
                       + '-InputNoise-' + str(Input_Noise_Level)

        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main with Opt-scheme', folder)

        print('------> Testing Results are: ')
        # to better evaluate the RL performance
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        print('      The indexes of episodes, where RL is worse than RA  are ', LessThanRA_Index)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        print('      The return differences of episodes, where RL is worse than RA  are ', LessThanRA)
        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)

        ave_Opt_Expect_Return = np.sum(Opt_Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of Opt Scheme is ', ave_Opt_Expect_Return)
        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The Average Return per episode of RA scheme is ', ave_RA_Return)

        print('*******> Testing Results for V2V link are: ')
        ave_Opt_Per_V2V_Rate = np.sum(Opt_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of Optimal scheme is ', ave_Opt_Per_V2V_Rate)
        ave_Per_V2V_Rate = np.sum(Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode  of RL scheme is ', ave_Per_V2V_Rate)
        ave_RA_Per_V2V_Rate = np.sum(RA_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of RA scheme is ', ave_RA_Per_V2V_Rate)

        print('*******> Testing Results for V2I link are: ')
        ave_Opt_Per_V2I_Rate = np.sum(Opt_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode of Optimal scheme is ', ave_Opt_Per_V2I_Rate)
        ave_Per_V2I_Rate = np.sum(Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode  of RL scheme is ', ave_Per_V2I_Rate)
        ave_RA_Per_V2I_Rate = np.sum(RA_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate per episode of RA scheme is ', ave_RA_Per_V2I_Rate)

        Interfernece_Normalizer = Num_Run_Episodes * Num_Test_Step
        ave_Opt_Per_V2B_Interference = np.sum(Opt_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of Optimal scheme is ', ave_Opt_Per_V2B_Interference)
        ave_Per_V2B_Interference = np.sum(Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of RL scheme is ', ave_Per_V2B_Interference)
        RA_ave_Per_V2B_Interference = np.sum(RA_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference per step of RA scheme is ', RA_ave_Per_V2B_Interference)

        # plot the results
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
        y2 = Opt_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.plot(x, y2, color='cyan', label='Optimal Scheme')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Curr_OS = os.name
        if Curr_OS == 'nt':
            # print('Current OS is Windows！')
            Fig_Dir = folder
        Fig_Name = 'Opt-D2DRLplot' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-D2DRLplot' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = range(Num_Run_Episodes)
        y = Expect_Return / Opt_Expect_Return
        y1 = RA_Expect_Return / Opt_Expect_Return
        y2 = Opt_Expect_Return / Opt_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.plot(x, y2, color='blue', label='Optimal Scheme')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Opt-Norm' + '-Episode-' + str(Num_Run_Episodes) \
                   + '-Step-' + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)
        Fig_Name1 = 'Opt-Norm' + '-Episode-' + str(Num_Run_Episodes) \
                    + '-Step-' + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # save the results to file
        if Curr_OS == 'nt':
            print('Save Test Results, Current OS is Windows！')
            Data_Dir = folder
        Data_Name = 'Opt-Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write testing results to data file
        pickle.dump((Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
                     Per_V2B_Interference,
                    RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
                    RA_Per_V2B_Interference,
                    Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate,
                    Opt_Per_V2B_Interference), file_to_open)
        # close data file
        file_to_open.close()

        # save key test variables
        Data_Name1 = 'Ave-Opt-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(
            Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'wb')
        # write the key results to data file
        pickle.dump((ave_Opt_Expect_Return, ave_Opt_Per_V2I_Rate,
                     ave_Opt_Per_V2B_Interference, ave_Opt_Per_V2V_Rate,
                     ave_Expected_Return, ave_Per_V2I_Rate,
                     ave_Per_V2B_Interference, ave_Per_V2V_Rate,
                     ave_RA_Return, ave_RA_Per_V2I_Rate,
                     RA_ave_Per_V2B_Interference, ave_RA_Per_V2V_Rate,
                     BetterThanRA_Num, LessThanRA, LessThanRA_Index), file_to_open)
        file_to_open.close()

        print('The Optimal Testing is finished!')
        save_flag = True

    else:
        print('To Run ROBUST test C-Decision withOUT Optimal Scheme!')

        [Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
         Per_V2B_Interference,
         RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
         RA_Per_V2B_Interference] \
            = BS_Agent.robust_run(Num_Run_Episodes, Num_Test_Step, Opt_Flag, Feedback_Interval,
                                  Input_Noise_Level, Feedback_Noise_Level)

        #  save the tested results to files with their corresponding simulation parameter settings
        curr_sim_set = 'Robust-Run-Result' + '-FB-' + str(Num_D2D_feedback) + '-Batch-' + str(Batch_Size) \
                       + '-Gamma-' + str(GAMMA) + '-V2Iweight-' + str(V2I_Weight) \
                       + '-FBInter-' + str(Feedback_Interval) + '-FBNoise-' + str(Feedback_Noise_Level) \
                       + '-InputNoise-' + str(Input_Noise_Level)

        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder in Test main without Opt-Scheme ', folder)

        print('------> Testing Results for V2V link are: ')
        # to better evaluate the RL performance
        LessThanRA_Index = np.where(Expect_Return - RA_Expect_Return < 0)
        print('      The indexes of episodes, where RL is worse than RA  are ', LessThanRA_Index)
        LessThanRA = (Expect_Return - RA_Expect_Return)[np.where(Expect_Return - RA_Expect_Return < 0)]
        print('      The return differences of episodes, where RL is worse than RA  are ', LessThanRA)
        BetterThanRA_Num = Num_Run_Episodes - len(LessThanRA_Index[0])
        print('      The number of episodes, where RL is better than RA  are ', BetterThanRA_Num)

        ave_Expected_Return = np.sum(Expect_Return) / Num_Run_Episodes
        print('      The average return of RL is ', ave_Expected_Return)
        ave_RA_Return = np.sum(RA_Expect_Return) / Num_Run_Episodes
        print('      The average return of RA scheme is ', ave_RA_Return)

        print('*******> Testing Results for V2V link are: ')
        ave_Per_V2V_Rate = np.sum(Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode  of RL scheme is ', ave_Per_V2V_Rate)
        ave_RA_Per_V2V_Rate = np.sum(RA_Per_V2V_Rate) / Num_Run_Episodes
        print('      The average V2V rate per episode of RA scheme is ', ave_RA_Per_V2V_Rate)

        print('*******> Testing Results for V2I link are: ')
        ave_Per_V2I_Rate = np.sum(Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate of RL scheme is ', ave_Per_V2I_Rate)
        ave_RA_Per_V2I_Rate = np.sum(RA_Per_V2I_Rate) / Num_Run_Episodes
        print('      The average V2I rate of RA scheme is ', ave_RA_Per_V2I_Rate)

        print('$$$$$$$> Testing Results for V2B Interference control are: ')
        Interfernece_Normalizer = Num_Run_Episodes * Num_Test_Step
        ave_Per_V2B_Interference = np.sum(Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference of RL scheme is ', ave_Per_V2B_Interference)
        RA_ave_Per_V2B_Interference = np.sum(RA_Per_V2B_Interference) / Interfernece_Normalizer
        print('      The average V2B interference of RA scheme is ', RA_ave_Per_V2B_Interference)

        # plot the results
        Curr_OS = os.name
        if Curr_OS == 'nt':
            # print('Current OS is Windows！')
            Fig_Dir = folder
        x = range(Num_Run_Episodes)
        y = Expect_Return
        y1 = RA_Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # plot the results normalize these return to see the gain percentage
        x = range(Num_Run_Episodes)
        y = Expect_Return / Expect_Return
        y1 = RA_Expect_Return / Expect_Return
        plt.figure()
        plt.plot(x, y, color='red', label='C-Decision')
        plt.plot(x, y1, color='green', label='Random Action')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Normalized Return")
        plt.grid(True)
        plt.title("RL Testing Results")
        plt.legend()
        Fig_Name = 'Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.png'
        Fig_Para = Fig_Dir + Fig_Name
        plt.savefig(Fig_Para, dpi=600)

        Fig_Name1 = 'Norm-Comp-RL-RA' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' \
                   + str(Num_Test_Step) + '.eps'
        Fig_Para = Fig_Dir + Fig_Name1
        plt.savefig(Fig_Para)

        # save the results to file
        if Curr_OS == 'nt':
            print('Save testing results！ Current OS is Windows！')
            Data_Dir = folder

        Data_Name = 'Testing-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para = Data_Dir + Data_Name
        # open data file
        file_to_open = open(Data_Para, 'wb')
        # write test results to data file
        pickle.dump((Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate,
                    Per_V2B_Interference,
                    RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate,
                    RA_Per_V2B_Interference), file_to_open)
        file_to_open.close()

        # save key test variables
        Data_Name1 = 'Ave-Test-Result' + '-Episode-' + str(Num_Run_Episodes) + '-Step-' + str(Num_Test_Step) + '.pkl'
        Data_Para1 = Data_Dir + Data_Name1
        # open data file
        file_to_open = open(Data_Para1, 'wb')
        # write key testing results to data file
        pickle.dump((ave_Expected_Return, ave_Per_V2I_Rate,
                     ave_Per_V2B_Interference, ave_Per_V2V_Rate,
                     ave_RA_Return, ave_RA_Per_V2I_Rate,
                     RA_ave_Per_V2B_Interference, ave_RA_Per_V2V_Rate,
                     BetterThanRA_Num, LessThanRA, LessThanRA_Index), file_to_open)
        file_to_open.close()

        save_flag = True

    return save_flag


if __name__ == '__main__':
    main()
