# implement the DNN in Each D2D and the DQN at the BS for the C-Decision scheme with the Real Feedback
# here we use D2D and V2V interchangeably

from scipy.special import comb
import numpy as np
import pickle
import keras
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import datetime
import os
import random


# define huber loss
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


# -------------------- BS --------------------------


class BS:
    """
    Define the BS DNN class
    """
    def __init__(self, num_d2d, input_d2d_info, num_d2d_feedback, num_d2d_neighbor, num_ch):
        self.num_D2D = num_d2d
        self.num_Neighbor = num_d2d_neighbor
        self.num_CH = num_ch
        self.num_Feedback = num_d2d_feedback
        self.num_Output = int(comb(self.num_CH, self.num_Neighbor))**self.num_D2D
        self.num_Input = self.num_D2D*self.num_Feedback
        self.input_D2D_Info = input_d2d_info
        self.num_One_D2D_Input = ((input_d2d_info - 1)*self.num_CH + 1)*self.num_Neighbor
        self.num_D2D_Input = num_d2d*self.num_One_D2D_Input
        self.model = self._create_model()
        self.target_model = self._create_model()  # target network

    def _create_model(self):
        # construct the DNN at each D2D and BQN at the BS
        # to implement end-to-end training, link DNNs and DQN together

        Num_D2D_Input = self.num_One_D2D_Input
        Num_D2D_Output = self.num_Feedback

        Num_Inner_Layer_1 = 32
        Num_Inner_Layer_2 = 64
        Num_Inner_Layer_3 = 32

        # implement DNN for 1st D2D
        D1_Input = Input(shape=(Num_D2D_Input,), name='D1_Input')
        D1 = Dense(Num_Inner_Layer_1, activation='relu')(D1_Input)
        D1 = Dense(Num_Inner_Layer_2, activation='relu')(D1)
        D1 = Dense(Num_Inner_Layer_3, activation='relu')(D1)
        D1_Output = Dense(Num_D2D_Output, activation='linear', name='D1_Output')(D1)

        # implement DNN for 2nd D2D
        D2_Input = Input(shape=(Num_D2D_Input,), name='D2_Input')
        D2 = Dense(Num_Inner_Layer_1, activation='relu')(D2_Input)
        D2 = Dense(Num_Inner_Layer_2, activation='relu')(D2)
        D2 = Dense(Num_Inner_Layer_3, activation='relu')(D2)
        D2_Output = Dense(Num_D2D_Output, activation='linear', name='D2_Output')(D2)

        # implement DNN for 3rd D2D
        D3_Input = Input(shape=(Num_D2D_Input,), name='D3_Input')
        D3 = Dense(Num_Inner_Layer_1, activation='relu')(D3_Input)
        D3 = Dense(Num_Inner_Layer_2, activation='relu')(D3)
        D3 = Dense(Num_Inner_Layer_3, activation='relu')(D3)
        D3_Output = Dense(Num_D2D_Output, activation='linear', name='D3_Output')(D3)

        # implement DNN for 4th D2D
        D4_Input = Input(shape=(Num_D2D_Input,), name='D4_Input')
        D4 = Dense(Num_Inner_Layer_1, activation='relu')(D4_Input)
        D4 = Dense(Num_Inner_Layer_2, activation='relu')(D4)
        D4 = Dense(Num_Inner_Layer_3, activation='relu')(D4)
        D4_Output = Dense(Num_D2D_Output, activation='linear', name='D4_Output')(D4)

        BS_Input = keras.layers.concatenate([D1_Output, D2_Output, D3_Output, D4_Output], name='BS_Input')

        BS_Dense1 = Dense(2400, activation='relu', name='BS_Dense1')(BS_Input)
        BS_Dense2 = Dense(1600, activation='relu', name='BS_Dense2')(BS_Dense1)
        BS_DNN = Dense(1200, activation='relu', name='BS')(BS_Dense2)

        Num_BS_Output = self.num_Output
        BS_output = Dense(Num_BS_Output, activation='linear', name='BS_output')(BS_DNN)

        # Define the model
        model = Model(inputs=[D1_Input, D2_Input, D3_Input, D4_Input],
                      outputs=[BS_output])

        # the default value of learning rate lr=0.001, change it if necessary
        rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        # Compile the model
        model.compile(optimizer=rms, loss=huber_loss)

        return model

    def train_dnn(self, data_train, labels, batch_size):
        # training method for the DNN and BS DQN
        epochs = 1   # use its default value
        Train_Result = self.model.fit(data_train, labels, batch_size=batch_size, epochs=epochs, verbose=0)

        return Train_Result

    def predict(self, data_test, target=False):
        # predict the value
        # target: True -> choose the target network; otherwise, choose the Q-function network
        if target:
            return self.target_model.predict(data_test)
        else:
            return self.model.predict(data_test)

    def predict_one_step(self, data_test, target=False):
        return self.predict(data_test, target=target)

    def update_target_model(self):
        # use current model weight to update target network
        self.target_model.set_weights(self.model.get_weights())


# -------------------- MEMORY --------------------------


class Memory:  # stored as ( s, a, r, s_ )
    # define the memory class to replay
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    # add sample
    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    # choose n samples from the memory
    def sample(self, n):
        # choose the min{batchsize, current_Num_samples}
        # here input n = BATCH_SIZE
        if len(self.samples) >= n:
            Samples_Indices = np.random.choice(len(self.samples), n, replace=False)
            Batch_Samples = np.array(self.samples)[Samples_Indices]
            return Batch_Samples
        else:
            # while number of current samples is less than n(that is the BATCH_SIZE),
            # repeated sample the current samples until we get BATCH_SIZE samples
            Batch_Samples = []
            while len(Batch_Samples) < n:
                index = np.random.randint(0, len(self.samples))
                Batch_Samples.append(self.samples[index])

            return Batch_Samples


# ------------------------AGENT--------------------

MEMORY_CAPACITY = 1000000  # size of replay memory
UPDATE_TARGET_FREQUENCY = 500  # update interval of target network
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay


class Agent:
    """
    Define the BS Agent class
    """
    def __init__(self, num_d2d, num_ch, num_neighbor, num_d2d_feedback, environment, curr_rl_config):
        self.num_Action = num_ch
        self.epsilon = MAX_EPSILON
        self.num_step = 0
        self.num_CH = num_ch
        self.num_D2D = num_d2d
        self.num_Neighbor = num_neighbor
        self.num_Feedback = num_d2d_feedback
        self.memory = Memory(MEMORY_CAPACITY)
        # D2D inputs: V2V channel gain on all channels, Interference on all channels (other V2V and V2I),
        #             V2I channel gain on all channels, transmit power
        self.input_D2D_Info = 4
        self.env = environment
        self.training = True
        self.brain = BS(self.num_D2D, self.input_D2D_Info, self.num_Feedback, self.num_Neighbor, self.num_CH)
        self.num_States = self.brain.num_D2D_Input
        self.num_RL_Actions = self.brain.num_Output
        # each D2D have to choose N_neighbor channels
        self.num_Actions = self.num_D2D*self.num_Neighbor
        self.action_all_with_power = np.zeros([self.num_D2D, self.num_Neighbor, self.num_CH], dtype='int32')
        self.action_all_with_power_training = np.zeros([self.num_D2D, self.num_Neighbor, self.num_CH], dtype='int32')
        self.batch_size = curr_rl_config.Batch_Size
        self.gamma = curr_rl_config.Gamma
        # the v2v rate weight
        self.v2v_weight = curr_rl_config.v2v_weight
        # the v2i rate weight
        self.v2i_weight = curr_rl_config.v2i_weight

    def select_action_while_training(self, state):
        # implement Epsilon-Greedy Strategy
        # according to current state, choose the proper action
        num_D2D = self.num_D2D
        num_neighbor = self.num_Neighbor
        # generate an action matrix for all D2D with their respective neighbors
        # initialize the Action_Matrix
        Action_Matrix = 100*np.ones((num_D2D, num_neighbor))

        CH_Set = range(0, self.num_CH)

        # decrease Epsilon linearly
        Epsilon_decrease_percentage = 0.8
        Epsilon_decrease_Episode = self.num_Episodes * Epsilon_decrease_percentage
        Epsilon_decrease_Steps = Epsilon_decrease_Episode * self.num_Train_Step * self.num_transition
        Epsilon_decrease_per_Step = (MAX_EPSILON - MIN_EPSILON) / Epsilon_decrease_Steps
        if self.num_step < Epsilon_decrease_Steps:
            self.epsilon = MAX_EPSILON - Epsilon_decrease_per_Step*self.num_step
        else:
            self.epsilon = MIN_EPSILON

        # track the training process
        if self.num_step % 50000 == 0:
            print('Current Epsilon while Training is ', self.epsilon, ' Current Training Step is ', self.num_step)

        if np.random.random() < self.epsilon:
            # generate action for each D2D randomly
            for D2D_loop in range(num_D2D):
                Action_Matrix[D2D_loop, :] = np.random.choice(CH_Set, num_neighbor)
        else:
            # choose the action index which maximize the Q Function
            Q_Pred = self.brain.predict_one_step(state, target=False)

            # Find the RL-actions to maximize Q value
            Action_Max = np.where(Q_Pred == np.max(Q_Pred))
            RL_Action = Action_Max[1][0]

            # Get the D2D Actions
            # change the RL_Actions [0,255] to D2D actions [a, a, a, a] where a in {0,1,2,3}
            D2D_Action = np.zeros(self.num_D2D, int)
            # Change  a_RL (Decimal)  to a (Quaternary)
            n = RL_Action
            a0 = n // (4 ** 3)
            a1 = (n % (4 ** 3)) // (4 ** 2)
            a2 = (n % (4 ** 2)) // (4 ** 1)
            a3 = n % (4 ** 1)
            D2D_Action[0] = a0
            D2D_Action[1] = a1
            D2D_Action[2] = a2
            D2D_Action[3] = a3
            Actions = np.reshape(D2D_Action, [self.num_D2D, num_neighbor])
            Action_Matrix = Actions

        return Action_Matrix.astype(np.int)

    def select_action_random(self, state):
        # choose the action Randomly
        num_D2D = self.num_D2D
        num_neighbor = self.num_Neighbor

        Action_Matrix = 100*np.ones((num_D2D, num_neighbor))
        CH_Set = range(0, self.num_CH)

        # generate action randomly
        for D2D_loop in range(num_D2D):
            Action_Matrix[D2D_loop, :] = np.random.choice(CH_Set, num_neighbor)

        return Action_Matrix.astype(np.int)

    def act(self, actions):
        # take actions and get reward, then update the state
        # update current time step
        self.num_step += 1
        [V2V_Rate, V2I_Rate, Interference] = self.env.compute_reward_with_channel_selection(actions)
        # update the environment
        self.env.renew_positions()
        self.env.renew_channels_fastfading()
        self.env.Compute_Interference(actions)

        return V2V_Rate, V2I_Rate, Interference

    def dump_act(self, actions):
        # take actions and get reward, do not update the state
        # used for the comparing schemes, such as, random action scheme
        [V2V_Rate, V2I_Rate, Interference] = self.env.compute_reward_with_channel_selection(actions)
        return V2V_Rate, V2I_Rate, Interference

    def train_observe(self, sample):
        # observe for training only
        # sample in (s, a, r, s_) format
        self.memory.add(sample)

    def get_state(self, idx):
        # get state from the environment
        # Input: indx[0] = target vehicle index, indx[1] = neighbor index

        # to normalize channel gain and interference to a reasonable range
        Constant_A = 80
        Constant_B = 60

        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0],
                       self.env.vehicles[idx[0]].destinations[idx[1]], :] - Constant_A)/Constant_B

        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - Constant_A)/Constant_B
        V2V_interference = (-self.env.V2V_Interference_all[idx[0], idx[1], :] - Constant_B)/Constant_B

        return V2V_channel, V2V_interference, V2I_channel

    def generate_d2d_transition(self, num_transitions):
        # take action via the Epsilon-Greedy strategy, observe the transition (S, A, R, S_) while training,
        # then add this transition to the Buffer   
        self.train_step = 0
        self.random_action = False  # adopt the Epsilon-Greedy strategy

        Num_States = self.num_States
        Num_Actions = self.num_Actions
        Num_One_D2D_Input = self.brain.num_One_D2D_Input
        CH_gain_Index = self.num_Neighbor * self.num_CH
        CH_Interf_Index = self.num_Neighbor * self.num_CH

        # record the reward per transitions
        Reward_Per_Transition = np.zeros(num_transitions)
        # weight for sum rate of V2V in the reward
        v2v_weight = self.v2v_weight
        # weight for sum rate of V2I in the reward
        v2i_weight = self.v2i_weight
        # normalize rate for V2V rate if necessary
        V2V_Rate_max = 1

        # generate num_transitions of transitions
        for self.train_step in range(num_transitions):
            # initialize temp variables
            if self.train_step == 0:
                Train_D2D_CH_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
                Train_D2D_Interf_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
                Train_D2D_V2I_CH_State = np.zeros((self.num_D2D, self.num_CH))
                Fixed_Power = self.env.V2V_power_dB_List[self.env.fixed_v2v_power_index]
                Train_D2D_Power_State = Fixed_Power * np.ones((self.num_D2D, self.num_Neighbor))

            # Get the related V2V and V2I channel and interference state for training
            for D2D_loop in range(self.num_D2D):
                for Neighbor_loop in range(self.num_Neighbor):
                    # Input: indx[0] = target vehicle index, indx[1] = neighbor index
                    index = [D2D_loop, Neighbor_loop]
                    [V2V_channel, V2V_interference, V2I_channel] = self.get_state(index)
                    Train_D2D_CH_State[D2D_loop, Neighbor_loop, :] = V2V_channel
                    Train_D2D_Interf_State[D2D_loop, Neighbor_loop, :] = V2V_interference
                Train_D2D_V2I_CH_State[D2D_loop, :] = V2I_channel

            # reshape the training data in (S, A, R, S_)
            # reshape the States for all D2D
            D2D_State = np.zeros((self.num_D2D, Num_One_D2D_Input))

            for D2D_loop in range(self.num_D2D):
                Current_CH_gain = np.reshape(Train_D2D_CH_State[D2D_loop, :, :], [1, CH_gain_Index])
                D2D_State[D2D_loop, 0:CH_gain_Index] = Current_CH_gain
                Current_Interf_gain = np.reshape(Train_D2D_Interf_State[D2D_loop, :, :], [1, CH_Interf_Index])
                D2D_State[D2D_loop, CH_gain_Index:2*CH_gain_Index] = Current_Interf_gain
                Current_V2I_gain = Train_D2D_V2I_CH_State[D2D_loop, :]
                D2D_State[D2D_loop, 2 * CH_gain_Index:3 * CH_gain_Index] = Current_V2I_gain
                D2D_State[D2D_loop, 3*CH_gain_Index: Num_One_D2D_Input] = Train_D2D_Power_State[D2D_loop, :]

            States = np.reshape(D2D_State, [1, Num_States])

            # two action selection strategies: Random Selection and Epsilon-Greedy Strategy
            if self.random_action:
                # Choose the actions randomly
                Train_D2D_Action_Matrix = self.select_action_random(States)
            else:
                # Epsilon-Greedy Strategy
                D1_State = np.reshape(D2D_State[0, :], [1, Num_One_D2D_Input])
                D2_State = np.reshape(D2D_State[1, :], [1, Num_One_D2D_Input])
                D3_State = np.reshape(D2D_State[2, :], [1, Num_One_D2D_Input])
                D4_State = np.reshape(D2D_State[3, :], [1, Num_One_D2D_Input])

                States_train = {'D1_Input': D1_State, 'D2_Input': D2_State,
                                'D3_Input': D3_State, 'D4_Input': D4_State}

                # choose action via Epsilon-Greedy strategy
                Train_D2D_Action_Matrix = self.select_action_while_training(States_train)

            # reshape the actions
            Actions = np.reshape(Train_D2D_Action_Matrix, [1, Num_Actions])

            # Take action and Get Reward
            [V2V_Rate, V2I_Rate, Interference] = self.act(Train_D2D_Action_Matrix)
            # compute sum rate of V2V link
            Train_D2D_Reward = np.sum(V2V_Rate, axis=1)  # sum by row
            Train_V2V_Rate = np.sum(Train_D2D_Reward)  # total reward

            # compute sum rate of V2V link
            Sum_V2I_Rate = np.sum(V2I_Rate)
            # normalize the V2V reward if necessary
            Norm_BS_Reward = Train_V2V_Rate/V2V_Rate_max
            # compute the current reward
            Reward = v2v_weight * Norm_BS_Reward + v2i_weight * Sum_V2I_Rate

            # record the current reward
            Reward_Per_Transition[self.train_step] = Reward

            # Get NEXT state: all related V2V and V2I channel and interference state for training
            Next_Train_D2D_CH_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
            Next_Train_D2D_Interf_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
            Next_Train_D2D_V2I_CH_State = np.zeros((self.num_D2D, self.num_CH))
            for D2D_loop in range(self.num_D2D):
                for Neighbor_loop in range(self.num_Neighbor):
                    # Input: index[0] = target vehicle index, index[1] = neighbor index
                    index = [D2D_loop, Neighbor_loop]
                    [V2V_channel, V2V_interference, V2I_channel] = self.get_state(index)
                    Next_Train_D2D_CH_State[D2D_loop, Neighbor_loop, :] = V2V_channel
                    Next_Train_D2D_Interf_State[D2D_loop, Neighbor_loop, :] = V2V_interference
                Next_Train_D2D_V2I_CH_State[D2D_loop, :] = V2I_channel

            # compute the NEXT state
            D2D_Next_State = np.zeros((self.num_D2D, Num_One_D2D_Input))
            for D2D_loop in range(self.num_D2D):
                Current_CH_gain = np.reshape(Next_Train_D2D_CH_State[D2D_loop, :, :], [1, CH_gain_Index])
                D2D_Next_State[D2D_loop, 0:CH_gain_Index] = Current_CH_gain
                Current_Interf_gain = np.reshape(Next_Train_D2D_Interf_State[D2D_loop, :, :], [1, CH_Interf_Index])
                D2D_Next_State[D2D_loop, CH_gain_Index:2 * CH_gain_Index] = Current_Interf_gain
                Current_V2I_gain = Next_Train_D2D_V2I_CH_State[D2D_loop, :]
                D2D_Next_State[D2D_loop, 2 * CH_gain_Index:3 * CH_gain_Index] = Current_V2I_gain
                D2D_Next_State[D2D_loop, 3 * CH_gain_Index: Num_One_D2D_Input] = Train_D2D_Power_State[D2D_loop, :]

            # the NEXT State
            States_ = np.reshape(D2D_Next_State, [1, Num_States])

            # sample in (s, a, r, s_) format
            sample = [States, Actions, Reward, States_]

            # add the sample (or transition) to the Buffer
            self.train_observe(sample)

        # return the reward
        return Reward_Per_Transition

    def replay(self):
        # use samples from Replay memory to train the whole C-Decision model

        # define the number of states and actions in RL
        Num_RL_Actions = self.num_RL_Actions
        Num_D2D = self.num_D2D
        Num_One_D2D_Input = self.brain.num_One_D2D_Input
        BATCH_SIZE = self.batch_size
        GAMMA = self.gamma
        # read samples from memory
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        # get the states and the next states from current sample
        test_states = np.zeros((batchLen, Num_D2D, Num_One_D2D_Input))
        test_states_ = np.zeros((batchLen, Num_D2D, Num_One_D2D_Input))
        for Sample_loop in range(batchLen):
            test_states[Sample_loop, :, :] = np.reshape(batch[Sample_loop][0], [Num_D2D, Num_One_D2D_Input])
            if batch[Sample_loop][3] is None:
                test_states_[Sample_loop, :, :] = np.zeros((Num_D2D, Num_One_D2D_Input))
            else:
                test_states_[Sample_loop, :, :] = np.reshape(batch[Sample_loop][3], [Num_D2D, Num_One_D2D_Input])

        Num_Batch = batchLen
        # initialize Current state as  0
        D1_State = np.zeros([Num_Batch, Num_One_D2D_Input])
        D2_State = np.zeros([Num_Batch, Num_One_D2D_Input])
        D3_State = np.zeros([Num_Batch, Num_One_D2D_Input])
        D4_State = np.zeros([Num_Batch, Num_One_D2D_Input])

        # initialize Next state as  0
        D1_State_ = np.zeros([Num_Batch, Num_One_D2D_Input])
        D2_State_ = np.zeros([Num_Batch, Num_One_D2D_Input])
        D3_State_ = np.zeros([Num_Batch, Num_One_D2D_Input])
        D4_State_ = np.zeros([Num_Batch, Num_One_D2D_Input])

        for Sample_loop in range(batchLen):
            D1_State[Sample_loop, :] = test_states[Sample_loop, 0, :]
            D2_State[Sample_loop, :] = test_states[Sample_loop, 1, :]
            D3_State[Sample_loop, :] = test_states[Sample_loop, 2, :]
            D4_State[Sample_loop, :] = test_states[Sample_loop, 3, :]
            D1_State_[Sample_loop, :] = test_states_[Sample_loop, 0, :]
            D2_State_[Sample_loop, :] = test_states_[Sample_loop, 1, :]
            D3_State_[Sample_loop, :] = test_states_[Sample_loop, 2, :]
            D4_State_[Sample_loop, :] = test_states_[Sample_loop, 3, :]

        # current state
        states = {'D1_Input': D1_State, 'D2_Input': D2_State,
                  'D3_Input': D3_State, 'D4_Input': D4_State}
        # Next state
        states_ = {'D1_Input': D1_State_, 'D2_Input': D2_State_,
                   'D3_Input': D3_State_, 'D4_Input': D4_State_}

        # get predictions
        p = self.brain.predict(states)  # Q-function network
        p_ = self.brain.predict(states_, target=True)  # target network

        # record the approximate target value
        y = np.zeros((batchLen, Num_RL_Actions))

        # initialize these state as  0
        D1_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])
        D2_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])
        D3_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])
        D4_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])

        for batch_Loop in range(batchLen):
            # fetch current sample(observation) from Replay Buffer
            # observation = {S, A, R, S_}
            o = batch[batch_Loop]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            # initialize current action
            a_RL = 0

            # note that we treat a[0] is the high bit,  a[3] is the low bit HERE!
            for D_loop in range(Num_D2D):
                a_RL += a[0][D_loop]*(4**(Num_D2D - D_loop - 1))

            t = p[batch_Loop]

            if s_ is None:
                t[a_RL] = r
            else:
                # use target network to evaluate Q(s,a) value
                t[a_RL] = r + GAMMA * np.amax(p_[batch_Loop])

            test_s = np.reshape(s, [Num_D2D, Num_One_D2D_Input])

            D1_Data_Train[batch_Loop] = test_s[0, :]
            D2_Data_Train[batch_Loop] = test_s[1, :]
            D3_Data_Train[batch_Loop] = test_s[2, :]
            D4_Data_Train[batch_Loop] = test_s[3, :]

            y[batch_Loop] = t

        # use the current samples to train RL DNN
        x = {'D1_Input': D1_Data_Train, 'D2_Input': D2_Data_Train,
             'D3_Input': D3_Data_Train, 'D4_Input': D4_Data_Train}

        # compute the Q mean and Q max mean of the target value respectively to evaluate the training progress
        # y: [batchLen x Num_RL_Actions]
        Q_batch = np.sum(y, axis=1) / Num_RL_Actions
        Q_mean = np.sum(Q_batch) / batchLen
        Q_max_batch = np.max(y, axis=1)
        Q_max_mean = np.sum(Q_max_batch)/batchLen

        # compute the mean and max value of the original Q function
        Orig_Q = np.sum(p, axis=1) / Num_RL_Actions
        Orig_Q_mean = np.sum(Orig_Q) / batchLen

        Orig_Q_max_batch = np.max(p, axis=1)
        Orig_Q_max_mean = np.sum(Orig_Q_max_batch)/batchLen

        # train the whole C-Decision model
        Train_Result = self.brain.train_dnn(x, y, BATCH_SIZE)

        return Train_Result, Q_mean, Q_max_mean, Orig_Q_mean, Orig_Q_max_mean

    def train(self, num_episodes, num_train_steps):
        # to train the C-Decision model

        self.num_Episodes = num_episodes
        self.num_Train_Step = num_train_steps
        BATCH_SIZE = self.batch_size
        GAMMA = self.gamma
        self.num_step = 0
        # make several transitions before each training
        self.num_transition = 50

        # record the training loss
        Train_Loss = np.ones((num_episodes, num_train_steps))
        # record the change of Target Q function
        Train_Q_mean = np.zeros((num_episodes, num_train_steps))
        Train_Q_max_mean = np.zeros((num_episodes, num_train_steps))
        # record the mean and max value of the origin Q function
        Orig_Train_Q_mean = np.zeros((num_episodes, num_train_steps))
        Orig_Train_Q_max_mean = np.zeros((num_episodes, num_train_steps))
        # record the reward per episode
        Reward_Per_Episode = np.zeros(num_episodes)
        Reward_Per_Train_Step = np.zeros((num_episodes, num_train_steps, self.num_transition))

        # track the simulation settings
        current_datetime = datetime.datetime.now()
        print(current_datetime.strftime('%Y/%m/%d %H:%M:%S'))
        print("Training Parameters Settings in the Train Function are: ")
        print('Number of feedback: ', self.num_Feedback)
        print('Discount Factor Gamma: ', GAMMA)
        print('Batch Size: ', BATCH_SIZE)
        print('Training Episodes: ', self.num_Episodes)
        print('Train Steps per Episode: ', self.num_Train_Step)
        print('V2V Rate weight: ', self.v2v_weight)
        print('V2I Rate weight: ', self.v2i_weight)

        V2I_Weight = self.v2i_weight

        # tracking the simulation
        Train_Episode_Interval = 20
        Train_Step_Interval = 10
        Save_Model_Interval = 5

        # save results in their corresponding simulation parameter settings
        curr_sim_set = 'Train-Result' + '-Feedback-' + str(self.num_Feedback) + '-BatchSize-' + str(BATCH_SIZE) \
                       + '-Gamma-' + str(GAMMA) + '-V2Iweight-' + str(V2I_Weight)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            print('Create the new folder while training to save results : ')
            print('   --->:', folder)

        curr_Result_Dir = folder

        for Episode_loop in range(self.num_Episodes):

            # start a new game for each episode
            self.env.new_random_game(self.num_D2D)

            # tracking the training process
            if (Episode_loop + 1) % Train_Episode_Interval == 0:
                Current_DateTime = datetime.datetime.now()
                print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                print('Current Training Episode: ', Episode_loop + 1, ' / Total Training Episodes:', self.num_Episodes)

            for Iteration_loop in range(self.num_Train_Step):

                # Tracking the simulation in each episode
                if (Episode_loop + 1) % Train_Episode_Interval == 0 and (Iteration_loop + 1) % Train_Step_Interval == 0:
                    Current_DateTime = datetime.datetime.now()
                    print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                    print('Current Training Step: ', Iteration_loop + 1, ' / Total Training Steps:', self.num_Train_Step)

                # make several transitions then begin training
                # take action and observe the transition (s, a, r, s_), then add it to the Buffer
                Reward_Per_Transition = self.generate_d2d_transition(self.num_transition)
                # record the reward per train step
                Reward_Per_Train_Step[Episode_loop, Iteration_loop, :] = Reward_Per_Transition

                # train the C-Decision model
                [Train_Result, Q_mean, Q_max_mean, Orig_Q_mean, Orig_Q_max_mean] = self.replay()

                # record the train loss
                Train_Loss[Episode_loop, Iteration_loop] = Train_Result.history['loss'][0]
                # record Q mean and Q_max mean of Target value respectively
                Train_Q_mean[Episode_loop, Iteration_loop] = Q_mean
                Train_Q_max_mean[Episode_loop, Iteration_loop] = Q_max_mean
                # record mean and max of original Q function
                Orig_Train_Q_mean[Episode_loop, Iteration_loop] = Orig_Q_mean
                Orig_Train_Q_max_mean[Episode_loop, Iteration_loop] = Orig_Q_max_mean

                # update target network
                if self.num_step % UPDATE_TARGET_FREQUENCY == 0:
                    self.brain.update_target_model()

            # compute the total reward for each episode
            Reward_Per_Episode[Episode_loop] = np.sum(Reward_Per_Train_Step[Episode_loop, :, :])

            # Save the model's weights of Q-Function Network and Target Network
            if (Episode_loop + 1) % Save_Model_Interval == 0:

                # record the current episode index
                Curr_Train_Episode = Episode_loop + 1

                model_dir = curr_Result_Dir
                model_name = 'Q-Network_model_weights' + '-Episode-' + str(Curr_Train_Episode) \
                             + '-Step-' + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
                model_para = model_dir + model_name
                # save the weights of Q-Function Network
                self.brain.model.save_weights(model_para)
                print('Save Q-Function Network model weights after Training at Episode :', Curr_Train_Episode)
                # save the Target Network's weights in case we need it
                target_model_name = 'Target-Network_model_weights' + '-Episode-' + str(Curr_Train_Episode) \
                                    + '-Step-' + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
                target_model_para = model_dir + target_model_name
                self.brain.target_model.save_weights(target_model_para)
                print('Save Target Network model weights after Training at Episode :', Curr_Train_Episode)

                # save current train loss and Q values
                Curr_Train_Loss_per_Episode = np.sum(Train_Loss, axis=1) / Curr_Train_Episode
                Curr_Train_Q_mean_per_Episode = np.sum(Train_Q_mean, axis=1) / Curr_Train_Episode
                Curr_Train_Q_max_mean_per_Episode = np.sum(Train_Q_max_mean, axis=1) / Curr_Train_Episode
                # original Q value
                Curr_Orig_Train_Q_mean_per_Episode = np.sum(Orig_Train_Q_mean, axis=1) / Curr_Train_Episode
                Curr_Orig_Train_Q_max_mean_per_Episode = np.sum(Orig_Train_Q_max_mean, axis=1) / Curr_Train_Episode

                Data_Dir = curr_Result_Dir
                Data_Name = 'Temp-Training-Result' + '-Episode-' + str(Curr_Train_Episode) \
                            + '-Step-' + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.pkl'
                Data_Para = Data_Dir + Data_Name
                # open data file
                file_to_open = open(Data_Para, 'wb')
                # write current train results to data file
                pickle.dump((Curr_Train_Episode,
                             Curr_Train_Loss_per_Episode, Train_Loss,
                             Curr_Train_Q_mean_per_Episode, Curr_Train_Q_max_mean_per_Episode,
                             Curr_Orig_Train_Q_mean_per_Episode, Curr_Orig_Train_Q_max_mean_per_Episode,
                             Reward_Per_Train_Step, Reward_Per_Episode), file_to_open)
                file_to_open.close()

        # return the train loss
        return Train_Loss, Reward_Per_Train_Step, Reward_Per_Episode, \
               Train_Q_mean, Train_Q_max_mean, Orig_Train_Q_mean, Orig_Train_Q_max_mean

    def generate_d2d_initial_states(self):
        # generate initial states for RL to run
        Train_D2D_CH_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
        Train_D2D_Interf_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
        Train_D2D_V2I_CH_State = np.zeros((self.num_D2D, self.num_CH))
        Fixed_Power = self.env.V2V_power_dB_List[self.env.fixed_v2v_power_index]
        Train_D2D_Power_State = Fixed_Power * np.ones((self.num_D2D, self.num_Neighbor))

        # Get all D2D channel and interference state for Testing
        for D2D_loop in range(self.num_D2D):
            for Neighbor_loop in range(self.num_Neighbor):
                # Input: indx[0] = target vehicle index, indx[1] = neighbor index
                index = [D2D_loop, Neighbor_loop]
                [V2V_channel, V2V_interference, V2I_channel] = self.get_state(index)
                Train_D2D_CH_State[D2D_loop, Neighbor_loop, :] = V2V_channel
                Train_D2D_Interf_State[D2D_loop, Neighbor_loop, :] = V2V_interference
            Train_D2D_V2I_CH_State[D2D_loop, :] = V2I_channel

        # reshape the training data in (S, A, R, S_)
        # reshape the States for all D2D
        Num_One_D2D_Input = self.brain.num_One_D2D_Input
        D2D_State = np.zeros((self.num_D2D, Num_One_D2D_Input))
        CH_gain_Index = self.num_Neighbor*self.num_CH
        CH_Interf_Index = self.num_Neighbor*self.num_CH

        for D2D_loop in range(self.num_D2D):
            Current_CH_gain = np.reshape(Train_D2D_CH_State[D2D_loop, :, :], [1, CH_gain_Index])
            D2D_State[D2D_loop, 0:CH_gain_Index] = Current_CH_gain
            Current_Interf_gain = np.reshape(Train_D2D_Interf_State[D2D_loop, :, :], [1, CH_Interf_Index])
            D2D_State[D2D_loop, CH_gain_Index:2*CH_gain_Index] = Current_Interf_gain
            Current_V2I_gain = Train_D2D_V2I_CH_State[D2D_loop, :]
            D2D_State[D2D_loop, 2 * CH_gain_Index:3 * CH_gain_Index] = Current_V2I_gain
            D2D_State[D2D_loop, 3 * CH_gain_Index: Num_One_D2D_Input] = Train_D2D_Power_State[D2D_loop, :]

        D1_Initial_State = np.reshape(D2D_State[0, :], [1, Num_One_D2D_Input])
        D2_Initial_State = np.reshape(D2D_State[1, :], [1, Num_One_D2D_Input])
        D3_Initial_State = np.reshape(D2D_State[2, :], [1, Num_One_D2D_Input])
        D4_Initial_State = np.reshape(D2D_State[3, :], [1, Num_One_D2D_Input])

        Initial_States = {'D1_Input': D1_Initial_State, 'D2_Input': D2_Initial_State,
                          'D3_Input': D3_Initial_State, 'D4_Input': D4_Initial_State}

        return Initial_States

    def run(self, num_episodes, num_test_step, opt_flag):
        # define run() to test the trained C-Decision model
        self.num_Episodes = num_episodes
        self.num_Test_Step = num_test_step

        # weight for the V2V sum rate
        v2v_weight = self.v2v_weight
        # weight for the V2I sum rate
        v2i_weight = self.v2i_weight

        # initialize variables to save the results
        Expect_Return = np.zeros(self.num_Episodes)
        Reward = np.zeros((self.num_Episodes, self.num_Test_Step))
        Per_V2V_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_D2D))
        Per_V2I_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))
        Per_V2B_Interference = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))

        # add the comparing scheme:
        # Random Action scheme: RA, where each D2D chooses its own action randomly
        RA_Flag = True
        RA_Expect_Return = np.zeros(self.num_Episodes)
        RA_Reward = np.zeros((self.num_Episodes, self.num_Test_Step))
        RA_Per_V2V_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_D2D))
        RA_Per_V2I_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))
        RA_Per_V2B_Interference = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))

        # implement Optimal Scheme (Opt) via Brute Force
        Opt_Flag = opt_flag
        if Opt_Flag:
            Opt_D2D_Action_Index = np.zeros((self.num_Episodes, self.num_Test_Step))
            Opt_Expect_Return = np.zeros(self.num_Episodes)
            Opt_Reward = np.zeros((self.num_Episodes, self.num_Test_Step))
            Opt_Per_V2V_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_D2D))
            Opt_Per_V2I_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))
            Opt_Per_V2B_Interference = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))

        # tracking the simulation
        Run_Episode_Interval = 20
        Run_Step_Interval = 50

        for Episode_loop in range(self.num_Episodes):

            # start a new game for each episode
            self.env.new_random_game(self.num_D2D)

            # Generate the states
            Initial_State = self.generate_d2d_initial_states()
            States = Initial_State

            # tracking the simulation
            if (Episode_loop + 1) % Run_Episode_Interval == 0:
                Current_DateTime = datetime.datetime.now()
                print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                print('Current Running Episode: ', Episode_loop + 1, ' / Total Running Episodes:', self.num_Episodes)

            for Run_loop in range(self.num_Test_Step):

                # compute the comparison schemes firstly
                if RA_Flag:
                    RA_D2D_Action = self.select_action_random(States)
                    # RA_V2V_Rate = self.act(RA_D2D_Action)
                    # Just compute the reward Not update the states
                    [RA_V2V_Rate, V2I_Rate, Interference] = self.dump_act(RA_D2D_Action)
                    # adopt weighted sum rate as the reward
                    Sum_V2I_Rate = np.sum(V2I_Rate)
                    Sum_V2V_Rate = np.sum(RA_V2V_Rate)
                    RA_D2D_Reward = v2v_weight*Sum_V2V_Rate + v2i_weight*Sum_V2I_Rate

                    RA_Reward[Episode_loop, Run_loop] = RA_D2D_Reward  # total reward
                    # RA: Calculate the Expected Return
                    RA_Expect_Return[Episode_loop] += RA_Reward[Episode_loop, Run_loop]
                    # record the related variables
                    RA_Per_V2V_Rate[Episode_loop, Run_loop, :] = np.sum(RA_V2V_Rate, axis=1)
                    RA_Per_V2I_Rate[Episode_loop, Run_loop, :] = V2I_Rate
                    RA_Per_V2B_Interference[Episode_loop, Run_loop, :] = Interference

                if Opt_Flag:
                    # implement Optimal scheme via Brute Force Search

                    # initialize variables
                    Num_Possisble_Action = self.num_CH ** self.num_D2D
                    Curr_Feasible_Reward = np.zeros(Num_Possisble_Action)
                    BF_V2V_Rate = np.zeros((Num_Possisble_Action, self.num_D2D))
                    BF_V2I_Rate = np.zeros((Num_Possisble_Action, self.num_CH))
                    BF_Interference = np.zeros((Num_Possisble_Action, self.num_CH))

                    for BF_loop in range(self.brain.num_Output):
                        # change the RL_Actions [0,255] to D2D actions [a, a, a, a] where a in {0,1,2,3}
                        D2D_Action = np.zeros(self.num_D2D, int)
                        n = BF_loop
                        a0 = n // (4 ** 3)
                        a1 = (n % (4 ** 3)) // (4 ** 2)
                        a2 = (n % (4 ** 2)) // (4 ** 1)
                        a3 = n % (4 ** 1)
                        D2D_Action[0] = a0
                        D2D_Action[1] = a1
                        D2D_Action[2] = a2
                        D2D_Action[3] = a3
                        Curr_D2D_Action = np.reshape(D2D_Action, [self.num_D2D, 1])
                        # Take action and Get Reward
                        # Just compute the reward Not update the states
                        [V2V_Rate, V2I_Rate, Interference] = self.dump_act(Curr_D2D_Action)
                        # adopt weighted sum rate as the reward
                        Sum_V2I_Rate = np.sum(V2I_Rate)
                        Sum_V2V_Rate = np.sum(V2V_Rate)
                        # record the current reward
                        Curr_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                        # record all related information
                        Curr_Feasible_Reward[BF_loop] = Curr_Reward
                        BF_V2V_Rate[BF_loop, :] = np.sum(V2V_Rate, axis=1)
                        BF_V2I_Rate[BF_loop, :] = V2I_Rate
                        BF_Interference[BF_loop, :] = Interference

                    Curr_Opt_Reward = np.max(Curr_Feasible_Reward)
                    # find the optimal reward and action, record the related variables
                    if Curr_Opt_Reward > 0:
                        Curr_Opt_Act_Index = np.argmax(Curr_Feasible_Reward)
                        Opt_Reward[Episode_loop, Run_loop] = Curr_Opt_Reward
                        Opt_Expect_Return[Episode_loop] += Opt_Reward[Episode_loop, Run_loop]
                        Opt_D2D_Action_Index[Episode_loop, Run_loop] = Curr_Opt_Act_Index
                        Curr_Opt_V2V_Rate = BF_V2V_Rate[Curr_Opt_Act_Index, :]
                        Curr_Opt_V2I_Rate = BF_V2I_Rate[Curr_Opt_Act_Index, :]
                        Curr_Opt_Interference = BF_Interference[Curr_Opt_Act_Index, :]
                        Opt_Per_V2V_Rate[Episode_loop, Run_loop, :] = Curr_Opt_V2V_Rate
                        Opt_Per_V2I_Rate[Episode_loop, Run_loop, :] = Curr_Opt_V2I_Rate
                        Opt_Per_V2B_Interference[Episode_loop, Run_loop, :] = Curr_Opt_Interference

                # run the C-Decision scheme

                # Generate Q(Stats,a) via putting the States into the trained model
                Q_Pred = self.brain.predict_one_step(States)

                # Find the RL-actions to maximize Q value
                Action_Max = np.where(Q_Pred == np.max(Q_Pred))

                RL_Action = Action_Max[1][0]

                # Get the D2D Actions
                # change the RL_Actions [0,255] to D2D actions [a, a, a, a] where a in {0,1,2,3}
                D2D_Action = np.zeros(self.num_D2D, int)
                # Change  a_RL (Decimal)  to a (Quaternary)
                n = RL_Action
                a0 = n // (4 ** 3)
                a1 = (n % (4 ** 3)) // (4 ** 2)
                a2 = (n % (4 ** 2)) // (4 ** 1)
                a3 = n % (4 ** 1)
                D2D_Action[0] = a0
                D2D_Action[1] = a1
                D2D_Action[2] = a2
                D2D_Action[3] = a3
                Actions = np.reshape(D2D_Action, [self.num_D2D, 1])

                # Take action and Get Reward
                # compute the reward and update to the “Next state”
                [V2V_Rate, V2I_Rate, Interference] = self.act(Actions)

                # adopt weighted sum rate as the reward
                Sum_V2I_Rate = np.sum(V2I_Rate)
                Sum_V2V_Rate = np.sum(V2V_Rate)
                D2D_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                Reward[Episode_loop, Run_loop] = D2D_Reward  # total reward
                Per_V2V_Rate[Episode_loop, Run_loop, :] = np.sum(V2V_Rate, axis=1)
                Per_V2I_Rate[Episode_loop, Run_loop, :] = V2I_Rate
                Per_V2B_Interference[Episode_loop, Run_loop, :] = Interference

                # Calculate the Expected Return, currently no discount
                Expect_Return[Episode_loop] += Reward[Episode_loop, Run_loop]

                # Tracking the simulation
                if (Episode_loop + 1) % Run_Episode_Interval == 0 and (Run_loop + 1) % Run_Step_Interval == 0:
                    Current_DateTime = datetime.datetime.now()
                    print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                    print('Current Running Step: ', Run_loop + 1, ' / Total Running Steps:', self.num_Test_Step)

                # Get Next State
                States = self.generate_d2d_initial_states()

        if RA_Flag and Opt_Flag:
            print('Finish Testing the C-Decision scheme with Real Feedback with Optimal Scheme!')
            return Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate, \
                   Per_V2B_Interference, \
                   RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate, \
                   RA_Per_V2B_Interference, \
                   Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate, \
                   Opt_Per_V2B_Interference
        else:
            if RA_Flag:
                print('Finish Testing the C-Decision scheme with Real Feedback '
                      ' without Implementing Optimal Scheme!')
                return Expect_Return, Reward, \
                       Per_V2V_Rate, Per_V2I_Rate, \
                       Per_V2B_Interference, \
                       RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate, \
                       RA_Per_V2B_Interference
            else:
                print('Finish Testing only the C-Decision scheme with Real Feedback')
                return Expect_Return, Reward, \
                       Per_V2V_Rate, Per_V2I_Rate, Per_V2B_Interference

    def robust_run(self, num_episodes, num_test_step, opt_flag, feedback_interval, input_noise_level, fb_noise_level):
        # define robust_run() to test the robustness of the trained model
        self.num_Episodes = num_episodes
        self.num_Test_Step = num_test_step

        # weight for the V2V sum rate
        v2v_weight = self.v2v_weight
        # weight for the V2I sum rate
        v2i_weight = self.v2i_weight

        # test the impact of different feedback update interval on the performance
        # feedback interval  i.e. 3 refers to every 3 slot only feedback once
        Feedback_Interval = feedback_interval
        # test the impact of noisy inputs on the performance
        Input_Noise_Level = input_noise_level

        # variables to add the noise to input
        Num_One_D2D_Input = self.brain.num_One_D2D_Input
        Noisy_D2D_States = np.zeros((self.num_D2D, Num_One_D2D_Input))
        CH_gain_Index = self.num_Neighbor * self.num_CH

        # test the impact of noisy feedback on the performance
        FB_Noise_Level = fb_noise_level

        # construct models to get the output of each D2D
        # then construct new BS model using noisy feedback as input
        # get the output of intermediate layer
        model = self.brain.model

        # define the model for each D2D
        D1_layer_name = 'D1_Output'
        D1_output_model = Model(inputs=model.input, outputs=model.get_layer(D1_layer_name).output)

        D2_layer_name = 'D2_Output'
        D2_output_model = Model(inputs=model.input, outputs=model.get_layer(D2_layer_name).output)

        D3_layer_name = 'D3_Output'
        D3_output_model = Model(inputs=model.input, outputs=model.get_layer(D3_layer_name).output)

        D4_layer_name = 'D4_Output'
        D4_output_model = Model(inputs=model.input, outputs=model.get_layer(D4_layer_name).output)

        # dictionary to store all D2D output model
        D2D_Output_Model = {'D1_model': D1_output_model, 'D2_model': D2_output_model,
                            'D3_model': D3_output_model, 'D4_model': D4_output_model}

        # temp variables to implement Noisy Feedback
        D2D_Output = np.zeros((self.num_D2D, self.brain.num_Feedback))
        Noisy_D2D_Output = np.zeros((self.num_D2D, self.brain.num_Feedback))
        Num_Feedback = self.brain.num_Feedback
        Num_D2D = self.num_D2D
        Num_BS_Output = self.brain.num_Output
        # define a new model using  BS_Input as input
        input_shape = Num_Feedback * Num_D2D
        BS_Input = Input(shape=(input_shape,), name='BS_Input')
        BS_DNN_1 = Dense(1200, activation='relu', name='BS_DNN_1')(BS_Input)
        BS_DNN_2 = Dense(800, activation='relu', name='BS_DNN_2')(BS_DNN_1)
        BS_DNN_3 = Dense(600, activation='relu', name='BS_DNN_3')(BS_DNN_2)

        # finally add the main logistic regression layer
        BS_output = Dense(Num_BS_Output, activation='linear', name='BS_output')(BS_DNN_3)
        # Define the model
        BS_model = Model(inputs=[BS_Input], outputs=[BS_output])  # output is not correct !!

        # the default value of learning rate lr=0.001, change it if necessary
        rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        # Compile the model
        BS_model.compile(optimizer=rms, loss=huber_loss)

        # get weights from trained model
        BS_DNN_1_weights = model.get_layer('BS_Dense1').get_weights()
        BS_DNN_2_weights = model.get_layer('BS_Dense2').get_weights()
        BS_DNN_3_weights = model.get_layer('BS').get_weights()
        BS_output_weights = model.get_layer('BS_output').get_weights()

        # load weights for each layer of BS_model
        BS_model.get_layer('BS_DNN_1').set_weights(BS_DNN_1_weights)
        BS_model.get_layer('BS_DNN_2').set_weights(BS_DNN_2_weights)
        BS_model.get_layer('BS_DNN_3').set_weights(BS_DNN_3_weights)
        BS_model.get_layer('BS_output').set_weights(BS_output_weights)

        # define the new model
        BS_output_model = Model(inputs=BS_model.input,
                                outputs=BS_model.output)

        # initialize variables to save the results
        Expect_Return = np.zeros(self.num_Episodes)
        Reward = np.zeros((self.num_Episodes, self.num_Test_Step))
        # record the related variables to compress the V2B interference
        Per_V2V_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_D2D))
        Per_V2I_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))
        Per_V2B_Interference = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))

        # add the comparing scheme:
        # Random Action scheme: RA, where each D2D chooses its own action randomly
        RA_Flag = True
        RA_Expect_Return = np.zeros(self.num_Episodes)
        RA_Reward = np.zeros((self.num_Episodes, self.num_Test_Step))
        # record the related variables to compress the V2B interference
        RA_Per_V2V_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_D2D))
        RA_Per_V2I_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))
        RA_Per_V2B_Interference = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))

        # implement Optimal Scheme (Opt) via Brute Force Search
        Opt_Flag = opt_flag
        if Opt_Flag:
            Opt_D2D_Action_Index = np.zeros((self.num_Episodes, self.num_Test_Step))
            Opt_Expect_Return = np.zeros(self.num_Episodes)
            Opt_Reward = np.zeros((self.num_Episodes, self.num_Test_Step))
            # record the related variables to compress the V2B interference
            Opt_Per_V2V_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_D2D))
            Opt_Per_V2I_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))
            Opt_Per_V2B_Interference = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))

        # tracking the simulation
        Run_Episode_Interval = 20
        Run_Step_Interval = 50

        for Episode_loop in range(self.num_Episodes):

            # start a new game for each episode
            self.env.new_random_game(self.num_D2D)

            # Generate the states
            Initial_State = self.generate_d2d_initial_states()
            States = Initial_State

            # tracking the simulation
            if (Episode_loop + 1) % Run_Episode_Interval == 0:
                Current_DateTime = datetime.datetime.now()
                print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                print('Current Running Episode: ', Episode_loop + 1, ' / Total Running Episodes:', self.num_Episodes)

            for Run_loop in range(self.num_Test_Step):

                # optimal scheme uses the exact states without any errors
                if Opt_Flag:
                    # implement Optimal scheme via Brute Force Search

                    # initialize variables
                    Num_Possisble_Action = self.num_CH ** self.num_D2D
                    Curr_Feasible_Reward = np.zeros(Num_Possisble_Action)
                    BF_V2V_Rate = np.zeros((Num_Possisble_Action, self.num_D2D))
                    BF_V2I_Rate = np.zeros((Num_Possisble_Action, self.num_CH))
                    BF_Interference = np.zeros((Num_Possisble_Action, self.num_CH))

                    for BF_loop in range(self.brain.num_Output):
                        # change the RL_Actions [0,255] to D2D actions [a, a, a, a] where a in {0,1,2,3}
                        D2D_Action = np.zeros(self.num_D2D, int)
                        n = BF_loop
                        a0 = n // (4 ** 3)
                        a1 = (n % (4 ** 3)) // (4 ** 2)
                        a2 = (n % (4 ** 2)) // (4 ** 1)
                        a3 = n % (4 ** 1)
                        D2D_Action[0] = a0
                        D2D_Action[1] = a1
                        D2D_Action[2] = a2
                        D2D_Action[3] = a3
                        Curr_D2D_Action = np.reshape(D2D_Action, [self.num_D2D, 1])
                        # Take action and Get Reward
                        # Just compute the reward Not update the states
                        [V2V_Rate, V2I_Rate, Interference] = self.dump_act(Curr_D2D_Action)

                        # record current reward
                        # adopt weighted sum rate as the reward
                        Sum_V2I_Rate = np.sum(V2I_Rate)
                        Sum_V2V_Rate = np.sum(V2V_Rate)
                        Curr_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                        # record all related information
                        Curr_Feasible_Reward[BF_loop] = Curr_Reward
                        BF_V2V_Rate[BF_loop, :] = np.sum(V2V_Rate, axis=1)
                        BF_V2I_Rate[BF_loop, :] = V2I_Rate
                        BF_Interference[BF_loop, :] = Interference

                    Curr_Opt_Reward = np.max(Curr_Feasible_Reward)
                    # here Curr_Opt_Reward > 0 means that there exists a feasible solution
                    if Curr_Opt_Reward > 0:
                        Curr_Opt_Act_Index = np.argmax(Curr_Feasible_Reward)
                        Opt_Reward[Episode_loop, Run_loop] = Curr_Opt_Reward
                        Opt_Expect_Return[Episode_loop] += Opt_Reward[Episode_loop, Run_loop]
                        Opt_D2D_Action_Index[Episode_loop, Run_loop] = Curr_Opt_Act_Index
                        Curr_Opt_V2V_Rate = BF_V2V_Rate[Curr_Opt_Act_Index, :]
                        Curr_Opt_V2I_Rate = BF_V2I_Rate[Curr_Opt_Act_Index, :]
                        Curr_Opt_Interference = BF_Interference[Curr_Opt_Act_Index, :]
                        Opt_Per_V2V_Rate[Episode_loop, Run_loop, :] = Curr_Opt_V2V_Rate
                        Opt_Per_V2I_Rate[Episode_loop, Run_loop, :] = Curr_Opt_V2I_Rate
                        Opt_Per_V2B_Interference[Episode_loop, Run_loop, :] = Curr_Opt_Interference

                if Input_Noise_Level == 0:
                    # compute the comparison schemes first
                    if RA_Flag:
                        RA_D2D_Action = self.select_action_random(States)
                        # Just compute the reward Not update the states
                        [RA_V2V_Rate, V2I_Rate, Interference] = self.dump_act(RA_D2D_Action)
                        # adopt weighted sum rate as the reward
                        Sum_V2I_Rate = np.sum(V2I_Rate)
                        Sum_V2V_Rate = np.sum(RA_V2V_Rate)
                        RA_D2D_Reward = v2v_weight*Sum_V2V_Rate + v2i_weight*Sum_V2I_Rate
                        RA_Reward[Episode_loop, Run_loop] = RA_D2D_Reward  # total reward
                        # RA: Calculate the Expected Return
                        RA_Expect_Return[Episode_loop] += RA_Reward[Episode_loop, Run_loop]

                        RA_Per_V2V_Rate[Episode_loop, Run_loop, :] = np.sum(RA_V2V_Rate, axis=1)
                        RA_Per_V2I_Rate[Episode_loop, Run_loop, :] = V2I_Rate
                        RA_Per_V2B_Interference[Episode_loop, Run_loop, :] = Interference

                # judge whether current step belongs to feedback updating step or not
                if Run_loop % Feedback_Interval == 0:
                    # feedback updating step, take action on the current state

                    if Input_Noise_Level == 0:
                        # without input noise

                        if FB_Noise_Level > 0:
                            # use current states to get the output of each D2D, i.e., feedback
                            # add noise to the feedback of each D2D according to the FB_Noise_Level
                            data = States
                            for D2D_loop in range(self.num_D2D):
                                # get current D2D output layer name
                                Curr_D2D_Index = D2D_loop + 1
                                model_name = 'D' + str(Curr_D2D_Index) + '_model'
                                Current_model = D2D_Output_Model[model_name]
                                D2D_Output[D2D_loop, :] = Current_model.predict(data)

                                # add feedback noise
                                for fb_loop in range(self.brain.num_Feedback):
                                    Curr_FB_Noise_dev = np.abs(D2D_Output[D2D_loop, fb_loop]) * FB_Noise_Level
                                    Curr_FB_Noise = np.random.normal(0, Curr_FB_Noise_dev, 1)
                                    Noisy_D2D_Output[D2D_loop, fb_loop] = D2D_Output[D2D_loop, fb_loop] + Curr_FB_Noise

                            # treat the noisy D2D output as new input of BS, termed as Noisy BS Input
                            Noisy_BS_Input = np.reshape(Noisy_D2D_Output, [1, Num_Feedback * Num_D2D])
                            data = {'BS_Input': Noisy_BS_Input}

                            # use the noisy feedback to get the noisy BS output
                            Noisy_BS_Output = BS_output_model.predict(data)

                            # use the current noisy BS output as the current Q_Pred
                            Q_Pred = Noisy_BS_Output
                        else:
                            # no feedback and input noise just as usual
                            # Generate Q(Stats,a) via putting the States into the trained model
                            Q_Pred = self.brain.predict_one_step(States)
                    else:
                        # add noise to the input states
                        if FB_Noise_Level == 0:
                            # without feedback noise just with input noise

                            Curr_D2D_CH_Gain = np.zeros((self.num_D2D, self.num_CH))
                            Curr_D2D_Interference = np.zeros((self.num_D2D, self.num_CH))
                            Curr_D2D_V2I_CH_Gain = np.zeros((self.num_D2D, self.num_CH))
                            Curr_D2D_Power = np.zeros((self.num_D2D, Num_One_D2D_Input - 3 * self.num_CH))
                            # add noise to current states
                            for D_loop in range(self.num_D2D):
                                State_Index = 'D' + str(D_loop + 1) + '_Input'
                                # find the current D2D channel gain vector from current "States"
                                Curr_D2D_CH_Gain[D_loop, :] = States[State_Index][0][0:self.num_CH]
                                Curr_D2D_Interference[D_loop, :] = States[State_Index][0][self.num_CH:2 * self.num_CH]
                                Curr_D2D_V2I_CH_Gain[D_loop, :] = States[State_Index][0][2 * self.num_CH:3 * self.num_CH]
                                Curr_D2D_Power[D_loop] = States[State_Index][0][3 * self.num_CH]
                                for CH_loop in range(self.num_CH):
                                    # add noise to channel gain
                                    Curr_CH_Noise_dev = np.abs(Curr_D2D_CH_Gain[D_loop, CH_loop]) * Input_Noise_Level
                                    Curr_CH_Noise = np.random.normal(0, Curr_CH_Noise_dev, 1)
                                    Noisy_D2D_States[D_loop, CH_loop] = Curr_D2D_CH_Gain[
                                                                            D_loop, CH_loop] + Curr_CH_Noise
                                    # add noise to interference
                                    Curr_Interf_Noise_dev = np.abs(Curr_D2D_Interference[D_loop, CH_loop]) \
                                                             * Input_Noise_Level
                                    Curr_Interf_Noise = np.random.normal(0, Curr_Interf_Noise_dev, 1)
                                    Noisy_D2D_States[D_loop, CH_gain_Index + CH_loop] = \
                                        Curr_D2D_Interference[D_loop, CH_loop] + Curr_Interf_Noise
                                    # add noise to V2I CH gain
                                    Curr_V2I_CH_Noise_dev = np.abs(Curr_D2D_V2I_CH_Gain[D_loop, CH_loop]) \
                                                             * Input_Noise_Level
                                    Curr_V2I_CH_Noise = np.random.normal(0, Curr_V2I_CH_Noise_dev, 1)
                                    Noisy_D2D_States[D_loop, 2 * CH_gain_Index + CH_loop] = \
                                        Curr_D2D_V2I_CH_Gain[D_loop, CH_loop] + Curr_V2I_CH_Noise
                                # here since each D2D knows exact transmit power,
                                # there is no need to add the noise to D2D' transmit power
                                Noisy_D2D_States[D_loop, 3 * CH_gain_Index] = Curr_D2D_Power[D_loop]

                            # get the noisy state
                            D1_Noisy_State = np.reshape(Noisy_D2D_States[0, :], [1, Num_One_D2D_Input])
                            D2_Noisy_State = np.reshape(Noisy_D2D_States[1, :], [1, Num_One_D2D_Input])
                            D3_Noisy_State = np.reshape(Noisy_D2D_States[2, :], [1, Num_One_D2D_Input])
                            D4_Noisy_State = np.reshape(Noisy_D2D_States[3, :], [1, Num_One_D2D_Input])

                            Noisy_States = {'D1_Input': D1_Noisy_State, 'D2_Input': D2_Noisy_State,
                                            'D3_Input': D3_Noisy_State, 'D4_Input': D4_Noisy_State}

                            # Generate Q(Stats,a)
                            Q_Pred = self.brain.predict_one_step(Noisy_States)

                            # use the noisy input states as current states to evaluate comparing scheme
                            States = Noisy_States

                            # compute the comparison schemes under input noise
                            if RA_Flag:
                                RA_D2D_Action = self.select_action_random(States)
                                # Just compute the reward Not update the states
                                [RA_V2V_Rate, V2I_Rate, Interference] = self.dump_act(RA_D2D_Action)
                                # adopt weighted sum rate as the reward -
                                Sum_V2I_Rate = np.sum(V2I_Rate)
                                Sum_V2V_Rate = np.sum(RA_V2V_Rate)
                                RA_D2D_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                                RA_Reward[Episode_loop, Run_loop] = RA_D2D_Reward  # total reward
                                # RA: Calculate the Expected Return, currently no discount
                                RA_Expect_Return[Episode_loop] += RA_Reward[Episode_loop, Run_loop]

                                RA_Per_V2V_Rate[Episode_loop, Run_loop, :] = np.sum(RA_V2V_Rate, axis=1)
                                RA_Per_V2I_Rate[Episode_loop, Run_loop, :] = V2I_Rate
                                RA_Per_V2B_Interference[Episode_loop, Run_loop, :] = Interference
                        else:
                            # have both input and feedback noise
                            Curr_D2D_CH_Gain = np.zeros((self.num_D2D, self.num_CH))
                            Curr_D2D_Interference = np.zeros((self.num_D2D, self.num_CH))
                            Curr_D2D_V2I_CH_Gain = np.zeros((self.num_D2D, self.num_CH))
                            Curr_D2D_Power = np.zeros((self.num_D2D, Num_One_D2D_Input - 3 * self.num_CH))
                            # add noise to current states
                            for D_loop in range(self.num_D2D):
                                State_Index = 'D' + str(D_loop + 1) + '_Input'
                                # find the current D2D channel gain vector from current "States"
                                Curr_D2D_CH_Gain[D_loop, :] = States[State_Index][0][0:self.num_CH]
                                Curr_D2D_Interference[D_loop, :] = States[State_Index][0][self.num_CH:2 * self.num_CH]
                                # add V2I channel gain vector
                                Curr_D2D_V2I_CH_Gain[D_loop, :] = States[State_Index][0][
                                                                  2 * self.num_CH:3 * self.num_CH]
                                Curr_D2D_Power[D_loop] = States[State_Index][0][3 * self.num_CH]
                                for CH_loop in range(self.num_CH):
                                    # add noise to channel gain
                                    Curr_CH_Noise_dev = np.abs(Curr_D2D_CH_Gain[D_loop, CH_loop]) * Input_Noise_Level
                                    Curr_CH_Noise = np.random.normal(0, Curr_CH_Noise_dev, 1)
                                    Noisy_D2D_States[D_loop, CH_loop] = Curr_D2D_CH_Gain[
                                                                            D_loop, CH_loop] + Curr_CH_Noise
                                    # add noise to interference
                                    Curr_Interf_Noise_dev = np.abs(Curr_D2D_Interference[D_loop, CH_loop]) \
                                                            * Input_Noise_Level
                                    Curr_Interf_Noise = np.random.normal(0, Curr_Interf_Noise_dev, 1)
                                    Noisy_D2D_States[D_loop, CH_gain_Index + CH_loop] = \
                                        Curr_D2D_Interference[D_loop, CH_loop] + Curr_Interf_Noise
                                    # add noise to V2I CH gain
                                    Curr_V2I_CH_Noise_dev = np.abs(Curr_D2D_V2I_CH_Gain[D_loop, CH_loop]) \
                                                            * Input_Noise_Level
                                    Curr_V2I_CH_Noise = np.random.normal(0, Curr_V2I_CH_Noise_dev, 1)
                                    Noisy_D2D_States[D_loop, 2 * CH_gain_Index + CH_loop] = \
                                        Curr_D2D_V2I_CH_Gain[D_loop, CH_loop] + Curr_V2I_CH_Noise
                                # here since each D2D knows exact transmit power,
                                # there is no need to add the noise to D2D' transmit power
                                Noisy_D2D_States[D_loop, 3 * CH_gain_Index] = Curr_D2D_Power[D_loop]

                            # get the noisy input
                            D1_Noisy_State = np.reshape(Noisy_D2D_States[0, :], [1, Num_One_D2D_Input])
                            D2_Noisy_State = np.reshape(Noisy_D2D_States[1, :], [1, Num_One_D2D_Input])
                            D3_Noisy_State = np.reshape(Noisy_D2D_States[2, :], [1, Num_One_D2D_Input])
                            D4_Noisy_State = np.reshape(Noisy_D2D_States[3, :], [1, Num_One_D2D_Input])

                            Noisy_States = {'D1_Input': D1_Noisy_State, 'D2_Input': D2_Noisy_State,
                                            'D3_Input': D3_Noisy_State, 'D4_Input': D4_Noisy_State}

                            # use the noisy states as current states
                            data = Noisy_States

                            for D2D_loop in range(self.num_D2D):
                                # get current D2D output layer name
                                Curr_D2D_Index = D2D_loop + 1
                                model_name = 'D' + str(Curr_D2D_Index) + '_model'
                                Current_model = D2D_Output_Model[model_name]
                                D2D_Output[D2D_loop, :] = Current_model.predict(data)

                                # add feedback noise
                                for fb_loop in range(self.brain.num_Feedback):
                                    Curr_FB_Noise_dev = np.abs(D2D_Output[D2D_loop, fb_loop]) * FB_Noise_Level
                                    Curr_FB_Noise = np.random.normal(0, Curr_FB_Noise_dev, 1)
                                    Noisy_D2D_Output[D2D_loop, fb_loop] = D2D_Output[D2D_loop, fb_loop] + Curr_FB_Noise

                            # treat the noisy D2D output as new input of BS, termed as Noisy BS Input
                            Noisy_BS_Input = np.reshape(Noisy_D2D_Output, [1, Num_Feedback * Num_D2D])
                            data = {'BS_Input': Noisy_BS_Input}

                            # use the noisy feedback to get the noisy BS output
                            Noisy_BS_Output = BS_output_model.predict(data)

                            # use the current noisy BS output as the current Q_Pred
                            Q_Pred = Noisy_BS_Output

                    # Find the RL-actions to maximize Q value
                    Action_Max = np.where(Q_Pred == np.max(Q_Pred))
                    RL_Action = Action_Max[1][0]

                    # Get the D2D Actions
                    D2D_Action = np.zeros(self.num_D2D, int)
                    # Change  a_RL (Decimal)  to a (Quaternary)
                    n = RL_Action
                    a0 = n // (4 ** 3)
                    a1 = (n % (4 ** 3)) // (4 ** 2)
                    a2 = (n % (4 ** 2)) // (4 ** 1)
                    a3 = n % (4 ** 1)
                    D2D_Action[0] = a0
                    D2D_Action[1] = a1
                    D2D_Action[2] = a2
                    D2D_Action[3] = a3
                    Actions = np.reshape(D2D_Action, [self.num_D2D, 1])

                    # Take action and Get Reward
                    [V2V_Rate, V2I_Rate, Interference] = self.act(Actions)

                    # adopt weighted sum rate as the reward
                    Sum_V2I_Rate = np.sum(V2I_Rate)
                    Sum_V2V_Rate = np.sum(V2V_Rate)
                    D2D_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                    Reward[Episode_loop, Run_loop] = D2D_Reward  # total reward

                    Per_V2V_Rate[Episode_loop, Run_loop, :] = np.sum(V2V_Rate, axis=1)
                    Per_V2I_Rate[Episode_loop, Run_loop, :] = V2I_Rate
                    Per_V2B_Interference[Episode_loop, Run_loop, :] = Interference

                    # record the actions for non feedback updating step
                    Curr_FB_Action = Actions

                else:

                    # Non feedback updating step
                    # adopt the action of previous feedback updating step on the current state

                    # compute the reward under the previous feedback updating step
                    # and update to the “Next state”
                    [V2V_Rate, V2I_Rate, Interference] = self.act(Curr_FB_Action)

                    # adopt weighted sum rate as the reward
                    Sum_V2I_Rate = np.sum(V2I_Rate)
                    Sum_V2V_Rate = np.sum(V2V_Rate)
                    D2D_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                    Reward[Episode_loop, Run_loop] = D2D_Reward  # total reward

                    Per_V2V_Rate[Episode_loop, Run_loop, :] = np.sum(V2V_Rate, axis=1)
                    Per_V2I_Rate[Episode_loop, Run_loop, :] = V2I_Rate
                    Per_V2B_Interference[Episode_loop, Run_loop, :] = Interference

                # Tracking the simulation
                if (Episode_loop + 1) % Run_Episode_Interval == 0 and (Run_loop + 1) % Run_Step_Interval == 0:
                    Current_DateTime = datetime.datetime.now()
                    print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                    print('Current Running Step: ', Run_loop + 1, ' / Total Running Steps:', self.num_Test_Step)

                # Calculate the Expected Return
                Expect_Return[Episode_loop] += Reward[Episode_loop, Run_loop]

                # Get Next State
                States = self.generate_d2d_initial_states()

        if RA_Flag and Opt_Flag:
            print('Finish ROBUST Test of the Real-FB C-Decision scheme with Optimal Scheme!')
            return Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate, \
                   Per_V2B_Interference, \
                   RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate, \
                   RA_Per_V2B_Interference, \
                   Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate, \
                   Opt_Per_V2B_Interference
        else:
            if RA_Flag:
                print('Finish ROBUST Test of the Real-FB C-Decision scheme withOUT Optimal Scheme!')
                return Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate, \
                       Per_V2B_Interference,\
                       RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate, \
                       RA_Per_V2B_Interference
            else:
                print('Finish ROBUST Test of the Real-FB C-Decision scheme ')
                return Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate, \
                       Per_V2B_Interference

    def evaluate_training_diff_trials(self, num_episodes, num_test_step, opt_flag,
                                      fixed_epsilon, num_evaluate_trials):
        # define run() to Evaluate the trained C-Decision model

        self.num_Episodes = num_episodes
        self.num_Test_Step = num_test_step

        # weight for the V2V sum rate
        v2v_weight = self.v2v_weight
        # weight for the V2I sum rate
        v2i_weight = self.v2i_weight
        # exploration rate for evaluation
        Fixed_Epsilon = fixed_epsilon
        # variables for random action
        num_neighbor = self.num_Neighbor
        CH_Set = range(0, self.num_CH)

        # evaluate the training process for several trials
        num_Evaluate_Trials = num_evaluate_trials
        # for the optimal performance
        Evaluated_Opt_Expect_Return = np.zeros(num_Evaluate_Trials)
        Evaluated_Opt_Reward = np.zeros((num_Evaluate_Trials, self.num_Test_Step))

        # add the comparing schemes:
        # Random Action scheme: RA, where each D2D chooses its own action randomly
        RA_Flag = True
        RA_Expect_Return = np.zeros((num_Evaluate_Trials, self.num_Episodes))
        RA_Reward = np.zeros((num_Evaluate_Trials, self.num_Episodes, self.num_Test_Step))

        # implement Optimal Scheme (Opt) via Brute Force Search
        Opt_Flag = opt_flag
        if Opt_Flag:
            Opt_D2D_Action_Index = np.zeros((num_Evaluate_Trials, self.num_Episodes, self.num_Test_Step))
            Opt_Expect_Return = np.zeros(num_Evaluate_Trials, self.num_Episodes)
            Opt_Reward = np.zeros((num_Evaluate_Trials, self.num_Episodes, self.num_Test_Step))
            # record the related variables
            Opt_Per_V2V_Rate = np.zeros((num_Evaluate_Trials, self.num_Episodes, self.num_Test_Step, self.num_D2D))
            Opt_Per_V2I_Rate = np.zeros((num_Evaluate_Trials, self.num_Episodes, self.num_Test_Step, self.num_CH))
            Opt_Per_V2B_Interference = np.zeros((num_Evaluate_Trials, self.num_Episodes, self.num_Test_Step, self.num_CH))

        # tracking the simulation
        Run_Episode_Interval = 40
        Run_Step_Interval = 50

        # parameters for evaluating
        Train_Evaluation_Flag = True
        if Train_Evaluation_Flag:
            Num_D2D_feedback = self.num_Feedback
            GAMMA = self.gamma
            V2I_Weight = self.v2i_weight
            num_train_steps = 20
            BATCH_SIZE = self.batch_size
            Evaluation_Episode_Interval = 5
            Num_Evaluation_Episodes = self.num_Episodes
            # record the return per episode
            Evaluation_Return_per_Episode = np.zeros((num_Evaluate_Trials, Num_Evaluation_Episodes))
            # record the reward per step
            Evaluation_Reward_per_Episode = np.zeros((num_Evaluate_Trials, Num_Evaluation_Episodes, self.num_Test_Step))

        for Trial_loop in range(num_Evaluate_Trials):

            # tracking different evaluate trails
            Current_DateTime = datetime.datetime.now()
            print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
            print('Current Evaluate Trials: ', Trial_loop + 1, ' / Total Evaluate Trials:', num_Evaluate_Trials)

            for Episode_loop in range(self.num_Episodes):

                # load the corresponding trained model
                if Train_Evaluation_Flag:

                    num_episodes = (Episode_loop + 1) * Evaluation_Episode_Interval
                    #  load the trained results according to their corresponding simulation parameter settings
                    curr_sim_set = 'Train-Result' + '-Feedback-' + str(Num_D2D_feedback) \
                                   + '-BatchSize-' + str(BATCH_SIZE) \
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
                    self.brain.model.load_weights(model_para)
                    # load Target Network weights
                    self.brain.target_model.load_weights(target_model_para)

                    # for debugging
                    if (Episode_loop + 1) % Run_Episode_Interval == 0:
                        print('Load the trained model successfully at trained episode = ', num_episodes)

                    # for each evaluation, we use the same seed
                    evaluate_seed_sequence = Trial_loop + 1
                    random.seed(evaluate_seed_sequence)
                    np.random.seed(evaluate_seed_sequence)
                    tf.set_random_seed(evaluate_seed_sequence)  # random seed for tensor flow

                # start a new game for each episode
                self.env.new_random_game(self.num_D2D)

                # Generate the states
                Initial_State = self.generate_d2d_initial_states()
                States = Initial_State

                # tracking the simulation
                if (Episode_loop + 1) % Run_Episode_Interval == 0:
                    Current_DateTime = datetime.datetime.now()
                    print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                    print('    Current Running Episode: ', Episode_loop + 1, ' / Total Running Episodes:', self.num_Episodes)

                for Run_loop in range(self.num_Test_Step):

                    # to get the ground truth, compute the optimal return only once
                    if Episode_loop == 0:
                        # implement Optimal scheme via Brute Force
                        # initialize variables
                        Num_Possisble_Action = self.num_CH ** self.num_D2D
                        Curr_Feasible_Reward = np.zeros(Num_Possisble_Action)
                        BF_V2V_Rate = np.zeros((Num_Possisble_Action, self.num_D2D))
                        BF_V2I_Rate = np.zeros((Num_Possisble_Action, self.num_CH))
                        BF_Interference = np.zeros((Num_Possisble_Action, self.num_CH))
                        for BF_loop in range(Num_Possisble_Action):
                            # change the RL_Actions [0,255] to D2D actions [a, a, a, a] where a in {0,1,2,3}
                            D2D_Action = np.zeros(self.num_D2D, int)
                            n = BF_loop
                            a0 = n // (4 ** 3)
                            a1 = (n % (4 ** 3)) // (4 ** 2)
                            a2 = (n % (4 ** 2)) // (4 ** 1)
                            a3 = n % (4 ** 1)
                            D2D_Action[0] = a0
                            D2D_Action[1] = a1
                            D2D_Action[2] = a2
                            D2D_Action[3] = a3
                            Curr_D2D_Action = np.reshape(D2D_Action, [self.num_D2D, 1])
                            # Take action and Get Reward
                            # Just compute the reward Not update the states
                            [V2V_Rate, V2I_Rate, Interference] = self.dump_act(Curr_D2D_Action)

                            # adopt weighted sum rate as the reward
                            Sum_V2I_Rate = np.sum(V2I_Rate)
                            Sum_V2V_Rate = np.sum(V2V_Rate)
                            Curr_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                            # record all related information
                            Curr_Feasible_Reward[BF_loop] = Curr_Reward
                            BF_V2V_Rate[BF_loop, :] = np.sum(V2V_Rate, axis=1)
                            BF_V2I_Rate[BF_loop, :] = V2I_Rate
                            BF_Interference[BF_loop, :] = Interference

                        Curr_Opt_Reward = np.max(Curr_Feasible_Reward)
                        # here Curr_Opt_Reward > 0 means that there exists a feasible solution
                        if Curr_Opt_Reward > 0:
                            # only record the related parameters when there exists at least one feasible solution
                            Evaluated_Opt_Reward[Trial_loop, Run_loop] = Curr_Opt_Reward
                            # Opt : Calculate the Expected Return
                            Evaluated_Opt_Expect_Return[Trial_loop] += Evaluated_Opt_Reward[Trial_loop, Run_loop]

                    if RA_Flag:
                        RA_D2D_Action = self.select_action_random(States)
                        # Just compute the reward Not update the states
                        [RA_V2V_Rate, V2I_Rate, Interference] = self.dump_act(RA_D2D_Action)

                        Sum_V2I_Rate = np.sum(V2I_Rate)
                        Sum_V2V_Rate = np.sum(RA_V2V_Rate)
                        RA_D2D_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate
                        RA_Reward[Trial_loop, Episode_loop, Run_loop] = RA_D2D_Reward  # total reward
                        # RA: Calculate the Expected Return
                        RA_Expect_Return[Trial_loop, Episode_loop] += RA_Reward[Trial_loop, Episode_loop, Run_loop]

                    if Opt_Flag:
                        # implement Optimal scheme via Brute Force Search

                        # initialize variables
                        Num_Possisble_Action = self.num_CH ** self.num_D2D
                        Curr_Feasible_Reward = np.zeros(Num_Possisble_Action)
                        BF_V2V_Rate = np.zeros((Num_Possisble_Action, self.num_D2D))
                        BF_V2I_Rate = np.zeros((Num_Possisble_Action, self.num_CH))
                        BF_Interference = np.zeros((Num_Possisble_Action, self.num_CH))
                        for BF_loop in range(self.brain.num_Output):
                            # change the RL_Actions [0,255] to D2D actions [a, a, a, a] where a in {0,1,2,3}
                            D2D_Action = np.zeros(self.num_D2D, int)
                            n = BF_loop
                            a0 = n // (4 ** 3)
                            a1 = (n % (4 ** 3)) // (4 ** 2)
                            a2 = (n % (4 ** 2)) // (4 ** 1)
                            a3 = n % (4 ** 1)
                            D2D_Action[0] = a0
                            D2D_Action[1] = a1
                            D2D_Action[2] = a2
                            D2D_Action[3] = a3
                            Curr_D2D_Action = np.reshape(D2D_Action, [self.num_D2D, 1])
                            # Take action and Get Reward
                            # Just compute the reward Not update the states
                            [V2V_Rate, V2I_Rate, Interference] = self.dump_act(Curr_D2D_Action)

                            # adopt weighted sum rate as the reward
                            Sum_V2I_Rate = np.sum(V2I_Rate)
                            Sum_V2V_Rate = np.sum(V2V_Rate)
                            Curr_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                            # record all related information
                            Curr_Feasible_Reward[BF_loop] = Curr_Reward
                            BF_V2V_Rate[BF_loop, :] = np.sum(V2V_Rate, axis=1)
                            BF_V2I_Rate[BF_loop, :] = V2I_Rate
                            BF_Interference[BF_loop, :] = Interference

                        Curr_Opt_Reward = np.max(Curr_Feasible_Reward)
                        # here Curr_Opt_Reward > 0 means that there exists a feasible solution
                        if Curr_Opt_Reward > 0:
                            # only record the related parameters when there exists at least one feasible solution
                            Curr_Opt_Act_Index = np.argmax(Curr_Feasible_Reward)
                            Opt_Reward[Trial_loop, Episode_loop, Run_loop] = Curr_Opt_Reward
                            # Opt : Calculate the Expected Return
                            Opt_Expect_Return[Trial_loop, Episode_loop] += Opt_Reward[Trial_loop, Episode_loop, Run_loop]
                            Opt_D2D_Action_Index[Episode_loop, Run_loop] = Curr_Opt_Act_Index
                            Curr_Opt_V2V_Rate = BF_V2V_Rate[Curr_Opt_Act_Index, :]
                            Curr_Opt_V2I_Rate = BF_V2I_Rate[Curr_Opt_Act_Index, :]
                            Curr_Opt_Interference = BF_Interference[Curr_Opt_Act_Index, :]
                            Opt_Per_V2V_Rate[Trial_loop, Episode_loop, Run_loop, :] = Curr_Opt_V2V_Rate
                            Opt_Per_V2I_Rate[Trial_loop, Episode_loop, Run_loop, :] = Curr_Opt_V2I_Rate
                            Opt_Per_V2B_Interference[Trial_loop, Episode_loop, Run_loop, :] = Curr_Opt_Interference

                    # here adopt Fixed Epsilon-Greedy Strategy to evaluate the training process
                    if np.random.random() < Fixed_Epsilon:
                        # generate action for each D2D randomly
                        D2D_Action = np.zeros((self.num_D2D, 1), int)
                        for D2D_loop in range(self.num_D2D):
                            D2D_Action[D2D_loop] = np.random.choice(CH_Set, num_neighbor)
                        Actions = D2D_Action

                    else:
                        # Generate Q(Stats,a) via putting the States into trained model
                        Q_Pred = self.brain.predict_one_step(States)

                        # Find the RL-actions to maximize Q value
                        Action_Max = np.where(Q_Pred == np.max(Q_Pred))
                        RL_Action = Action_Max[1][0]

                        # Get the D2D Actions
                        # change the RL_Actions [0,255] to D2D actions [a, a, a, a] where a in {0,1,2,3}
                        D2D_Action = np.zeros(self.num_D2D, int)
                        # Change  a_RL (Decimal)  to a (Quaternary)
                        n = RL_Action
                        a0 = n // (4 ** 3)
                        a1 = (n % (4 ** 3)) // (4 ** 2)
                        a2 = (n % (4 ** 2)) // (4 ** 1)
                        a3 = n % (4 ** 1)
                        D2D_Action[0] = a0
                        D2D_Action[1] = a1
                        D2D_Action[2] = a2
                        D2D_Action[3] = a3
                        Actions = np.reshape(D2D_Action, [self.num_D2D, 1])

                    # Take action and Get Reward
                    [V2V_Rate, V2I_Rate, Interference] = self.act(Actions)
                    # adopt weighted sum rate as the reward
                    Sum_V2I_Rate = np.sum(V2I_Rate)
                    Sum_V2V_Rate = np.sum(V2V_Rate)
                    D2D_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                    Evaluation_Reward_per_Episode[Trial_loop, Episode_loop, Run_loop] = D2D_Reward

                    # Tracking the simulation
                    if (Episode_loop + 1) % Run_Episode_Interval == 0 and (Run_loop + 1) % Run_Step_Interval == 0:
                        Current_DateTime = datetime.datetime.now()
                        print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                        print('              Current Running Step: ', Run_loop + 1,
                              ' / Total Running Steps:', self.num_Test_Step)

                    # Calculate the Expected Return
                    Evaluation_Return_per_Episode[Trial_loop, Episode_loop] += \
                        Evaluation_Reward_per_Episode[Trial_loop, Episode_loop, Run_loop]

                    # Get Next State
                    States = self.generate_d2d_initial_states()

                # print the optimal scheme performance for debugging
                if (Trial_loop + 1) == num_evaluate_trials and Episode_loop == 0:
                    print('Current Evaluation Trial:', Trial_loop)
                    print('Current Running Episode: ', Episode_loop)
                    print('The optimal return = ', Evaluated_Opt_Expect_Return)

        if RA_Flag and Opt_Flag:
            print('Finish Evaluation the Real-FB C-Decision scheme with Optimal Scheme!')
            return Evaluation_Return_per_Episode, Evaluation_Reward_per_Episode, \
                   RA_Expect_Return, RA_Reward, \
                   Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate, \
                   Opt_Per_V2B_Interference
        else:
            if RA_Flag:
                print('Finish Evaluation the Real-FB C-Decision scheme without Optimal Scheme!')
                return Evaluated_Opt_Expect_Return, \
                       Evaluation_Return_per_Episode, Evaluation_Reward_per_Episode, \
                       RA_Expect_Return, RA_Reward
            else:
                print('Finish Evaluation the Real-FB C-Decision scheme only!')
                return Evaluation_Return_per_Episode, Evaluation_Reward_per_Episode
