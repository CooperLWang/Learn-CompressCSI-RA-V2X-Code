# implement the DNN in Each D2D and the DQN at the BS for the D-Decision scheme with the Binary Feedback
# here we use D2D and V2V interchangeably

import numpy as np
import pickle
import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
import tensorflow as tf
import datetime
import os


def huber_loss(y_true, y_pred):
    # define huber loss
    return tf.losses.huber_loss(y_true, y_pred)


def sign_binarize_input(x):
    # use sign function to binarize the input Tensor x
    x_bin = tf.keras.backend.sign(x)
    return [x_bin]


def grad_control_bin(x):
    # stop gradient of binarize_input(x) in the backward pass
    x_bin_grad_control = x + tf.stop_gradient(sign_binarize_input(x) - x)

    x_bin_grad = tf.reshape(x_bin_grad_control, shape=tf.shape(x))

    return x_bin_grad


class BS:
    """
    Define the BS DNN class
    """
    def __init__(self, num_d2d, input_d2d_info, num_d2d_feedback, num_d2d_neighbor, num_ch, num_bs_output):
        self.num_D2D = num_d2d
        self.num_Neighbor = num_d2d_neighbor
        self.num_CH = num_ch
        self.num_Feedback = num_d2d_feedback
        self.num_Output = num_bs_output
        self.num_Input = self.num_D2D*self.num_Feedback
        self.input_D2D_Info = input_d2d_info
        self.num_One_D2D_Input = ((input_d2d_info - 1)*self.num_CH + 1)*self.num_Neighbor
        self.num_D2D_Input = num_d2d*self.num_One_D2D_Input
        self.model = self._create_model()
        self.target_model = self._create_model()  # target network

    def _create_model(self):
        # construct D2D DNN and BS DNN together

        Num_D2D_Input = self.num_One_D2D_Input
        Num_D2D_Output = self.num_Feedback

        # implement fixed real feedback while varying the number of binary output
        Num_D2D_Real_Feedback = 3

        # implement DNN for 1st D2D
        D1_Input = Input(shape=(Num_D2D_Input,), name='D1_Input')
        D1 = Dense(16, activation='relu')(D1_Input)
        D1 = Dense(32, activation='relu')(D1)
        D1 = Dense(16, activation='relu')(D1)
        D1_Output = Dense(Num_D2D_Real_Feedback, activation='linear', name='D1_Output')(D1)
        Pre_Binary_D1_Output = Dense(Num_D2D_Output, activation='tanh', name='Pre_Binary_D1_Output')(D1_Output)
        Binary_D1_Output = Lambda(grad_control_bin, name='Binary_D1_Output')(Pre_Binary_D1_Output)

        # implement DNN for 2nd D2D
        D2_Input = Input(shape=(Num_D2D_Input,), name='D2_Input')
        D2 = Dense(16, activation='relu')(D2_Input)
        D2 = Dense(32, activation='relu')(D2)
        D2 = Dense(16, activation='relu')(D2)
        D2_Output = Dense(Num_D2D_Real_Feedback, activation='linear', name='D2_Output')(D2)
        Pre_Binary_D2_Output = Dense(Num_D2D_Output, activation='tanh', name='Pre_Binary_D2_Output')(D2_Output)
        Binary_D2_Output = Lambda(grad_control_bin, name='Binary_D2_Output')(Pre_Binary_D2_Output)

        # implement DNN for 3rd D2D
        D3_Input = Input(shape=(Num_D2D_Input,), name='D3_Input')
        D3 = Dense(16, activation='relu')(D3_Input)
        D3 = Dense(32, activation='relu')(D3)
        D3 = Dense(16, activation='relu')(D3)
        D3_Output = Dense(Num_D2D_Real_Feedback, activation='linear', name='D3_Output')(D3)
        Pre_Binary_D3_Output = Dense(Num_D2D_Output, activation='tanh', name='Pre_Binary_D3_Output')(D3_Output)
        Binary_D3_Output = Lambda(grad_control_bin, name='Binary_D3_Output')(Pre_Binary_D3_Output)

        # implement DNN for 4th D2D
        D4_Input = Input(shape=(Num_D2D_Input,), name='D4_Input')
        D4 = Dense(16, activation='relu')(D4_Input)
        D4 = Dense(32, activation='relu')(D4)
        D4 = Dense(16, activation='relu')(D4)
        D4_Output = Dense(Num_D2D_Real_Feedback, activation='linear', name='D4_Output')(D4)
        Pre_Binary_D4_Output = Dense(Num_D2D_Output, activation='tanh', name='Pre_Binary_D4_Output')(D4_Output)
        Binary_D4_Output = Lambda(grad_control_bin, name='Binary_D4_Output')(Pre_Binary_D4_Output)

        # BS Aggregation DNN
        BS_Input = keras.layers.concatenate([Binary_D1_Output, Binary_D2_Output,
                                             Binary_D3_Output, Binary_D4_Output], name='BS_Input')
        BS_Dense1 = Dense(500, activation='relu', name='BS_Dense1')(BS_Input)
        BS_Dense2 = Dense(400, activation='relu', name='BS_Dense2')(BS_Dense1)
        BS = Dense(300, activation='relu', name='BS')(BS_Dense2)
        Num_BS_Output = self.num_Output

        # implement fixed real feedback for BS while varying the number of binary output -- May 31 2019
        Num_BS_Real_Ouput = 16
        BS_Output = Dense(Num_BS_Real_Ouput, activation='linear', name='BS_Output')(BS)
        Pre_Binary_BS_Output = Dense(Num_BS_Output, activation='tanh', name='Pre_Binary_BS_Output')(BS_Output)
        Binary_BS_Output = Lambda(grad_control_bin, name='Binary_BS_Output')(Pre_Binary_BS_Output)

        # implement Decision DNN for 1st D2D
        Num_D2D_Decide_Output = self.num_CH
        D1_Decide_Input = keras.layers.concatenate([D1_Input, Binary_BS_Output], name='D1_Decide_Input')
        D1_Decide = Dense(80, activation='relu')(D1_Decide_Input)
        D1_Decide = Dense(40, activation='relu')(D1_Decide)
        D1_Decide = Dense(20, activation='relu')(D1_Decide)
        D1_Decide_Output = Dense(Num_D2D_Decide_Output, activation='linear', name='D1_Decide_Output')(D1_Decide)

        # implement Decision DNN for 2nd D2D
        D2_Decide_Input = keras.layers.concatenate([D2_Input, Binary_BS_Output], name='D2_Decide_Input')
        D2_Decide = Dense(80, activation='relu')(D2_Decide_Input)
        D2_Decide = Dense(40, activation='relu')(D2_Decide)
        D2_Decide = Dense(20, activation='relu')(D2_Decide)
        D2_Decide_Output = Dense(Num_D2D_Decide_Output, activation='linear', name='D2_Decide_Output')(D2_Decide)

        # implement Decision DNN for 3rd D2D
        D3_Decide_Input = keras.layers.concatenate([D3_Input, Binary_BS_Output], name='D3_Decide_Input')
        D3_Decide = Dense(80, activation='relu')(D3_Decide_Input)
        D3_Decide = Dense(40, activation='relu')(D3_Decide)
        D3_Decide = Dense(20, activation='relu')(D3_Decide)
        D3_Decide_Output = Dense(Num_D2D_Decide_Output, activation='linear', name='D3_Decide_Output')(D3_Decide)

        # implement Decision DNN for 4th D2D
        D4_Decide_Input = keras.layers.concatenate([D4_Input, Binary_BS_Output], name='D4_Decide_Input')
        D4_Decide = Dense(80, activation='relu')(D4_Decide_Input)
        D4_Decide = Dense(40, activation='relu')(D4_Decide)
        D4_Decide = Dense(20, activation='relu')(D4_Decide)
        D4_Decide_Output = Dense(Num_D2D_Decide_Output, activation='linear', name='D4_Decide_Output')(D4_Decide)

        # Define the model
        model = Model(inputs=[D1_Input, D2_Input, D3_Input, D4_Input],
                      outputs=[D1_Decide_Output, D2_Decide_Output, D3_Decide_Output, D4_Decide_Output])

        # the default value of learning rate lr=0.001, change it if necessary
        rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        # Compile the model
        model.compile(optimizer=rms, loss=huber_loss)

        return model

    def train_dnn(self, data_train, labels, batch_size):
        # training method for DNNs
        epochs = 1
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
        # one step predict
        return self.predict(data_test, target=target)

    def update_target_model(self):
        # use current model weights to update target network
        self.target_model.set_weights(self.model.get_weights())

# -------------------- MEMORY --------------------------
# define the memory class to replay


class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    # add sample
    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        # choose the n samples from the memory
        if len(self.samples) >= n:
            Samples_Indices = np.random.choice(len(self.samples), n, replace=False)
            Batch_Samples = np.array(self.samples)[Samples_Indices]
            return Batch_Samples
        else:
            # repeated sample the current samples until we get n samples
            Batch_Samples = []
            while len(Batch_Samples) < n:
                index = np.random.randint(0, len(self.samples))
                Batch_Samples.append(self.samples[index])
            return Batch_Samples


# ------------------------AGENT--------------------
MEMORY_CAPACITY = 1000000
UPDATE_TARGET_FREQUENCY = 500
MAX_EPSILON = 1
MIN_EPSILON = 0.01


class Agent:
    """
    Define the BS Agent class
    """
    def __init__(self, num_d2d, num_ch, num_neighbor, num_d2d_feedback, environment, curr_rl_config):
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
        self.num_BS_Output = curr_rl_config.Num_BS_Output   # number of compressed global information outputted by BS
        self.brain = BS(self.num_D2D, self.input_D2D_Info, self.num_Feedback,
                        self.num_Neighbor, self.num_CH, self.num_BS_Output)
        self.num_States = self.brain.num_D2D_Input
        self.num_Actions = self.num_D2D*self.num_Neighbor
        self.action_all_with_power = np.zeros([self.num_D2D, self.num_Neighbor, self.num_CH], dtype='int32')
        self.action_all_with_power_training = np.zeros([self.num_D2D, self.num_Neighbor, self.num_CH], dtype='int32')
        self.batch_size = curr_rl_config.Batch_Size
        self.gamma = curr_rl_config.Gamma
        self.v2v_weight = curr_rl_config.v2v_weight
        self.v2i_weight = curr_rl_config.v2i_weight

    def select_action_while_training(self, state):
        # according to current state, choose the proper action
        num_D2D = self.num_D2D
        num_neighbor = self.num_Neighbor
        Action_Matrix = 100*np.ones((num_D2D, num_neighbor))
        CH_Set = range(0, self.num_CH)

        # anneal Epsilon linearly from MAX_EPSILON to MIN_EPSILON
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
            # choose the action index which maximizes the Q Function of each D2D
            Q_Pred = self.brain.predict_one_step(state, target=False)

            D2D_Action = np.zeros((self.num_D2D, 1), int)
            # get the action for each D2D from each D2D's DQN
            for D_loop in range(self.num_D2D):
                # use the current Q function to predict the max action
                Action_Index = np.where(Q_Pred[D_loop][0] == np.max(Q_Pred[D_loop][0]))
                if len(Action_Index) == 1:
                    D2D_Action[D_loop] = Action_Index[0][0]
                else:
                    D2D_Action[D_loop] = Action_Index[0]
                    print('While Training: Current Q Predict is', Q_Pred[D_loop][0], 'at the -', D_loop, '-D2D')
                    print('                Current Action is ', Action_Index)
            Action_Matrix = D2D_Action
        return Action_Matrix.astype(np.int)

    def select_action_random(self, state):
        # choose the action Randomly
        num_D2D = self.num_D2D
        num_neighbor = self.num_Neighbor
        Action_Matrix = 100*np.ones((num_D2D, num_neighbor))
        CH_Set = range(0, self.num_CH)
        # generate action for each D2D randomly
        for D2D_loop in range(num_D2D):
            Action_Matrix[D2D_loop, :] = np.random.choice(CH_Set, num_neighbor)
        return Action_Matrix.astype(np.int)

    def act(self, actions):
        # agent executes the action
        # update current time step
        self.num_step += 1
        # take actions and get reward
        [V2V_Rate, V2I_Rate, Interference] = self.env.compute_reward_with_channel_selection(actions)
        # update the state
        self.env.renew_positions()
        self.env.renew_channels_fastfading()
        self.env.Compute_Interference(actions)
        return V2V_Rate, V2I_Rate, Interference

    def dump_act(self, actions):
        # take actions but do not update the state
        # use for the comparing schemes, such as, random action scheme, ...
        # take actions and get reward
        [V2V_Rate, V2I_Rate, Interference] = self.env.compute_reward_with_channel_selection(actions)
        return V2V_Rate, V2I_Rate, Interference

    def train_observe(self, sample):  #
        # Collect Data in (s, a, r, s_) format for training
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
        self.random_action = False
        Num_States = self.num_States
        Num_Actions = self.num_Actions
        Num_One_D2D_Input = self.brain.num_One_D2D_Input
        CH_gain_Index = self.num_Neighbor * self.num_CH
        CH_Interf_Index = self.num_Neighbor * self.num_CH

        # record the reward per transitions
        Reward_Per_Transition = np.zeros(num_transitions)
        # weight for the V2V sum rate
        v2v_weight = self.v2v_weight
        # weight for the V2I sum rate
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

            # Get all D2D channel and interference state for training
            for D2D_loop in range(self.num_D2D):
                for Neighbor_loop in range(self.num_Neighbor):
                    # Input: index[0] = target vehicle index, index[1] = neighbor index
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
                D2D_State[D2D_loop, 2*CH_gain_Index:3 * CH_gain_Index] = Current_V2I_gain
                D2D_State[D2D_loop, 3*CH_gain_Index: Num_One_D2D_Input] = Train_D2D_Power_State[D2D_loop, :]

            States = np.reshape(D2D_State, [1, Num_States])

            # two different action selection strategies: Random Selection and Epsilon-Greedy Strategy
            if self.random_action:
                # Choose the actions randomly
                Train_D2D_Action_Matrix = self.select_action_random(States)
            else:
                D1_State = np.reshape(D2D_State[0, :], [1, Num_One_D2D_Input])
                D2_State = np.reshape(D2D_State[1, :], [1, Num_One_D2D_Input])
                D3_State = np.reshape(D2D_State[2, :], [1, Num_One_D2D_Input])
                D4_State = np.reshape(D2D_State[3, :], [1, Num_One_D2D_Input])

                States_train = {'D1_Input': D1_State, 'D2_Input': D2_State,
                                'D3_Input': D3_State, 'D4_Input': D4_State}

                # choose action via Epsilon-Greedy strategy
                Train_D2D_Action_Matrix = self.select_action_while_training(States_train)

            Actions = np.reshape(Train_D2D_Action_Matrix, [1, Num_Actions])

            # Take action and Get Reward
            V2V_Rate, V2I_Rate, Interference = self.act(Train_D2D_Action_Matrix)
            Train_D2D_Reward = np.sum(V2V_Rate, axis=1)  # sum by row
            Train_BS_Reward = np.sum(Train_D2D_Reward)  # total reward
            Sum_V2I_Rate = np.sum(V2I_Rate)
            Norm_BS_Reward = Train_BS_Reward / V2V_Rate_max
            Reward = v2v_weight * Norm_BS_Reward + + v2i_weight*Sum_V2I_Rate

            # record the current reward
            Reward_Per_Transition[self.train_step] = Reward

            # Get NEXT state: all D2D channel and interference state for training
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

            D2D_Next_State = np.zeros((self.num_D2D, Num_One_D2D_Input))
            for D2D_loop in range(self.num_D2D):
                Current_CH_gain = np.reshape(Next_Train_D2D_CH_State[D2D_loop, :, :], [1, CH_gain_Index])
                D2D_Next_State[D2D_loop, 0:CH_gain_Index] = Current_CH_gain
                Current_Interf_gain = np.reshape(Next_Train_D2D_Interf_State[D2D_loop, :, :], [1, CH_Interf_Index])
                D2D_Next_State[D2D_loop, CH_gain_Index:2 * CH_gain_Index] = Current_Interf_gain
                # add V2I channel gain as D2D's input  ---June 5 2019
                Current_V2I_gain = Next_Train_D2D_V2I_CH_State[D2D_loop, :]
                D2D_Next_State[D2D_loop, 2*CH_gain_Index:3 * CH_gain_Index] = Current_V2I_gain
                D2D_Next_State[D2D_loop, 3 * CH_gain_Index: Num_One_D2D_Input] = Train_D2D_Power_State[D2D_loop, :]

            States_ = np.reshape(D2D_Next_State, [1, Num_States])

            # sample in (s, a, r, s_) format for whole D-Decision model
            sample = [States, Actions, Reward, States_]

            # add the sample (or transition) to the Buffer
            self.train_observe(sample)

        return Reward_Per_Transition

    def replay(self):
        # define the replay to generate training samples from Memory
        Num_RL_Actions = self.num_Actions
        Num_D2D = self.num_D2D
        Num_One_D2D_Input = self.brain.num_One_D2D_Input
        BATCH_SIZE = self.batch_size
        GAMMA = self.gamma
        # read samples from memory
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        test_states = np.zeros((batchLen, Num_D2D, Num_One_D2D_Input))
        test_states_ = np.zeros((batchLen, Num_D2D, Num_One_D2D_Input))
        for Sample_loop in range(batchLen):
            test_states[Sample_loop, :, :] = np.reshape(batch[Sample_loop][0], [Num_D2D, Num_One_D2D_Input])
            if batch[Sample_loop][3] is None:
                test_states_[Sample_loop, :, :] = np.zeros((Num_D2D, Num_One_D2D_Input))
            else:
                test_states_[Sample_loop, :, :] = np.reshape(batch[Sample_loop][3], [Num_D2D, Num_One_D2D_Input])

        Num_Batch = batchLen

        # initialize states
        D1_State = np.zeros([Num_Batch, Num_One_D2D_Input])
        D2_State = np.zeros([Num_Batch, Num_One_D2D_Input])
        D3_State = np.zeros([Num_Batch, Num_One_D2D_Input])
        D4_State = np.zeros([Num_Batch, Num_One_D2D_Input])
        # initialize next states
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

        states = {'D1_Input': D1_State, 'D2_Input': D2_State,
                  'D3_Input': D3_State, 'D4_Input': D4_State}

        states_ = {'D1_Input': D1_State_, 'D2_Input': D2_State_,
                   'D3_Input': D3_State_, 'D4_Input': D4_State_}

        p = self.brain.predict(states)  # Q-function network
        p_ = self.brain.predict(states_, target=True)  # target network

        # initialize the target value of Q-function
        y = np.zeros((Num_D2D, batchLen, Num_RL_Actions))

        # initialize these state as  0
        D1_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])
        D2_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])
        D3_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])
        D4_Data_Train = np.zeros([Num_Batch, Num_One_D2D_Input])

        for batch_Loop in range(batchLen):
            # fetch current sample(observation) from Replay Buffer
            # observation = {S, A, R, S_}
            o = batch[batch_Loop]
            s = o[0]      # get current state
            a = o[1]      # get current action
            r = o[2]      # get current reward
            s_ = o[3]     # get the NEXT state

            # here each D2D has a Q function, and train these Q functions together via using the global reward
            for D_loop in range(Num_D2D):
                # get current action index
                a_RL = a[0][D_loop]
                # get the prediction of current D2D's Q function
                t = p[D_loop][batch_Loop]
                if s_ is None:
                    t[a_RL] = r
                else:
                    # use target network to evaluate Q(s,a) value
                    # use the current D2D's target networks
                    t[a_RL] = r + GAMMA * np.amax(p_[D_loop][batch_Loop])

                y[D_loop][batch_Loop] = t

            # get the current testing states
            test_s = np.reshape(s, [Num_D2D, Num_One_D2D_Input])
            D1_Data_Train[batch_Loop] = test_s[0, :]
            D2_Data_Train[batch_Loop] = test_s[1, :]
            D3_Data_Train[batch_Loop] = test_s[2, :]
            D4_Data_Train[batch_Loop] = test_s[3, :]

        # use the current samples to train the model
        x = {'D1_Input': D1_Data_Train, 'D2_Input': D2_Data_Train,
             'D3_Input': D3_Data_Train, 'D4_Input': D4_Data_Train}

        y_train = {'D1_Decide_Output': y[0], 'D2_Decide_Output': y[1],
                   'D3_Decide_Output': y[2], 'D4_Decide_Output': y[3]}

        # train the model
        Train_Result = self.brain.train_dnn(x, y_train, BATCH_SIZE)

        # initialize Q mean and Q max of Target value
        Q_mean = np.zeros(Num_D2D)
        Q_max_mean = np.zeros(Num_D2D)
        # for original Q function value
        Orig_Q_mean = np.zeros(Num_D2D)
        Orig_Q_max_mean = np.zeros(Num_D2D)
        for D_loop in range(Num_D2D):
            # calculate the target Q value
            Q_batch = np.sum(y[D_loop], axis=1) / Num_RL_Actions
            Q_mean[D_loop] = np.sum(Q_batch) / batchLen
            Q_max_batch = np.max(y[D_loop], axis=1)
            Q_max_mean[D_loop] = np.sum(Q_max_batch) / batchLen
            # record the original Q function value
            Orig_Q_batch = np.sum(p[D_loop], axis=1) / Num_RL_Actions
            Orig_Q_mean[D_loop] = np.sum(Orig_Q_batch) / batchLen
            Orig_Q_max_batch = np.max(p[D_loop], axis=1)
            Orig_Q_max_mean[D_loop] = np.sum(Orig_Q_max_batch) / batchLen

        return Train_Result, Q_mean, Q_max_mean, Orig_Q_mean, Orig_Q_max_mean

    def train(self, num_episodes, num_train_steps):
        # to train the D-Decision model
        self.num_Episodes = num_episodes
        self.num_Train_Step = num_train_steps
        BATCH_SIZE = self.batch_size
        GAMMA = self.gamma
        Num_D2D = self.num_D2D
        # make several transitions before each training
        self.num_transition = 50

        # record the training loss
        Train_Loss = np.ones((Num_D2D, num_episodes, num_train_steps))
        # record the change of Target Q function
        Train_Q_mean = np.zeros((Num_D2D, num_episodes, num_train_steps))
        Train_Q_max_mean = np.zeros((Num_D2D, num_episodes, num_train_steps))
        # record the original Q value
        Orig_Train_Q_mean = np.zeros((Num_D2D, num_episodes, num_train_steps))
        Orig_Train_Q_max_mean = np.zeros((Num_D2D, num_episodes, num_train_steps))
        self.num_step = 0

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
        print('Number of BS output is: ', self.num_BS_Output)
        print('V2V Rate weight: ', self.v2v_weight)
        print('V2I Rate weight: ', self.v2i_weight)

        V2I_Weight = self.v2i_weight

        # tracking the simulation
        Train_Episode_Interval = 200
        Train_Step_Interval = 10
        Save_Model_Interval = 200

        # save results in their corresponding simulation parameter settings
        curr_sim_set = 'Train-Result' + '-SignFB-' + str(self.num_Feedback) + '-Batch-' + str(BATCH_SIZE) \
                       + '-Gamma-' + str(GAMMA) + '-BSOutput-' + str(self.num_BS_Output) \
                       + '-V2Iweight-' + str(V2I_Weight)
        folder = os.getcwd() + '\\' + curr_sim_set + '\\'
        if not os.path.exists(folder):
            os.makedirs(folder)
            # print('Create the new folder while training to save results : ', folder)
            print('Create the new folder while training to save results : ')
            print('   --->:', folder)

        curr_Result_Dir = folder

        # main loop
        for Episode_loop in range(self.num_Episodes):

            # start a new game for each episode   Feb. 22 2019
            self.env.new_random_game(self.num_D2D)

            # tracking the training process
            if (Episode_loop + 1) % Train_Episode_Interval == 0:
                Current_DateTime = datetime.datetime.now()
                print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                print('Current Training Episode: ', Episode_loop + 1, ' / Total Training Episodes:', self.num_Episodes)

            for Iteration_loop in range(self.num_Train_Step):

                # Tracking the simulation
                if (Episode_loop + 1) % Train_Episode_Interval == 0 and (Iteration_loop + 1) % Train_Step_Interval == 0:
                    Current_DateTime = datetime.datetime.now()
                    print(Current_DateTime.strftime('%Y/%m/%d %H:%M:%S'))
                    print('Current Training Step: ', Iteration_loop + 1, ' / Total Training Steps:', self.num_Train_Step)

                # make several transitions then begin training
                Reward_Per_Transition = self.generate_d2d_transition(self.num_transition)
                # record the reward per train step
                Reward_Per_Train_Step[Episode_loop, Iteration_loop, :] = Reward_Per_Transition

                # train the model
                [Train_Result, Q_mean, Q_max_mean, Orig_Q_mean, Orig_Q_max_mean] = self.replay()

                # record the train loss, Q mean  and Q max mean for each D2D
                for D_loop in range(Num_D2D):
                    Loss_Index = 'D' + str(D_loop + 1) + '_Decide_Output_loss'
                    Train_Loss[D_loop, Episode_loop, Iteration_loop] = Train_Result.history[Loss_Index][0]
                    # record the target Q value
                    Train_Q_mean[D_loop, Episode_loop, Iteration_loop] = Q_mean[D_loop]
                    Train_Q_max_mean[D_loop, Episode_loop, Iteration_loop] = Q_max_mean[D_loop]
                    # record the original Q Value
                    Orig_Train_Q_mean[D_loop, Episode_loop, Iteration_loop] = Orig_Q_mean[D_loop]
                    Orig_Train_Q_max_mean[D_loop, Episode_loop, Iteration_loop] = Orig_Q_max_mean[D_loop]

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
                if Curr_Train_Episode % Train_Episode_Interval == 0:
                    print('Save Q-Function Network model weights after Training at Episode :', Curr_Train_Episode)
                # save the Target Network's weights in case we need it
                target_model_name = 'Target-Network_model_weights' + '-Episode-' + str(Curr_Train_Episode) \
                                    + '-Step-' + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.h5'
                target_model_para = model_dir + target_model_name
                self.brain.target_model.save_weights(target_model_para)
                if Curr_Train_Episode % Train_Episode_Interval == 0:
                    print('Save Target Network model weights after Training at Episode :', Curr_Train_Episode)

                # save current train loss and Q values
                # recompute current train loss and Q values for each D2D
                Curr_Train_Loss_per_Episode = np.zeros((Num_D2D, num_episodes))
                Curr_Train_Q_mean_per_Episode = np.zeros((Num_D2D, num_episodes))
                Curr_Train_Q_max_mean_per_Episode = np.zeros((Num_D2D, num_episodes))
                Curr_Orig_Train_Q_mean_per_Episode = np.zeros((Num_D2D, num_episodes))
                Curr_Orig_Train_Q_max_mean_per_Episode = np.zeros((Num_D2D, num_episodes))

                for D_loop in range(Num_D2D):
                    # Train_Loss
                    Curr_Train_Loss_per_Episode[D_loop, :] = np.sum(Train_Loss[D_loop, :, :], axis=1) / num_train_steps
                    # Target Q Value
                    Curr_Train_Q_mean_per_Episode[D_loop, :] = np.sum(Train_Q_mean[D_loop, :, :],
                                                                      axis=1) / num_train_steps
                    Curr_Train_Q_max_mean_per_Episode[D_loop, :] = np.sum(Train_Q_max_mean[D_loop, :, :],
                                                                          axis=1) / num_train_steps
                    # Original Q value
                    Curr_Orig_Train_Q_mean_per_Episode[D_loop, :] = np.sum(Orig_Train_Q_mean[D_loop, :, :],
                                                                           axis=1) / num_train_steps
                    Curr_Orig_Train_Q_max_mean_per_Episode[D_loop, :] = np.sum(Orig_Train_Q_max_mean[D_loop, :, :],
                                                                               axis=1) / num_train_steps

                Data_Dir = curr_Result_Dir
                Data_Name = 'Temp-Training-Result' + '-Episode-' + str(Curr_Train_Episode) \
                            + '-Step-' + str(num_train_steps) + '-Batch-' + str(BATCH_SIZE) + '.pkl'

                Data_Para = Data_Dir + Data_Name
                # open data file
                file_to_open = open(Data_Para, 'wb')
                # write D2D_Sample to data file
                pickle.dump((Curr_Train_Episode,
                             Curr_Train_Loss_per_Episode, Train_Loss,
                             Curr_Train_Q_mean_per_Episode, Curr_Train_Q_max_mean_per_Episode,
                             Curr_Orig_Train_Q_mean_per_Episode, Curr_Orig_Train_Q_max_mean_per_Episode,
                             Reward_Per_Train_Step, Reward_Per_Episode), file_to_open)
                file_to_open.close()

        return Train_Loss, Reward_Per_Train_Step, Reward_Per_Episode, \
               Train_Q_mean, Train_Q_max_mean, Orig_Train_Q_mean, Orig_Train_Q_max_mean

    def generate_d2d_initial_states(self):
        # generate initial states for RL to Test
        Train_D2D_CH_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
        Train_D2D_Interf_State = np.zeros((self.num_D2D, self.num_Neighbor, self.num_CH))
        Train_D2D_V2I_CH_State = np.zeros((self.num_D2D, self.num_CH))
        Fixed_Power = self.env.V2V_power_dB_List[self.env.fixed_v2v_power_index]
        Train_D2D_Power_State = Fixed_Power * np.ones((self.num_D2D, self.num_Neighbor))

        # Get all D2D channel and interference states for Testing
        for D2D_loop in range(self.num_D2D):
            for Neighbor_loop in range(self.num_Neighbor):
                # Input: indx[0] = target vehicle index, indx[1] = neighbor index
                index = [D2D_loop, Neighbor_loop]
                [V2V_channel, V2V_interference, V2I_channel] = self.get_state(index)
                Train_D2D_CH_State[D2D_loop, Neighbor_loop, :] = V2V_channel
                Train_D2D_Interf_State[D2D_loop, Neighbor_loop, :] = V2V_interference
            Train_D2D_V2I_CH_State[D2D_loop, :] = V2I_channel

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
        # define run() to test the trained model
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

        # add the comparing schemes:
        # Random Action scheme: RA, where each D2D chooses its own action randomly
        RA_Flag = True
        RA_Expect_Return = np.zeros(self.num_Episodes)
        RA_Reward = np.zeros((self.num_Episodes, self.num_Test_Step))
        RA_Per_V2V_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_D2D))
        RA_Per_V2I_Rate = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))
        RA_Per_V2B_Interference = np.zeros((self.num_Episodes, self.num_Test_Step, self.num_CH))

        # implement Optimal Scheme (Opt) via Brute Force Search
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
                    # implement Random Action scheme
                    RA_D2D_Action = self.select_action_random(States)
                    # Just compute the reward Not update the states
                    [RA_V2V_Rate, V2I_Rate, Interference] = self.dump_act(RA_D2D_Action)
                    Sum_V2I_Rate = np.sum(V2I_Rate)
                    Sum_V2V_Rate = np.sum(RA_V2V_Rate)
                    RA_D2D_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate
                    RA_Reward[Episode_loop, Run_loop] = RA_D2D_Reward
                    RA_Expect_Return[Episode_loop] += RA_Reward[Episode_loop, Run_loop]
                    RA_Per_V2V_Rate[Episode_loop, Run_loop, :] = np.sum(RA_V2V_Rate, axis=1)
                    RA_Per_V2I_Rate[Episode_loop, Run_loop, :] = V2I_Rate
                    RA_Per_V2B_Interference[Episode_loop, Run_loop, :] = Interference

                if Opt_Flag:
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
                        Sum_V2I_Rate = np.sum(V2I_Rate)
                        Sum_V2V_Rate = np.sum(V2V_Rate)
                        Curr_Reward = v2v_weight * Sum_V2V_Rate + v2i_weight * Sum_V2I_Rate

                        # record all related information
                        Curr_Feasible_Reward[BF_loop] = Curr_Reward
                        BF_V2V_Rate[BF_loop, :] = np.sum(V2V_Rate, axis=1)
                        BF_V2I_Rate[BF_loop, :] = V2I_Rate
                        BF_Interference[BF_loop, :] = Interference

                    Curr_Opt_Reward = np.max(Curr_Feasible_Reward)
                    if Curr_Opt_Reward > 0:
                        # only record the related parameters when there exists at least one feasible solution
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

                # Generate Q(Stats,a) via putting the States into trained model
                Q_Pred = self.brain.predict_one_step(States)

                D2D_Action = np.zeros((self.num_D2D, 1), int)
                # get the action for each D2D from each D2D's DQN
                for D_loop in range(self.num_D2D):
                    Action_Index = np.where(Q_Pred[D_loop][0] == np.max(Q_Pred[D_loop][0]))
                    if len(Action_Index) == 1:
                        D2D_Action[D_loop] = Action_Index[0][0]
                    else:
                        # when there are two actions leading to the same reward, just choose one of them
                        D2D_Action[D_loop] = Action_Index[0]
                        print('While Testing: Current Q Predict is', Q_Pred[D_loop][0], 'at the -', D_loop, '-D2D')
                        print('                Current Action is ', Action_Index)

                # Take action and Get Reward
                [V2V_Rate, V2I_Rate, Interference] = self.act(D2D_Action)
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
            print('Finish Running Binary-FB D-Decision Test  with Optimal Scheme!')
            return Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate, \
                   Per_V2B_Interference, \
                   RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate, \
                   RA_Per_V2B_Interference, \
                   Opt_Expect_Return, Opt_Reward, Opt_Per_V2V_Rate, Opt_Per_V2I_Rate, \
                   Opt_Per_V2B_Interference
        else:
            if RA_Flag:
                print('Finish Running Binary-FB D-Decision Test without Implementing Optimal Scheme!')
                return Expect_Return, Reward, Per_V2V_Rate, Per_V2I_Rate, \
                       Per_V2B_Interference, \
                       RA_Expect_Return, RA_Reward, RA_Per_V2V_Rate, RA_Per_V2I_Rate, \
                       RA_Per_V2B_Interference
            else:
                print('Finish Running Binary-FB D-Decision Test only！')
                return Expect_Return, Reward
