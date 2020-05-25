# set the key parameters for the C-Decision scheme with the Real Feedback
import datetime  # show the date and current time


class RL_Config:
    """
    Define the Config class
    """
    def __init__(self):
        # key parameters for RL training process
        self.Num_Feedback = 3    # number of feedback for each D2D
        self.Num_Episodes = 40     # number of episodes for training
        self.Num_Train_Steps = 20   # number of steps in each Episode
        self.Batch_Size = 256   # size of the mini-batch for replay
        self.Gamma = 0.2     # discount factor in RL
        self.Num_Run_Episodes = 10  # number of episodes for testing
        self.Num_Test_Steps = 50   # number of step in each testing Episode
        self.Opt_Flag = False    # whether run the optimal scheme while testing the trained model
        # the v2v rate weight
        self.v2v_weight = 1
        # the v2i rate weight
        self.v2i_weight = 1
        self.feedback_interval = 1  # feedback interval  i.e. 3 refers to every 3 slot only feedback once
        self.input_noise_level = 0.01  # percentage of white noise adding to the inputs
        self.feedback_noise_level = 0.01  # percentage of white noise adding to the inputs

    def set_key_value(self, num_feedback, gamma, batch_size, v2v_weight, v2i_weight):
        # set the key values for training
        self.Num_Feedback = num_feedback    # number of feedback for each D2D
        self.Gamma = gamma     # discount factor in RL
        self.Batch_Size = batch_size
        self.v2v_weight = v2v_weight
        self.v2i_weight = v2i_weight

    def display(self):
        # track the simulation settings
        current_datetime = datetime.datetime.now()
        print(current_datetime.strftime('%Y/%m/%d %H:%M:%S'))
        print("Current Training Parameters Settings are: ")
        print('Number of feedback: ', self.Num_Feedback)
        print('Discount Factor Gamma: ', self.Gamma)
        print('Batch Size: ', self.Batch_Size)
        print('Training Episodes: ', self.Num_Episodes)
        print('Train Steps per Episode: ', self.Num_Train_Steps)

    def set_test_values(self, num_test_episodes, num_test_steps, opt_flag, v2v_weight, v2i_weight):
        # set the key values for testing
        self.Num_Run_Episodes = num_test_episodes
        self.Num_Test_Steps = num_test_steps
        self.Opt_Flag = opt_flag
        self.v2v_weight = v2v_weight
        self.v2i_weight = v2i_weight

    def set_robust_test_values(self, num_test_episodes, num_test_steps, opt_flag,
                               v2v_weight, v2i_weight,
                               feedback_interval,
                               input_noise_level, feedback_noise_level):
        # set the key values for testing
        self.Num_Run_Episodes = num_test_episodes
        self.Num_Test_Steps = num_test_steps
        self.Opt_Flag = opt_flag
        self.v2v_weight = v2v_weight
        self.v2i_weight = v2i_weight
        # robust parameter setting
        self.feedback_interval = feedback_interval
        self.input_noise_level = input_noise_level
        self.feedback_noise_level = feedback_noise_level
