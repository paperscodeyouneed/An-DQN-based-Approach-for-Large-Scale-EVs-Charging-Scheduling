# A-Noisy-Dueling-Architecture-for-Large-Scale-EV-ChargingScheduling
This is an official implementation in Pytorch


## Abstract
    The surge charging demand of large-scale electric vehicles (EVs) in the rush hour or open holidays poses challenges to the EV charging management system. Unreasonable EV charging scheduling plan is definitely inefficient and might lead to the unsatisfactory of customers and providers of charging service. Therefore, it is important to manage the large-scale charging scheduling for improving the service quality of large-scale EV charging services. With this concern, an improved Deep Q-Network (DQN) based approach is proposed to find the optimal scheduling plan at the time of peak demand for EV charging. To obtain better global optimization ability and faster convergence speed, an improved noisy-dueling architecture which can make efficient use of experience knowledge about environment to locate the global optimization / suboptimal is designed. The experiment results under a complex real city scenario showcases the advantages of our proposal in terms of well-performance on both calculation efficiency and scheduling efficiency.

## Notes

# A

## All python packages and their versions for running our codes are listed below:

1. matplotlib = 3.7.1,
2. numpy = 1.23.4,
3. pandas = 1.5.3,
4. scipy = 1.9.3,
5. pytorch = 1.7.1,
6. ipython = 8.10.0,
7. typing_extensions == 4.4.0,

## The scheduling result of our best

The best result of our proposal is shown and tested in the jupyter note book file STATE_TESTING.ipynb
(The result is stored in an 2-d array, each item is organized as [charging station number, distance, relative EV number ] orderly, before testify the result, you need to transform relative EV number into EV number by method transfer_order_to_no of class Environment , and example is offered to reproduce our result shown in the paper is in STATE_TESTING.ipynb)

# B

## Method to repreduce our code
We implement this algorithm in PyCharm 2019.1 and python version >= 3.7
You can run our code by running the  file "A DQN-based Two-stage Scheduling Method for Large-Scale EVs Charging Service\Agent\agent.py" directly in Pycharm.                       

# C

About our code

For convinience to reproduce our proposal, we collected and sampled 1000 trajectories with relative higher service quality (in terms of lower average idling time and average queueing time). You can generate datasets for training models with the trajectories by running the Agent/agent.py directly

The every .py file and its function is summarized as below:

①   Environment/environmet.py : offering an environmrnt to schedule an EV to CS, it could calculate the queueing time and the idling time of EVs;
②   Model/models.py : define the EV-selecting model and CS-selecting model;
③   TEST/test.py : it offers an example to reproduce our best scheduling plan;
④   Utils/datahandler.py : it offers an series utils to storing experience into replay buffer, calculating discounted reward, extract exprience from datsets;
⑤   Utils/data_squeeze.py : it helps to construct dataset from 1000 trajectories we sampled;
⑥   Utils/replay_buffer.py : it offers a simple implement of ReplayBuffer;
⑦   Utils/store_queueing_time_into_file.py : it offers a method to construct pre_train_value_list which store a list that consist of the average queueing time;
