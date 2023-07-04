# A-Noisy-Dueling-Architecture-for-Large-Scale-EV-ChargingScheduling
This is an official implementation in Pytorch


## Abstract
    Abstract — With the rapid development of electric vehicles (EVs) industry, EV charging service becomes more and more important. Especially, in the case of suddenly drop of air temperature or open holidays that large-scale EVs seeking for charging devices at a short time. In such scenario, inefficient EV charging scheduling service may lead to a bad service quality, for example, long queueing times for EVs and unreasonable idling time for charging devices. To deal with this issue, this paper propose a Deep-Q-Network (DQN) based two-stage scheduling method for the large-scale EVs charging service. Fine-grained states with two delicate neural networks are proposed to optimize the sequencing of EVs and charging station (CS) arrangement. Two efficient algorithms are presented to obtain the optimal EVs charging scheme with a large-scale EVs charging demand. Three case studies show the superiority of our proposal, in terms of a high service quality (minimized average queuing time of EVs and maximized charging performance at both EV and CS sides) and achieve greater scheduling efficiency. 

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
