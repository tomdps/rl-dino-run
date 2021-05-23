import argparse
import sys
import os
import numpy as np

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gym
from gym.core import GoalEnv
from gym.envs.registration import register


import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino

from stable_baselines import DQN, HER, ACER, ACKTR
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common.policies import CnnPolicy as CnnActorCritic
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import BaseCallback, EvalCallback
from stable_baselines.results_plotter import load_results, ts2xy

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = self.training_env.unwrapped.game.get_score()
        summary = tf.Summary(value=[tf.Summary.Value(tag='score', simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='DQN')
    parser.add_argument('--double', default=True)
    parser.add_argument('--dueling', default=True)
    parser.add_argument('--prio_exp_replay', default=False)    
    parser.add_argument('--her', type=bool, default=False)
    parser.add_argument('--goal_selection', type=str, default='future')
    parser.add_argument('--horizon', type=int, default=1000)
    parser.add_argument('--nb_iter',  default=100000)
    parser.add_argument('--frame_stack', default=True)  
    parser.add_argument('--logdir', type=str, default='./log')
    args = parser.parse_args()
    return args

#HYPERPARAMETERS
BUFFER_SIZE = 30000
BATCH_SIZE = 32
INIT_EPS = 1
FINAL_EPS = 0.00001
EXPL_FRACTION = 0.02
LR = 0.0005
TARGET_UPDATE = 2000

def run():
    args = parse_arguments()
    print(args.double)
    env = gym.make('ChromeDino-v1')
    env = make_dino(env, timer=True, frame_stack=args.frame_stack)

    eval_env = gym.make('ChromeDinoNoBrowser-v0')
    eval_env = make_dino(eval_env, timer=True, frame_stack=args.frame_stack)

    eval_callback = EvalCallback(eval_env, best_model_save_path=args.logdir + '/model/',
                             log_path=args.logdir, eval_freq=5000,
                             deterministic=True, render=False)
    
    if args.model == 'DQN':
        model = DQN(
            CnnPolicy, 
            env=env,
            double_q=args.double,
            prioritized_replay=args.prio_exp_replay,
            policy_kwargs=dict(dueling=args.dueling),
            learning_rate=LR,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            exploration_initial_eps=INIT_EPS,
            exploration_final_eps=FINAL_EPS,
            exploration_fraction=EXPL_FRACTION,
            target_network_update_freq=TARGET_UPDATE,
            tensorboard_log=args.logdir, 
            seed=420, 
            verbose=1)

    elif args.model== 'HER':
        model = HER(
            'CnnPolicy', 
            env=env,
            model_class=DQN,
            goal_selection_strategy=args.goal_selection,
            tensorboard_log=args.logdir, 
            seed=420, 
            verbose=1)

    elif args.model== 'ACER':
        model = ACER(
            CnnActorCritic, 
            env=env,
            tensorboard_log=args.logdir, 
            seed=420, 
            verbose=1)

    elif args.model== 'ACKTR':
        model = ACKTR(
            CnnActorCritic, 
            env=env,
            tensorboard_log=args.logdir, 
            seed=420, 
            verbose=1)

    else:
        print("Not implmented")
        sys.exit()

    model.learn(total_timesteps=args.nb_iter, callback=[TensorboardCallback()])

    model.save(args.logdir)
    
    env.close()
    
if __name__ == "__main__":
    run()
