# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rl2 import *
#from pong import *
#from dqn import *
from taxi import *
from dt import *
from log import *
import tensorflow as tf
import utils

def learn_dt(agent):
    # Parameters
    log_fname = '../taxi_dt.log'
    max_depth = 10
    n_batch_rollouts = 100
    max_samples = 10000
    max_iters = 5
    train_frac = 0.8
    is_reweight = True
    n_test_rollouts = 50
    save_dirname = 'tmp/taxi'
    save_fname = 'dt_taxi.pk'
    save_viz_fname = 'dt_taxi.dot'
    is_train = True
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    #env = get_pong_env()
    env = gym.make("Taxi-v3").env
    #teacher = DQNPolicy(env, model_path)
    teacher = agent
    state_transformer = utils.state_transformer
    student = DTPolicy(max_depth, state_transformer)

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, state_transformer, n_test_rollouts)
    log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

    #Test path
    #tree_viz = load_grahviz(save_dirname, save_viz_fname)
    #print(tree_viz)
    #get_depth(env, student, state_transformer, tree_viz)

    #Get branches
    all_branches = list(student.branches_retrieve())
    print(all_branches)
    return student

def bin_acts():
    # Parameters
    seq_len = 10
    n_rollouts = 10
    log_fname = 'taxi_options.log'
    model_path = 'model-atari-taxi-1/saved'
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    env = get_pong_env()
    teacher = DQNPolicy(env, model_path)

    # Action sequences
    seqs = get_action_sequences(env, teacher, seq_len, n_rollouts)

    for seq, count in seqs:
        log('{}: {}'.format(seq, count), INFO)

def print_size():
    # Parameters
    dirname = 'results/run9'
    fname = 'dt_policy.pk'

    # Load decision tree
    dt = load_dt_policy(dirname, fname)

    # Size
    print(dt.tree.tree_.node_count)

if __name__ == '__main__':
    env = gym.make("Taxi-v3").env
    agent = train()
    #student1 = learn_dt(agent)
    #student2 = learn_dt(agent)
    #sim1, sim2 = test_similar(student1, student2, env, agent)
    #print(sim1*100)
    #print(sim2*100)
    student = learn_dt(agent)
    get_depth(student, [[0,10,10,10]])

    #viz = load_grahviz("tmp/pong", "dt_policy1.dot")
    #print(viz)
