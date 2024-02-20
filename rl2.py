import numpy as np
from log import *
import pydotplus
from IPython.display import Image
from sklearn import tree
from gym.envs.toy_text.taxi import *
from taxi import *
import gym
from sklearn.tree import DecisionTreeClassifier
import utils

env_taxi = gym.make("Taxi-v3").env

def get_rollout(env, policy):
    env_decode = TaxiEnv()
    obs, info = env.reset()
    row, column, passe, des = env_decode.decode(obs)
    state_list = (row, column, passe, des)
    obs = np.reshape(obs, [1, 1])
    rollout = []

        # Action
    act = policy.predict(obs)
    

        # Step
    next_obs, rew, done, info, _ = env.step(act)

        # Rollout (s, a, r)
    rollout.append((state_list, act, rew, obs))

        # Update (and remove LazyFrames)
    obs = np.array(next_obs)
       

    return rollout

def get_rollout_student(env, policy):
    env_decode = TaxiEnv()
    obs, info = env.reset()
    row, column, passe, des = env_decode.decode(obs)
    state_tuple = (row, column, passe, des)
    state_list = list(state_tuple)
    obs = np.reshape(obs, [1, 1])
    rollout = []
    print(state_list)
        # Action
    act = policy.predict(state_list)
        # Step
    next_obs, rew, done, info, _ = env.step(act[0])

        # Rollout (s, a, r)
    rollout.append((state_tuple, act, rew, obs))

        # Update (and remove LazyFrames)
    obs = np.array(next_obs)
       

    return rollout


def get_rollouts(env, policy, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy))
    return rollouts

def get_rollouts_student(env, policy, render, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout_student(env, policy))
    return rollouts

def calc_p(p):
    return np.max(p) - np.min(p)

def sum(l):
    s = 0
    for i in l:
        s += i
    return s

def calc_ps(qs):
    ps = []
    for q in qs:
        ps.append(calc_p(q))
    return ps/sum(ps)

def _sample(obss, acts, qs, max_pts, is_reweight):
    # Step 1: Compute probabilities
    #print(qs)
    #ps = np.max(np.array(qs), axis=1) - np.min(np.array(qs), axis=1)
    #print(ps)
    #ps = ps / np.sum(ps)
    print(qs)
    ps = calc_ps(qs)
    print(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=min(max_pts, np.sum(ps > 0)), replace=False)    
    o = [] 
    a = []
    qss = []
    for i in idx:
        o.append(obss[i])
        a.append(acts[i])
        qss.append(qs[i])
    # Step 3: Obtain sampled indices
    print(obss[idx[0]])
    #return obss[idx], acts[idx], qs[idx]
    return o, a, qss

class TransformerPolicy:
    def __init__(self, policy, state_transformer):
        self.policy = policy
        self.state_transformer = state_transformer

    def predict(self, obss):
        #return self.policy.predict(np.array([self.state_transformer(obs) for obs in obss]))
        #if (len(obss) == 1):
        #    obss = utils.decoder(obss)
        print(list(obs) for obs in obss)
        return self.policy.predict(obs for obs in obss)
    def decision_path(self, obss):
        return self.policy.decision_path(np.array([self.state_transformer(obs) for obs in obss]))

    def branches_retrieve(self):
        return self.policy.branches_retrieve()

def test_policy(env, policy, state_transformer, n_test_rollouts):
    #wrapped_student = TransformerPolicy(policy, state_transformer)
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = get_rollout_student(env, policy)
        cum_rew += sum((rew for _, _, rew, _ in student_trace))
    return cum_rew / n_test_rollouts

# Get explaination
def get_depth(policy, obs):
    #feature_names= list(str(range(0,512)))
    #classes = str([0,1,2,3,4,5])

    #dot_data = viz
    #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph = pydotplus.graph_from_dot_file("tmp/taxi/dt_taxi.dot")

    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = 0'
            node.set('label', '<br/>'.join(labels))
            node.set_fillcolor('white')

    decision_paths = get_explanation(policy, obs)


    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]            
            node.set_fillcolor('green')
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

            node.set('label', '<br/>'.join(labels))
    
    Image(graph.create_png())
    graph.write_dot('mock.dot')

def get_path(env, policy, state_transformer):
    wrapped_student = TransformerPolicy(policy, state_transformer)
    student_path = get_explanation(env, wrapped_student, False)
    return student_path

def get_explanation(policy, obs):
    #obs, done = np.array(env.reset()), False
    #rollout = []

    #if render:
    #    env.unwrapped.render()

        # Action
    #act = policy.predict(obs)

    decision_path = policy.decision_path(obs)
    
    return decision_path

def identify_best_policy(env, policies, state_transformer, n_test_rollouts):
    log('Initial policy count: {}'.format(len(policies)), INFO)
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1)/2)
        log('Current policy count: {}'.format(n_policies), INFO)

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = test_policy(env, policy, state_transformer, n_test_rollouts)
            new_policies.append((policy, new_rew))
            log('Reward update: {} -> {}'.format(rew, new_rew), INFO)

        policies = new_policies

    if len(policies) != 1:
        raise Exception()

    return policies[0][0]


def _get_action_sequences_helper(trace, seq_len):
    acts = [act for _, act, _ in trace]
    seqs = []
    for i in range(len(acts) - seq_len + 1):
        seqs.append(acts[i:i+seq_len])
    return seqs

def get_action_sequences(env, policy, seq_len, n_rollouts):
    # Step 1: Get action sequences
    seqs = []
    for _ in range(n_rollouts):
        trace = get_rollout(env, policy, False)
        seqs.extend(_get_action_sequences_helper(trace, seq_len))

    # Step 2: Bin action sequences
    counter = {}
    for seq in seqs:
        s = str(seq)
        if s in counter:
            counter[s] += 1
        else:
            counter[s] = 1

    # Step 3: Sort action sequences
    seqs_sorted = sorted(list(counter.items()), key=lambda pair: -pair[1])

    return seqs_sorted

def test_similar(student1, student2, env, teacher, n_test=100):
    rollouts = get_rollouts(env, teacher, False, n_test)
    obss = [obs for obs, _, _, _ in rollouts]
    acts = [act for _, act, _, _ in rollouts]
    acts1 = []
    acts2 = []
    a = []
    for obs in obss:
        acts1.append(student1.predict(obs))
        acts2.append(student2.predict(obs))
    
    for act in acts:
        a.append(act)
    a1 = []
    a2 = []
    for a in acts1:
        a1.append(a[0])
    for a in acts2:
        a2.append(a[0])
    correct1 = 0
    correct2 = 0
    for i in range(len(a1)):
        if (a1[i] == acts[i]):
            correct1 += 1
        if (a2[i] == acts[i]):
            correct2 += 1
    correct3 = 0
    for i in range(len(a1)):
        if (a1[i] == a2[i]):
            correct3 += 1
    return correct1/n_test, correct2/n_test, correct3/n_test

def initial_data(env, teacher, n_batch_rollouts):
    trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    return trace

def train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts, id):
    # Step 0: Setup
    obss, acts, qs = [], [], []
    students = []
    #wrapped_student = TransformerPolicy(student, state_transformer)
    
    # Step 1: Generate some supervised traces into the buffer
    #trace = get_rollouts(env, teacher, False, n_batch_rollouts)
    trace = id
    obss.extend((state_transformer(obs) for obs, _, _, _ in trace))
    acts.extend((act for _, act, _, _ in trace))
    qs.extend(teacher.predict_q(raw for _, _, _, raw  in trace))
    print("Step 1 done")

    # Step 2: Dagger outer loop
    for i in range(max_iters):
        log('Iteration {}/{}'.format(i, max_iters), INFO)

        # Step 2a: Train from a random subset of aggregated data
        cur_obss, cur_acts, cur_qs = _sample(obss, acts, qs, max_samples, is_reweight)
        log('Training student with {} points'.format(len(cur_obss)), INFO)
        student.train(cur_obss, cur_acts, train_frac)

        # Step 2b: Generate trace using student
        #student_trace = get_rollouts_student(env, wrapped_student, False, n_batch_rollouts)
        student_trace = get_rollouts_student(env, student, False, n_batch_rollouts)
        student_obss = [obs for obs, _, _, _ in student_trace]

        teacher_obss = [obs for _, _, _, obs in student_trace]
        print("Step 2b done")
        
        # Step 2c: Query the oracle for supervision
        #print(teacher_obss)
        teacher_acts = []
        teacher_qs=[]
        for obs in teacher_obss:
            teacher_qs.append(teacher.predict_q(obs)) # at the interface level, order matters, since teacher.predict may run updates
            teacher_acts.append(teacher.predict(obs))
        print("Step 2c done")

        # Step 2d: Add the augmented state-action pairs back to aggregate
        obss.extend((state_transformer(obs) for obs in student_obss))
        acts.extend(teacher_acts)
        qs.extend(teacher_qs)
        print("step 2d done")

        # Step 2e: Estimate the reward
        cur_rew = sum((rew for _, _, rew, _ in student_trace)) / n_batch_rollouts
        log('Student reward: {}'.format(cur_rew), INFO)

        students.append((student.clone(), cur_rew))

        print("step 2e done")

    max_student = identify_best_policy(env, students, state_transformer, n_test_rollouts)

    return max_student

""" model = train()
rollout = get_rollout(env_taxi, model)
rollouts = get_rollouts(env_taxi, model, False, 100)

tree = DecisionTreeClassifier(max_depth=5)
obs = []
act = []
q = []
for data in rollouts:
    obs.append(list(data[0]))
    act.append(data[1])


train_obs = obs[:80]
train_act = act[:80]
test_obs = obs[80:]
test_act = act[80:]

tree = tree.fit(train_obs, train_act)

transformer = TransformerPolicy(tree, utils.state_transformer)

pred = transformer.predict(obs for obs, _, _ in rollouts[80:])

sum = 0
for i in range(len(test_act)):
    if test_act[i] == pred[i]:
        sum += 1
print(sum/len(test_act)*100) """