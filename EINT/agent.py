import gym
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
import textworld.gym
    
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from dqn import *

import datetime
import math
import nltk
from nltk.tokenize import word_tokenize
import os
from glob import glob
import jericho
import re
from textworld.gym.spaces import text_spaces
from textworld.gym.envs.utils import shuffled_cycle
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import CountVectorizer

from collections import OrderedDict
from collections import deque

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess(text, chars='', remove_all_special=True, expand=True, split_numbers=True):
    # fix bad newlines (replace with spaces), unify quotes
    text = text.replace('\\n', ' ').replace('‘', '\'').replace('’', '\'').replace('”', '"').replace('“', '"')

    # optionally remove all given characters
    for c in chars:
        if c in text:
            text = text.replace(c, '')

    # convert to lowercase
    text = text.lower()

    # remove all characters except alphanum, spaces and - ' "
    if remove_all_special:
        text = re.sub('[^ \-\sA-Za-z0-9"\']+', ' ', text)

    # split numbers into digits to avoid infinite vocabulary size if random numbers are present:
    if split_numbers:
        text = re.sub('[0-9]', ' \g<0> ', text)

    # expand unambiguous 'm, 't, 're, ... expressions
    if expand:
        text = text. \
            replace('\'m ', ' am '). \
            replace('\'re ', ' are '). \
            replace('won\'t', 'will not'). \
            replace('n\'t', ' not'). \
            replace('\'ll ', ' will '). \
            replace('\'ve ', ' have '). \
            replace('\'s', ' \'s')

    return text

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Source: https://stackoverflow.com/questions/34968722/softmax-function-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()




class Agent:
    def __init__(self, env):
        self.env = env
        self.tokenizer = Tokenizer(num_words=20200)
        self.experience = []

        self.prioritized_experiences_queue = deque(maxlen=64)
        self.unique_prioritized_experiences_queue = deque(maxlen=64)

        self.state_action_history = {}

        self.model = None

        self.model_state = None
        self.model_action = None
        self.model_dot_state_action = None

        
        self.log_folder = "logs"
        os.makedirs(self.log_folder, exist_ok=True)
        #self.cv = CountVectorizer(token_pattern="(.*)")
        #self.cv.fit(env.observation_space.vocab)
        #self.features = self.cv.get_feature_names()

        #self.commands = iter(env.game.main_quest.commands)

        #self.word2int = {}
        #self.int2word = {}

        #for i in range(0,len(self.features)):
        #    word = self.features[i]

         #   self.word2int[word] = i
        #    self.int2word[i] = word

        #self.commandsRewards = {}
        #self.actioncommands = []

    def create_model(self, embedding_dimensions, lstm_dimensions, dense_dimensions, optimizer, embeddings=None,
                     embeddings_trainable=True):
        num_words = len(self.tokenizer.word_index)
        logger.info('Creating a model based on %s unique tokens.', num_words)

        # create the shared embedding layer (with or without pre-trained weights)
        embedding_shared = None

        
        embedding_shared = Embedding(num_words + 1, embedding_dimensions, input_length=None, mask_zero=True,
                                         trainable=embeddings_trainable, name="embedding_shared")

        input_state = Input(batch_shape=(None, None), name="input_state")
        input_action = Input(batch_shape=(None, None), name="input_action")

        embedding_state = embedding_shared(input_state)
        embedding_action = embedding_shared(input_action)

        lstm_shared = LSTM(lstm_dimensions, name="lstm_shared")
        lstm_state = lstm_shared(embedding_state)
        lstm_action = lstm_shared(embedding_action)

        dense_state = Dense(dense_dimensions, activation='tanh', name="dense_state")(lstm_state)
        dense_action = Dense(dense_dimensions, activation='tanh', name="dense_action")(lstm_action)

        model_state = Model(inputs=input_state, outputs=dense_state, name="state")
        model_action = Model(inputs=input_action, outputs=dense_action, name="action")

        self.model_state = model_state
        self.model_action = model_action

        input_dot_state = Input(shape=(dense_dimensions,))
        input_dot_action = Input(shape=(dense_dimensions,))
        dot_state_action = Dot(axes=-1, normalize=True, name="dot_state_action")([input_dot_state, input_dot_action])

        model_dot_state_action = Model(inputs=[input_dot_state, input_dot_action], outputs=dot_state_action,
                                       name="dot_state_action")
        self.model_dot_state_action = model_dot_state_action

        model = Model(inputs=[model_state.input, model_action.input],
                      outputs=model_dot_state_action([model_state.output, model_action.output]),
                      name="model")
        model.compile(optimizer=optimizer, loss='mse')

        self.model = model

        print('---------------')
        print('Complete model:')
        model.summary()
        print('---------------')


    def initialize_tokens(self, vocabulary=None):
        if vocabulary:
            logger.info('Initializing tokens by loading them from %s', vocabulary)

            try:
                with open(vocabulary, "r") as f:
                    words = f.readlines()
                self.tokenizer.fit_on_texts(words)
                return    
            except IOError as e:
                logger.warning('Could not find the specified vocabulary file %s: %s; Sampling new data instead',
                vocabulary, e)

    def vectorize(self, texts, max_len=None):
        if not texts:
            return []

        sequences = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=max_len)

        # tokenizer can return empty sequences for actions such as '...', fix these:
        if not sequences.any():
            return np.asarray([[0]])
        for i in range(len(sequences)):
            if len(sequences[i]) < 1:
                sequences[i] = [0]

        return sequences


    def q_precomputed_state(self, state, actions, softmax_selection=False, penalize_history=False):
        state_dense = self.model_state.predict([state.reshape((1, len(state)))])[0]

        q_max = -np.math.inf
        best_action = 0

        q_values = np.zeros(len(actions))

        logger.debug('q for state %s', state)
        for i in range(len(actions)):

            action = actions[i]
            action_dense = self.model_action.predict([action.reshape((1, len(action)))])[0]

            q = self.model_dot_state_action.predict(
                [state_dense.reshape((1, len(state_dense))), action_dense.reshape((1, len(action_dense)))])[0][0]

            if penalize_history:
                # apply intrinsic motivation (penalize already visited (state, action) tuples)
                history = self.get_history(state, action)
                if history:
                    # q is a cosine similarity (dot product of normalized vectors), ergo q is in [-1; 1]
                    # map it to [0; 1]
                    q = (q + 1) / 2

                    q = q ** (history + 1)

                    # map q back to [-1; 1]
                    q = (q * 2) - 1

            logger.debug('q for action %s is %s', action, q)

            q_values[i] = q

            if q > q_max:
                q_max = q
                best_action = i

        if softmax_selection:
            probabilities = softmax(q_values)
            x = random.random()
            for i in range(len(actions)):
                if x <= probabilities[i]:
                    return i, q_values[i]
                x -= probabilities[i]

        return best_action, q_max

    def add_to_history(self, state, action):
        state = tuple(np.trim_zeros(state, 'f'))
        action = tuple(np.trim_zeros(action, 'f'))

        if (state, action) in self.state_action_history:
            self.state_action_history[(state, action)] += 1
        else:
            self.state_action_history[(state, action)] = 1

    def get_history(self, state, action):
        state = tuple(np.trim_zeros(state, 'f'))
        action = tuple(np.trim_zeros(action, 'f'))

        if (state, action) in self.state_action_history:
            return self.state_action_history[(state, action)]
        return 0
    
    def reset_history(self):
        """
        resets the history; called every time a game episode ends
        :return:
        """
        self.state_action_history = {}

    def experience_to_sequences(self, state, action, reward, state_next, actions_next, done):
        # vectorize the text samples into 2D tensors of word indices
        state_sequence = self.vectorize([state])[0]
        action_sequence = self.vectorize([action])[0]
        state_next_sequence = self.vectorize([state_next])[0]
        actions_next_sequences = self.vectorize(actions_next)

        return state_sequence, action_sequence, reward, state_next_sequence, actions_next_sequences, done

    def store_experience(self, state_text, action_text, reward, state_next_text, actions_next_texts, done,
                         store_text_only=False):
        """
        Speichert tuple in replay-Memory
        """
    
        if store_text_only:
            self.experience.append((state_text, action_text, reward, state_next_text, actions_next_texts, done))
            return

        experience = self.experience_to_sequences(state_text, action_text, reward, state_next_text, actions_next_texts,                                         done)
        self.experience.append(experience)

        if reward <= 0:
            return

        exp_list = (experience[0].tolist(), experience[1].tolist(), experience[2], experience[3].tolist())

        if reward > 0 and (exp_list not in self.unique_prioritized_experiences_queue):
            self.unique_prioritized_experiences_queue.append(exp_list)
            self.prioritized_experiences_queue.append(experience)



    def get_action(self, state, actions, epsilon=0):
        if epsilon == 1 or (epsilon > 0 and 1 > epsilon > random.random()):
            return random.randint(0, len(actions) - 1), None

        state = self.vectorize([state])[0]
        actions = self.vectorize(actions)

        # return an action with maximum Q value
        return self.q_precomputed_state(state, actions, softmax_selection=False, penalize_history=True)
        

    def train(self, episodes=1, batch_size=256, gamma=0.95,epsilon=1, epsilon_decay=0.99,
              prioritized_fraction=0, test_interval=1,test_steps=1,checkpoint_steps=128, log_prefix='model'):

        train_rewards_history = []
        test_rewards_history = []
        test_rewards = 0

        # batch_prioritized is the number of prioritized samples to get
        batch_prioritized = int(batch_size * prioritized_fraction)
        # batch is the number of any samples to get
        batch = batch_size - batch_prioritized

        for episode in range(episodes):
            print("Episode: ", episode+1, " Epsilon: ", epsilon)

            train_rewards = self.play(episodes=1, store_experience=True, epsilon=epsilon)
            train_rewards_history.append(train_rewards)

            if ((episode + 1) % test_interval) == 0:
                test_rewards = self.play(episodes=test_steps, store_experience=False, epsilon=0)
                test_rewards_history.append(test_rewards)
                

            if len(self.experience) < 1:
                return

            batches = np.random.choice(len(self.experience), batch)

            if len(self.prioritized_experiences_queue) > 0:
                batches_prioritized = np.random.choice(len(self.prioritized_experiences_queue), batch_prioritized)
            else:
                batches_prioritized = np.random.choice(len(self.experience), batch_prioritized)

            states = [None] * batch_size
            actions = [None] * batch_size
            targets = np.zeros((batch_size, 1))

            for b in range(batch_size):

                # non-prioritized data:
                if b < batch:
                    state, action, reward, state_next, actions_next, finished = self.experience[batches[b]]
                # prioritized data (if there are any)
                elif len(self.prioritized_experiences_queue) > 0:
                    state, action, reward, state_next, actions_next, finished = self.prioritized_experiences_queue[
                        batches_prioritized[b - batch]]
                # get non-prioritized if there are no prioritized
                else:
                    state, action, reward, state_next, actions_next, finished = self.experience[
                        batches_prioritized[b - batch]]

                _, current_q = self.q_precomputed_state(state, [action], penalize_history=False)
                alpha = 1

                target = current_q + alpha * (reward - current_q)

                if not finished:
                    # get an action with maximum Q value
                    _, q_max = self.q_precomputed_state(state_next, actions_next, penalize_history=False)
                    target += alpha * gamma * q_max

                states[b] = state
                actions[b] = action
                targets[b] = target

            # pad the states and actions so that each sample in this batch has the same size
            states = pad_sequences(states)
            actions = pad_sequences(actions)

            logger.debug('states %s', states)
            logger.debug('actions %s', actions)
            logger.debug('targets %s', targets)

            callbacks = []

            # add a tensorboard callback on the last episode
            #if i + 1 == episodes:
            #    callbacks = [self.tensorboard]

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=0,
                           callbacks=callbacks)

            epsilon *= epsilon_decay

            # every checkpoint_steps, write down train and test reward history and the model object
            if ((episode + 1) % checkpoint_steps) == 0:

                file_name = 'ep' + str(episode) + '_' + datetime.datetime.now().strftime('%m-%d-%H_%M_%S')

                with open(self.log_folder + '/' + log_prefix + '_train_' + file_name + '.txt', 'w') as file:
                    for simulator_rewards in train_rewards_history:
                        for rewards in simulator_rewards:
                            for reward in rewards:
                                file.write('{:.1f}'.format(reward) + ' ')
                            file.write(',')
                        file.write('\n')

                with open(self.log_folder + '/' + log_prefix + '_test_' + file_name + '.txt', 'w') as file:
                    for simulator_rewards in test_rewards_history:
                        for rewards in simulator_rewards:
                            for reward in rewards:
                                file.write('{:.1f}'.format(reward) + ' ')
                            file.write(',')
                        file.write('\n')

                # save the model
                self.model.save(self.log_folder + '/' + log_prefix + file_name + '.h5')
            
        return




            

    def play(self, episodes=1, store_experience=True,initialize_only=False,epsilon=1, render=False):
        for episode in range(episodes):
            experiences = []
            episode_reward = 0
            state, infos = self.env.reset()
            state = preprocess(state)
            actions = [preprocess(a) for a in infos["admissible_commands"]]
            old_reward = 0
            moves = 0
            avg_moves, avg_scores = [], []
            self.reset_history()

            while True:
                action, q_value = self.get_action(state, actions, epsilon)
                if render:
                    logger.info('State: %s', state)
                    logger.info('Best action: %s, q=%s', actions[action], q_value)
                
                if not initialize_only:
                    self.add_to_history(self.vectorize([state])[0], self.vectorize([actions[action]])[0])

                last_state = state
                last_action = actions[action]


                next_state, reward, done, infos = self.env.step(actions[action])

                next_state = preprocess(next_state)
                actions = [preprocess(a) for a in infos["admissible_commands"]]

                textreward = reward

                #scale reward
                reward /= infos["max_score"] # (reward/infos["max_score"] * 2) - 1

                if store_experience:
                    experiences.append((last_state,last_action,reward,next_state, actions, done))

                if(reward != old_reward):
                    old_reward = reward
                
                moves += 1

            
                if store_experience and (moves < self.env._max_episode_steps or initialize_only):
                    for last_state, last_action, reward, state, actions, done in experiences:
                        self.store_experience(last_state, last_action, reward,next_state, actions, done, initialize_only)


                state = next_state
                
                

                if done and infos["max_score"] == old_reward:
                    print("You won!")
                    break
                    
                if done:
                    break
                
            avg_moves.append(moves)
            avg_scores.append(textreward)
            if(epsilon==0):
                    msg = "  \tTest: steps: {:5.1f}; score: {:4.1f} / {}."
            else:
                msg = "  \tTrain: steps: {:5.1f}; score: {:4.1f} / {}."
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))
        return np.mean(avg_scores)


if __name__ == "__main__":
    path = "EINT/tw_games/tw-cooking-recipe3+take3+cook+cut+go9-D397IBQkHe9Ws6ka.ulx"
    gameFiles = path
    if os.path.isdir(path):
        gameFiles = glob(os.path.join(path,"*.ulx"))

    request_infos = textworld.EnvInfos(admissible_commands=True, max_score=True, verbs=True, command_templates=True, entities=True)

    env_id = textworld.gym.register_game(path,request_infos=request_infos, max_episode_steps=50)

    env = gym.make(env_id)

    agent = Agent(env)
    agent.initialize_tokens("EINT/vocab.txt")
    agent.create_model(128,256,10,Adam())
    agent.train(episodes=256)
    input("Play?")
    agent.play(episodes=1,epsilon=0, render=True)