# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # if state not in self.qValues:
        #   return 0.0
        # else:
        #   return self.qValues[(state, action)]
        if self.qValues[(state, action)] == None:
          return 0.0
        else: 
          return self.qValues[(state, action)]
        # return self.qValues[(state, action)]

        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(state)) == 0:
          return 0.0
        
        values = {}
        for action in self.getLegalActions(state):
          values[action] = self.getQValue(state, action)
        # return max(values)

        # return max(values)
        if(len(values) != 0):
          return values[max(values)]
        else:
          return 0.0

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if(len(legalActions) == 0):
          return None
        


        # # TRY 5 - RETURNING FINAL CHOICE FROM LIST OF ACTIONS WHERE BESTACTIONS = ACTIONS WHERE VALUE > THRESHOLD
        # values = util.Counter()
        # threshold = float('-inf')
        # for action in legalActions:
        #   QVal = self.getQValue(state, action)
        #   if QVal > threshold:
        #     threshold = QVal
        # goalQ = threshold
        # bestActions = []
        # for action in legalActions:
        #   if values[action] == goalQ:
        #     bestActions.append(action)
        # return random.choice(bestActions)


        # # TRY 4 - RETURNING FINAL CHOICE FROM LIST OF VALUES WHERE QVALUE AT ACTION == self.GETVALUE(STATE)
        goalQ = self.getValue(state)
        allActions = {}
        bestActions = []
        for action in legalActions:
          allActions[action] = self.getQValue(state, action)
          if allActions[action] == goalQ:
            bestActions.append(action)

        return random.choice(bestActions)

        # # TRY #1 - RETURNING FIRST ACTION WHERE QVALUE AT ACTION == TARGET (TARGET = GETVALUE)
        # bestAction = 0
        # targetQ = self.getValue(state)
        # for action in legalActions:
        #     if (targetQ == self.getQValue(state, action)):
        #       bestAction = action
        # return bestAction
        
        # # TRY #2 - random.choice() ON LIST OF ACTIONS WHERE QVAL AT ACTION == TARGET (TARGER = GETVALUE)
        # targetQ = self.getValue(state)
        # options = {}
        # for action in legalActions:
        #     if (targetQ == self.getQValue(state, action)):
        #       options[action] = self.getQValue(state, action)
        # return random.choice(list(options.keys()))
        
        # # TRY #3 - random.choice() ON  LIST OF ACTIONS WHERE QVAL AT ACTION == MAX VALUE OF OPTONS
        # options = {}
        # for action in legalActions:
        #     options[action] = self.getQValue(state, action)
        # bestActions = [k for k,v in options.items() if v == max(options)]
        # return random.choice(bestActions)
        
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if(len(legalActions) == 0):
          return None

        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        else:
          action = self.getPolicy(state)
          
        return action
        util.raiseNotDefined()

        

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        oldQ = self.getQValue(state, action)
        newQ = oldQ + self.alpha * (reward + self.discount * self.getValue(nextState) - oldQ)
        self.qValues[(state, action)] = newQ
        # return newQ
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
