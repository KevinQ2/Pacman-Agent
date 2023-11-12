# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function
from queue import Empty

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        self.currentState = state
        self.legalActions = state.getLegalPacmanActions()
        self.score = state.getScore()

        # Key values used for hashing
        self.pacman = state.getPacmanPosition()
        self.ghosts = state.getGhostPositions()
        self.food = state.getFood()

    def __eq__(self, other):
        if other != Empty:
            if self.pacman == other.pacman and self.ghosts == other.ghosts:
                return True
        
        return False

    def __hash__(self):
        # Only hash relevant values
        return int(hash(self.pacman) + 13 * hash(self.ghosts[0]) + 113 * hash(self.food))

class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 10,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        self.qValue = util.Counter()
        self.visits = util.Counter()

        # Store to record value of transitions
        self.previousState = Empty
        self.previousAction = Empty

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return endState.score - startState.score

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        return self.qValue[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        # Q-value could be < 0, so should not default maxQ as 0
        maxQ = Empty

        for action in state.legalActions:
            qValue = self.getQValue(state, action) 

            if maxQ == Empty:
                maxQ = qValue
            elif qValue > maxQ:
                maxQ = qValue

        if maxQ == Empty:
            return 0
        else:
            return maxQ

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        # Credit: update formula <- lecture slides
        currentQ = self.getQValue(state, action)
        self.qValue[(state, action)] = currentQ + self.getAlpha() * (reward + (self.getGamma() * self.maxQValue(nextState)) - currentQ)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        self.visits[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        return self.visits[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        # Least-pick exploration
        if counts < self.getMaxAttempts():
            return -counts
        else:
            return utility - 1000
            

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)

        # Choose action
        bestExplore = Empty
        bestAction = Empty

        if self.getEpisodesSoFar() > self.getNumTraining():
            # Training phase passed, choose best utility
            for action in legal:
                qValue = self.getQValue(stateFeatures, action)

                if bestExplore == Empty:
                    bestExplore = qValue
                    bestAction = action
                elif qValue > bestExplore:
                    bestExplore = qValue
                    bestAction = action
        else:
            # Explore
            if util.flipCoin(self.epsilon):
                bestAction = random.choice(legal)
            else:
                # Least pick exploration
                for action in legal:
                    exploreValue = self.explorationFn(self.getQValue(stateFeatures, action), self.getCount(stateFeatures, action))

                    if bestExplore == Empty:
                        bestExplore = exploreValue
                        bestAction = action
                    elif exploreValue > bestExplore:
                        bestExplore = exploreValue
                        bestAction = action
        
        # Update Q-values and count
        previousState = self.previousState
        
        if previousState != Empty:
            self.learn(previousState, self.previousAction, self.computeReward(previousState, stateFeatures), stateFeatures)

        self.updateCount(stateFeatures, bestAction)

        # Track previous
        self.previousState = stateFeatures
        self.previousAction = bestAction

        return bestAction

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
            print(GameStateFeatures(state))
        
        # Learn final state, and reset values
        stateFeatures = GameStateFeatures(state)
        previousState = self.previousState

        self.learn(previousState, self.previousAction, self.computeReward(previousState, stateFeatures), stateFeatures)
        self.previousState = Empty
        self.previousAction = Empty

