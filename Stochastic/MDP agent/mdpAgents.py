# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print "Starting up MDPAgent!"
        name = "Pacman"
        self.currentTrack = None    # for debugging purposes

        self.width = None
        self.height = None

        # trackers (to save computational power)
        self.foundFoods = []
        self.capsules = []
        self.ghostSpawn = []
        self.deadends = []
        self.baseValueMap = {}

        # the rewards depending on whats on a certain coordinate
        self.discount = 0.6 # how relevant future steps are
        self.foodReward = 1

        self.pacmanReward = -0.0001 # negative to encourage moving
        self.pathReward = 0 # 0, because negative may discourage pacman from moving due to position of walls
        self.deadEndRewardReduction = -self.foodReward
        self.ghostRespawnReward = -self.foodReward * 2    # dangerous when ghosts may respawn
        self.capsuleReward = -self.foodReward  # want to save these for when they're needed
        
        self.allDirections = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]  # excluding STOP


    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        #print "Running registerInitialState for MDPAgent!"
        #print "I'm at:"
        #print api.whereAmI(state)
        self.currentTrack = api.whereAmI(state) # for debugging purposes

        # retrieve the necessary state information
        self.foundFoods = api.food(state)
        self.capsules = api.capsules(state)
        self.ghostSpawn = api.ghosts(state)
        
        self.registerDimensions(state)
        self.registerPaths(state)
        self.registerValues()
        
    # This is what gets run in between multiple games
    def final(self, state):
        #print "Looks like the game just ended!"
        #print "Food left: ", len(self.foundFoods)

        # reset values
        self.currentTrack = api.whereAmI(state) # for debugging purposes

        self.width = None
        self.height = None

        self.foundFoods = []
        self.capsules = []
        self.ghostSpawn = []
        self.deadends = []
        self.baseValueMap = {}
        

    # registers the width-height of the current game
    def registerDimensions(self, state):
        corners = api.corners(state)

        # get width and height
        for coordinate in corners:
            if coordinate[0] > self.width:
                self.width = coordinate[0]
            if coordinate[1] > self.height:
                self.height = coordinate[1]
        
        # correct dimensions, considering coordinates start at 0
        self.width += 1
        self.height += 1

    # registers the paths (and deadends) of the current game   
    def registerPaths(self, state):
        walls = api.walls(state)

        # set values for each traversable coordinate
        for i in range(self.width):
            for j in range(self.height):
                if (i, j) not in walls:
                    self.baseValueMap[(i, j)] = self.pathReward

        # register deadends (only 1 legal move)
        for coordinate in self.baseValueMap.keys():
            count = 0
            for direction in self.allDirections:
                if self.getTransitionVector(coordinate, direction) != coordinate:
                    count += 1

            if count == 1:
                self.deadends.append(coordinate)

    # registers the values of states (food, capsules, deadends)
    def registerValues(self):
        for food in self.foundFoods:
            self.baseValueMap[food] = self.foodReward
        
        for capsule in self.capsules:
            self.baseValueMap[capsule] = self.capsuleReward

        for deadend in self.deadends:
            self.baseValueMap[deadend] += self.deadEndRewardReduction


    # get the final converged policy
    def getPolicy(self, valueMap):
        policy = dict()

        # create initial policy
        for coordinate in valueMap.keys():
            policy[coordinate] = self.getMaxExpectedUtility(valueMap, coordinate)[0]

        return self.convergePolicy(valueMap, False, policy, 0)

    # iterate computation of values until convergence is maintained for 3 consecutive recursions
    def convergePolicy(self, valueMap, converged, policy, count):
        if converged == True:
            count += 1
            if count == 3:
                return (valueMap, policy)
        else:
            count = 0

        tempMap = dict(valueMap)
        tempPolicy = dict(policy)

        # update utility rewards
        for coordinate in valueMap.keys():
            tempMap[coordinate] += (self.discount * self.getExpectedUtility(valueMap, coordinate))

        # update policy
        for coordinate in policy.keys():
            tempPolicy[coordinate] = self.getMaxExpectedUtility(tempMap, coordinate)[0]

        return self.convergePolicy(tempMap, (policy == tempPolicy), tempPolicy, count)

    # get the next expected utility of a certain coordinate
    def getExpectedUtility(self, valueMap, coordinate):
        accumulator = 0

        for direct in self.allDirections:
            # calculate utility value of a given direction
            accumulator += 0.8 * valueMap[self.getTransitionVector(coordinate, direct)]
            accumulator += 0.1 * valueMap[self.getTransitionVector(coordinate, self.translateDirection(direct, Directions.WEST))]
            accumulator += 0.1 * valueMap[self.getTransitionVector(coordinate, self.translateDirection(direct, Directions.EAST))]

        return accumulator / 4

    # get the maximum expected utility, and it's direction
    def getMaxExpectedUtility(self, valueMap, coordinate):
        bestDirection = None
        bestValue = None

        for direct in self.allDirections:
            # calculate utility value of a given direction
            accumulator = 0
            accumulator += 0.8 * valueMap[self.getTransitionVector(coordinate, direct)]
            accumulator += 0.1 * valueMap[self.getTransitionVector(coordinate, self.translateDirection(direct, Directions.WEST))]
            accumulator += 0.1 * valueMap[self.getTransitionVector(coordinate, self.translateDirection(direct, Directions.EAST))]

            # update the "max" utility
            if bestValue == None:
                bestDirection = direct
                bestValue = accumulator
            elif accumulator > bestValue:
                bestDirection = direct
                bestValue = accumulator

        return (bestDirection, bestValue)


    # translate vector based on direction
    def translateVector(self, coordinate, direction):
        if direction == Directions.WEST:
            return (coordinate[0] - 1, coordinate[1])
        elif direction == Directions.EAST:
            return (coordinate[0] + 1, coordinate[1])
        elif direction == Directions.NORTH:
            return (coordinate[0], coordinate[1] + 1)
        elif direction == Directions.SOUTH:
            return (coordinate[0], coordinate[1] - 1)
        else:
            return (coordinate[0], coordinate[1])

    # translate vector based on direction, keep current coordinate if direction leads to a wall
    def getTransitionVector(self, coordinate, direction):
        newVector = self.translateVector(coordinate, direction)
            
        if newVector not in self.baseValueMap:
            newVector = coordinate
        
        return newVector

    # returns the final direction of 2 directions
    def translateDirection(self, directionx, directiony):
        dirValue = {
            Directions.NORTH : 0,
            Directions.EAST : 1,
            Directions.SOUTH : 2,
            Directions.WEST : 3,
            Directions.STOP : 0
        }

        value = (dirValue[directionx] + dirValue[directiony]) % 4

        if value == 0:
            return Directions.NORTH
        elif value == 1:
            return Directions.EAST
        elif value == 2:
            return Directions.SOUTH
        elif value == 3:
            return Directions.WEST


    # get the reward of a coordinate depending on the type of path it is
    def getPathReward(self, coordinate):
        if coordinate in self.deadends:
            return self.deadEndRewardReduction
        else:
            return self.pathReward

    # returns targets found and its distance within a certain number of moves
    def getTargetsInRadius(self, source, targets, moves):
        affectedVectors = list()

        newVectors = list()
        newVectors.append(source)

        found = {}  #coordinate: distance

        # search within a limited number of moves
        for i in range(moves):
            tempVectors = list()

            # search adjacent to the last batch of explored coordinates
            for vect in newVectors:
                for direct in self.allDirections:
                    newVector = self.getTransitionVector(vect, direct)
                    
                    if newVector not in affectedVectors:
                        # record number of moves to reach target
                        if newVector in targets:
                            found[newVector] = i + 1

                        affectedVectors.append(newVector)
                        tempVectors.append(newVector)

            newVectors = tempVectors
        
        return found


    # debugging: displays the map and policy of coordinates
    def displayPolicy(self, pacman, ghosts, policy):
        text = ""
        directionText = {
            Directions.WEST: "<",
            Directions.EAST: ">",
            Directions.NORTH: "^",
            Directions.SOUTH: ",",
        }

        for j in range(self.height):
            for i in range(self.width):
                if (i, self.height - 1 - j) not in policy:
                    text += "#"
                else:
                    if (i, self.height - 1 - j) == self.currentTrack:
                        text += "@"
                    elif (i, self.height - 1 - j) == pacman:
                        text += "P"
                    elif (i, self.height - 1 - j) in ghosts:
                        text += "G"
                    else:
                        text += directionText[policy[(i, self.height - 1 - j)]]
            text += "\n"

        return text

    # debugging: allows navigation of utilities in the current game state
    def viewUtility(self, valueMap, pacman, ghosts, policy):
        keyDirection = {
            "w": Directions.NORTH,
            "a": Directions.WEST,
            "s": Directions.SOUTH,
            "d": Directions.EAST,
        }
            
        userInput = raw_input()    # allows me to view policy of each move in the game
        
        if userInput in keyDirection.keys():
            self.currentTrack = self.getTransitionVector(self.currentTrack, keyDirection[userInput])
            print self.displayPolicy(pacman, ghosts, policy)
            print self.currentTrack, valueMap[self.currentTrack]
            self.viewUtility(valueMap, pacman, ghosts, policy)
        elif userInput == "get":
            print self.getLayout(pacman, ghosts)
            self.viewUtility(valueMap, pacman, ghosts, policy)
        else:
            return

    # debugging: displays layout text that can be pasted into a .lay file
    def getLayout(self, pacman, ghosts):
        text = ""

        for j in range(self.height):
            for i in range(self.width):
                if (i, self.height - 1 - j) not in self.baseValueMap:
                    text += "%"
                else:
                    if (i, self.height - 1 - j) == pacman:
                        text += "P"
                    elif (i, self.height - 1 - j) in ghosts:
                        text += "G"
                    elif (i, self.height - 1 - j) in self.foundFoods:
                        text += "."
                    else:
                        text += " "
            text += "\n"

        return text


    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)

        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        pacman = api.whereAmI(state)

        # update state trackers (same result can be obtained from api calls, but is less computationally taxing)
        if pacman in self.foundFoods:
            self.foundFoods.remove(pacman)
            self.baseValueMap[pacman] = self.getPathReward(pacman)
        elif pacman in self.capsules:
            self.capsules.remove(pacman)
            self.baseValueMap[pacman] = self.getPathReward(pacman)

        # set state dependent rewards
        closeGhostReward = -self.foodReward * 2 * len(self.foundFoods)   # large amounts of food can skew the rewards
        farGhostReward = -self.foodReward * 0.5 * len(self.foundFoods)
        lastFoodReward = -closeGhostReward * self.discount # more risk can be taken if it is a terminating state
        urgentCapsuleReward = -closeGhostReward * self.discount  # capsules can save pacman from dire situations
        scaredGhostReward = min(self.foodReward * 5, -farGhostReward)

        # last food reward (terminating state)
        if len(self.foundFoods) == 1:
            self.baseValueMap[self.foundFoods[0]] = lastFoodReward

        # set rewards of current state
        currentValueMap = dict(self.baseValueMap)

        currentValueMap[pacman] = self.pacmanReward

        ghostLocations = api.ghosts(state)
        for i in range(len(ghostLocations)):
            # ghost locations are floats when vulnerable 
            ghostLocations[i] = tuple(map(int, ghostLocations[i]))

        ghosts = api.ghostStatesWithTimes(state)
        ghostDistances = self.getTargetsInRadius(pacman, ghostLocations, 10)
        
        safeRespawn = True  # respawn area is dangerous if ghosts can be eaten
        danger = False  # determines importance of capsules

        # set ghost rewards based on their states
        for (coordinate, state) in ghosts:
            # ghost locations are floats when vulnerable 
            coordinate = tuple(map(int, coordinate))

            # if ghosts can be eaten, respawn area is dangerous
            if state > 0:
                safeRespawn = False

            if coordinate in ghostDistances.keys():
                # chase if theres a slightly above average chance to catch ghost
                if ghostDistances[coordinate] < state * 0.5:
                    currentValueMap[coordinate] = scaredGhostReward
                elif state < 10:
                    # be afraid of ghost depending on how far away it is
                    if ghostDistances[coordinate] < 6:
                        currentValueMap[coordinate] = closeGhostReward
                        danger = True
                    else:
                        currentValueMap[coordinate] = farGhostReward
            else:
                # ghost is too far to be of a threat
                currentValueMap[coordinate] = self.pathReward


        # hostile ghosts nearby -> increased capsule reward
        if danger:
            for cap in self.capsules:
                currentValueMap[cap] = urgentCapsuleReward

        # respawn area is dangerous if ghosts can be eaten
        if safeRespawn == False:
            for ghostRespawn in self.ghostSpawn:
                currentValueMap[ghostRespawn] = self.ghostRespawnReward

        # policy iteration process
        (currentValueMap, policy) = self.getPolicy(currentValueMap)

        # select optimal action
        newDirect = policy[pacman]
        
        # debugging purposes
        #print self.displayPolicy(pacman, ghostLocations, policy)
        #print policy[pacman]
        #self.viewUtility(currentValueMap, pacman, ghostLocations, policy)

        return api.makeMove(newDirect, legal)
