import sys
import os

sys.path.append(os.path.curdir)

from agents import *
from random import choice

from search import *
from search import breadth_first_tree_search

# ------------------------------------------------------------------------------------------------------

class Food(Thing):
    pass

class Forest2D(GraphicEnvironment):
    def percept(self, agent):
        '''return a list of things that are in our agent's location'''
        things = self.list_things_at(agent.location)
        loc = copy.deepcopy(agent.location) # find out the target location
        if agent.direction.direction == Direction.R:
            loc[0] += 1
        elif agent.direction.direction == Direction.L:
            loc[0] -= 1
        elif agent.direction.direction == Direction.D:
            loc[1] += 1
        elif agent.direction.direction == Direction.U:
            loc[1] -= 1
        if not self.is_inbounds(loc):
            things.append(Bump())
        return things

    def program(self, percept):
    # Determine the next action based on the percept
    # For now, let's just keep moving in the current direction
        return self.direction
    
    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        if action == 'turnright':
            #agent.moveforward()
            agent.turn(Direction.R)
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
        elif action == 'turnleft':
            #agent.moveforward()
            agent.turn(Direction.L)      
            print('{} decided to {} at location: {}'.format(str(agent)[1:-1], action, agent.location))
        elif action == 'moveforward':
            agent.moveforward()
            print('{} decided to move {}wards at location: {}'.format(str(agent)[1:-1], agent.direction.direction, agent.location))
        elif action == "eat":
            items = self.list_things_at(agent.location, tclass=Food)
            if len(items) != 0:
                if agent.eat(items[0]):
                    print('{} ate {} at location: {}'
                          .format(str(agent)[1:-1], str(items[0])[1:-1], agent.location))
                    self.delete_thing(items[0])

        # a message and check to see if the agent is out of bounds 
        # if the snake gets out of bounds, he dies and game is over
        loc = copy.deepcopy(agent.location) # find out the target location            
        if self.is_inbounds(loc):
            print('Agent is within environment bounds in location {}'.format(agent.location))
        else:
            agent.alive = False
            print('Agent is NOT within environment bounds in location {}. Game over'.format(agent.location))

    def is_done(self):
        '''By default, we're done when we can't find a live agent, ~
        but to prevent killing our snake, we will stop before itself - when there is no more food'''
        no_edibles = not any(isinstance(thing, Food) for thing in self.things)
        dead_agents = not any(agent.is_alive() for agent in self.agents)
        return dead_agents or no_edibles

    # a function to use my own in_bounds function 
    def is_inbounds(self, location):
        x, y = location
        return (self.x_start <= x and x < self.x_end and self.y_start <= y and y < self.y_end)

    # a function to generate num of apples of choice at a random grid location
    def add_random_apple(self, num_apples):
        for i in range(num_apples):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            apple = Food()
            self.add_thing(apple, [x,y])
        return [x,y]
    
# 1.1 BUILDING YOUR WORLD ---------------------------------------------------------------------------------

class EnergeticSnake(Agent):
    location = [0,1]
    direction = Direction("down")
    performance= 0 
    
    def moveforward(self, success=True):
        '''moveforward possible only if success (i.e. valid destination location)'''
        if not success: 
            return
        if self.direction.direction == Direction.R:
            self.location[0] += 1
        elif self.direction.direction == Direction.L:
            self.location[0] -= 1
        elif self.direction.direction == Direction.D:
            self.location[1] += 1
        elif self.direction.direction == Direction.U:
            self.location[1] -= 1
            
    def turn(self, d):
        self.direction = self.direction + d

    def eat(self, thing):
        '''returns True upon success or False otherwise'''
        if isinstance(thing, Food):
            self.performance += 1 #update performance measure if an apple is eaten
            return True
        return False

# Agent 1: This agent makes decisions completly at random, without considering its percept history
def RandomAgentProgram(percepts):
    ''' An agent that chooses an action at random, ignoring all percepts'''
    choice = random.choice(('turnright', 'turnleft', 'moveforward', 'eat'))
    return choice 

# Agent 2: this agent makes its decisions based on its current percepts(ignoring the rest of the percept history), 
# using a set of condition action rules. 
def SimpleReflexAgentProgram(percepts):
    for p in percepts:
        if isinstance(p, Food):
            return 'eat'
        elif isinstance(p,Bump):
            turn = False
            choice = random.choice(('turnright', 'turnleft'))
            return choice
    return 'moveforward'

# Agent 3: This agent makes decisions based on current percepts + makes decisions while maintaining an internal model of the world
# the model is updated based on the percepts/actions. 
# note: using ModelBasedVacuumAgent() from agents.py as a reference

def ModelBasedAgentProgram():
    model = {'food_eaten': None, 'location': None} # None cause agent doesnt know it beforehand

    def program(percepts):
        food = False
        bump = False
        for p in percepts:
            if isinstance(p, Food):
                if model['food_eaten'] is None:
                    model['food_eaten'] = 1
                else:
                    model['food_eaten'] += 1            
                food = True
            elif isinstance(p,Bump):
                bump = True
            else:
                model['location'] = p.location

        if food:
            return 'eat'
        elif bump:
            return random.choice(('turnright', 'turnleft'))
        else:
            return 'moveforward'
            
    return program

# 1.2 SEARCHING YOUR WORLD -----------------------------------------------------------------------------

class SnakeAppleGame(Problem):
    def __init__(self, initial, goal, environment, agent):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)
        self.environment = environment
        self.direction = agent.direction

    # Citation: using class PlanRoute(Problem) from search.py as a reference for some of the code below
    def actions(self, state):
        possible_actions = ['up', 'down', 'left', 'right']

        if self.direction == 'up':
            possible_actions.remove('down')
        elif self.direction == 'down':
            possible_actions.remove('up')
        elif self.direction == 'left':
            possible_actions.remove('right')
        elif self.direction == 'right':
            possible_actions.remove('left')

        # Prevent Bumps
        # list(possible_actions) is a copy of the list above so to easilty iterate and to modify it as we go
        # state - current position of the snake
        # new_state - coordinates of snake after after taking an action 
        for action in list(possible_actions):
            new_state = self.result(state, action)
            if not self.environment.is_inbounds(new_state):
                possible_actions.remove(action)
        
        return possible_actions
    
    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        x, y = state
        proposed_loc = list()
        
        if action == 'up':
            proposed_loc = (x, y + 1)
        elif action == 'down':
            proposed_loc = (x, y - 1)
        elif action == 'left':
            proposed_loc = (x - 1, y)
        elif action == 'right':
            proposed_loc = (x + 1, y)
        else:
            raise Exception('InvalidOrientation') # having this here for consistency with search.py
            
        return proposed_loc
        
    # snake goal state is when it eats an apple, goal is to eat an apple, snake gets to the grid/location where the apple is
    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """
        return state == self.goal

    # the path cost for this is incrementing one each time the snake takes a move
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1        


# THIS IS THE FUNCTION TO RUN AGENTS (1.1 BUILDING YOUR WORLD )--------------------------------------------------------

# snake is green color square 
# apple is red color square
# If you wanna see the colorful blocks on jupyter, use this import stataement in there
# import collections
# collections.Iterable = collections.abc.Iterable
# collections.Sequence = collections.abc.Sequence

def run_my_agent(agent):
    
    forest = Forest2D(5,5, color={'EnergeticSnake': (0,255,0), 'Food': (200, 0, 0)})

    # Store performance result for each agent snake
    performance_results = []

    agent_name, agent_program = agent
    
    print("Running {} ...".format(str(agent_name)))
    
    # Check if agent_program is a class and create an instance if necessary
    # need this for random and simplereflex agent
    if isinstance(agent_program, type):
        agent_program = agent_program()
        
    # Creating agent instances
    agent = EnergeticSnake(agent_program)
    
    # Adding agent to the forest environment 
    forest.add_thing(agent, [0,0])

    # Adding food(apple) to the environment, randomly with the specific num of apple
    forest.add_random_apple(1)

    #optional - This is how you manually add apples of your choice in the grids to the forest2D - useful for the search
    #apple1 = Food()
    #apple2 = Food()
    #forest.add_thing(apple1, [3,3])
    #forest.add_thing(apple2, [4,4])

    print("snake started at [0,0], facing down. Let's see if he found any food!")
    # remove the second paramenter 0 if you want a second by second print output
    forest.run(20, 0)

    performance_results.append((agent_name, agent.performance))


    # Print the performance results
    for agent_name, performance in performance_results:
        print('Performance of {} : {}'.format(str(agent_name), str(performance)))


    return performance_results

# THIS IS THE FUNCTION TO RUN SEARCH (1.2 SEARCHING YOUR WORLD)--------------------------------------------------------

# uninformed search

# 1. uninformed search - breadth_first_graph_search(BFS)
def run_my_BFS(problem):
    print("Running breadth_first_graph_search(BFS)...")
    solution = breadth_first_graph_search(problem)
    
    if solution is not None:
        path = solution.path()
        print("Path snake took to reach goal(apple):", path)
    else:
        print("No solution found. None Nodes ")
    return solution

# 2. uninformed search - depth_first_graph_search(DFS)
def run_my_DFS(problem):
    print("Running depth_first_graph_search(DFS)...")
    solution = depth_first_graph_search(problem)
    
    if solution is not None:
        path = solution.path()
        print("Path snake took to reach goal(apple):", path)
    else:
        print("No solution found. None Nodes ")
    return solution


# 3. uninformed search - iterative_deepening_search
def run_my_iterative_deep_search(problem):
    print("Running iterative_deepening_search...")
    solution = iterative_deepening_search(problem)
    
    if solution is not None:
        path = solution.path()
        print("Path snake took to reach goal(apple):", path)
    else:
        print("No solution found. None Nodes ")
    return solution

# 4. Informed search - best first graph search
def run_my_best_first_search(problem):
    print("Running best_first_graph_search...")
    # f : takes in an apple gives it the score of my choosing of 1
    solution = best_first_graph_search(problem, lambda apple : 1)
    
    if solution is not None:
        path = solution.path()
        print("Path snake took to reach goal(apple):", path)
    else:
        print("No solution found. None Nodes ")
    return solution

# 5.  Informed search - uniform_cost_search
def run_my_uniform_cost_search(problem):
    print("Running uniform_cost_search...")
    solution = uniform_cost_search(problem)
    
    if solution is not None:
        path = solution.path()
        print("Path snake took to reach goal(apple):", path)
    else:
        print("No solution found. None Nodes ")
    return solution

# Informed search - astar_search
def run_my_astar_search(problem):
    print("Running astar_search...")
    solution = astar_search(problem, lambda node : euclidean_distance([node.state[0]], [node.state[1]]))
    
    if solution is not None:
        path = solution.path()
        print("Path snake took to reach goal(apple):", path)
    else:
        print("No solution found. None Nodes ")
    return solution

def run_my_search_world(searcher):
    forest = Forest2D(5,5, color={'EnergeticSnake': (0,255,0), 'Food': (200, 0, 0)})

    # Create a snake agent
    snake_agent_for_search = EnergeticSnake()

    # initial position of the snake
    initial = (0, 0)

    # position of the apple
    goal = (3, 3)

    # Create the problem instance
    problem = SnakeAppleGame(initial, goal, forest, snake_agent_for_search)
    
    searcher(problem)
    
def compare_searches():
    forest = Forest2D(5,5, color={'EnergeticSnake': (0,255,0), 'Food': (200, 0, 0)})

    # Create a snake agent
    snake_agent_for_search = EnergeticSnake()

    # initial position of the snake
    initial = (0, 0)

    # position of the apple
    goal = (3, 3)

    # Create the problem instance
    problem = SnakeAppleGame(initial, goal, forest, snake_agent_for_search)
    
    header = ["Algorithm", "Actions/Goal Test/Result"]
    
    # using the compare searchers from search.py
    compare_searchers([problem], header,
                      searchers=[breadth_first_graph_search,
                                 depth_first_graph_search,
                                 iterative_deepening_search])

---------------------------------------------------------------------------------

# If you want to run an agent, replace with the names commented below ('name', AgentProgram), brackets included

# !!! Note for ruari: uncomment this line below one by one to run the agents

# performance_result = run_my_agent(('RandomAgentProgram', RandomAgentProgram))
# performance_result = run_my_agent(('SimpleReflexAgentProgram', SimpleReflexAgentProgram))
# performance_result = run_my_agent(('ModelBasedAgentProgram', ModelBasedAgentProgram()))
    
# Note: uncomment here as deired to run the desired uninformed search

# UNINFORMED SEARCH TECHNIQUES

# 1. this is breadth_first_graph_search (BFS)
# run_my_search_world(run_my_BFS)

# 2. this is depth_first_graph_search (DFS)
# run_my_search_world(run_my_DFS)

# 3. this is iterative_deepening_search
# run_my_search_world(run_my_iterative_deep_search)

# INFORMED SEARCH TECHNIQUES

# 4. this is best_first_graph_search 
# run_my_search_world(run_my_best_first_search)

# 5. this is uniform_cost_search 
# run_my_search_world(run_my_uniform_cost_search)

# 6. this is astar_search 
# run_my_search_world(run_my_astar_search)

# COMPARE SEARCHES

compare_searches()
    