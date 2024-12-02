# Agents-and-Search-Knowledge-Representation
Knowledge Representation Assignment from my Masters in Artificial Intelligence using vital knowledge from the libraries involved in the AIMA-python public library. 

1.1 BUILDING YOUR AGENT BASED WORLD
____________________________________

This part of the project is used to create a 2D world that will play host to a game to be played by my chosen agent types. There is detailed explantion on the mechanism of the world, task environment, PEAS description etc. The agents I chose for the game are Random Agemt, Simple Reflex Agent and Model Based Agent. 

My project takes inspiration from the classic Snake and Apple game. I have a 2D Environment (Forest 2D) which is my grid world.When building the word, I keep 3 things in mind: snake, apple and boundary. In the context of the environment, I can change its size, check if the snake crosses its boundary and place its food source(apple) randomly or in a specific grid.
 1. Snake: This is the agent. For the context of this game, only its head is considered and takes up 1 grid space. The head can control which direction it is going to move in: up, down, left or right. (Note: Snake cannot turn back on itself as it's considered touching its body). The snake always starts in a default starting position, facing down (of course, this
 can be tweaked as needed).

 2. Apple: This is the food source that the relevant agent type tries to get to. If the snake finds an apple, it eats it. This is how the measurement of success is defined. The more apples the snake eats, the better its performance measure. In the game, we have the option to generate ‘n’ number of apples at a random grid location (subject to environment size) or place the apple in a known grid location.
  
 3. I've adapted the game such that the game ends/is complete when there are no more apples provided to the environment or when the snake goes out of bounds from the environment. So, if the snake hits a wall, it dies.

1.2 SEARCHING YOUR WORLD
_________________________

The goal state for the search world is to get to the optimal path to efficiently get to the apple. 

 This project uses 3 uninformed search techniques to compare, experiment and achieve this:
 1. Breadth first Graph search
 2. Depth First Graph Search
 3. Iterative Deepening Search

References: Review of the "agents.ipynb" amd "search.ipynb" from the AIMA-python repository. 
