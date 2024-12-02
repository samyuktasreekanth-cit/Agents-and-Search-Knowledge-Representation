# Assignment 1: Agents-and-Search-Knowledge-Representation
Knowledge Representation Assignment from the Masters in Artificial Intelligence course from MTU using vital knowledge from the libraries involved in the AIMA-python public library. 

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
___________________________________________________________________

# Assignment 2: 

This project utilised knowledge in Logical reasoning via first-order logic (FOL), Baysian Networks, Implementation and analysis of Naive Bayes Classifiers, Data processing and Pre-processing using the IRIS dataset. 
Encode the KB,encode the facts and run the Inference using forward chaining and backward chaining algorithm provided by the AIMA library to derive all possible inferences from the given facts and rules. Compare and contrast the approach. 
 
 1.1 LOGICAL REASONING
 In first-order logic (FOL) define the following clauses relating to family relationships:
1. Sibling Relationship: If two people share the same mother and father, and they are not the sameperson, then they are siblings.
2. Parent Relationship: If someone is a parent of a child, then that child is not the parent of the person.
3. Maternal Grandparent Relationship: A person is a maternal grandparent of a child if they are the mother of the child’s parent.
4. Spousal Relationship: If someone is a spouse of another person, then vice-versa holds true, and a person is not a spouse of themselves.
 Now provide facts related to these relationships, and we can then employ inference mechnisms to derive new knowledge about the relationships.
 • Alice and Bob are the parents of Carol.
 • Alice and Bob are the parents of Dave.
 • Eve is the spouse of Dave.
 • Carol is the parent of Frank.
 Using these facts, we can infer additional information, such as:
 • Carol and Dave are siblings.
 • Alice is the maternal grandparent of Frank.
 • Dave and Eve are spouses to each other.
 • Bob is also a grandparent of Frank, but since we’ve only defined “MaternalGrandparent”, we cant infer “PaternalGrandparent” without additional rules.

