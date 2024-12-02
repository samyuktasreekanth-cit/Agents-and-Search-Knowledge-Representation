print("Assignment2.py...")

from utils import *
from logic import *
from notebook import psource

from probability import *
from utils import print_table
from notebook import psource, pseudocode, heatmap

import pandas as pd
import matplotlib as plt


from learning import *
from probabilistic_learning import *
from notebook import *
from learning import NaiveBayesLearner

def one_point_one():
    print("\n 1.1 LOGICAL REASONING ")

    # Encode the KB
    #--------------

    clauses = []
    s1, s2, m, f, p, c, g, sp1, sp2 = expr('s1, s2, m, f, p, c, g, sp1, sp2')

    # Sibling relationship
    clauses.append(expr('Parent(m, s1) & Parent(m, s2) & Parent(f, s1) & Parent(f, s2) & NotTheSamePerson(s1, s2) ==> Sibling(s1, s2)'))

    # Parent relationship
    clauses.append(expr('Parent(p, c) ==> Different(c, p)'))

    # Maternal grandparent relatipnship
    clauses.append(expr('Parent(g, p) & Parent(p, c) & Female(g) ==> MaternalGrandparent(g, c)'))

    # Spousal relaionship
    clauses.append(expr('Spouse(sp1, sp2) & Different(sp1, sp2)==> Spouse(sp2, sp1)'))
    

    # Encode the facts
    #-----------------

    # Alice and Bob are the parents of Carol.

    clauses.append(expr('Parent(Alice, Carol)'))
    clauses.append(expr('Parent(Bob, Carol)'))

    # Alice and Bob are the parents of Dave.
    clauses.append(expr('Parent(Alice, Dave)'))
    clauses.append(expr('Parent(Bob, Dave)'))

    #Eve is the spouse of Dave
    clauses.append(expr('Spouse(Eve, Dave)'))
    
    # Carol is the parent of Frank
    clauses.append(expr('Parent(Carol, Frank)'))

    # Alice, Carol and Eve are female
    clauses.append(expr('Female(Alice)'))
    clauses.append(expr('Female(Carol)'))
    clauses.append(expr('Female(Eve)'))

    #Person is person - attempt
    # clauses.append(expr('SamePerson(Alice, Alice)'))
    # clauses.append(expr('SamePerson(Bob, Bob)'))
    # clauses.append(expr('SamePerson(Carol, Carol)'))
    # clauses.append(expr('SamePerson(Dave, Dave)'))
    # clauses.append(expr('SamePerson(Eve, Eve)'))
    # clauses.append(expr('SamePerson(Frank, Frank)'))

    people = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank']
    for person1 in people:
        for person2 in people:
            if person1 != person2:
                clauses.append(expr('NotTheSamePerson({}, {})'.format(person1, person2)))


    # Infer using forward chaining algorithm provided by the AIMA library
    #---------------------------------------------------------------------

    crime_kb = FolKB(clauses)

    print("\n")

    print("Forward Chaining:")
    print("--------------------")
    answer1 = fol_fc_ask(crime_kb, expr('Sibling(x,y)'))
    print(list(answer1))

    print("Backward Chaining:")
    print("--------------------")
    answerB1 = fol_bc_ask(crime_kb, expr('Sibling(x,y)'))
    print(list(answerB1))
    
    print("\n")

    print("Forward Chaining:")
    print("--------------------")
    answer2 = fol_fc_ask(crime_kb, expr('MaternalGrandparent(x, Frank)'))
    print(list(answer2))

    print("Backward Chaining:")
    print("--------------------")
    answerB2 = fol_bc_ask(crime_kb, expr('MaternalGrandparent(x, Frank)'))
    print(list(answerB2))

    print("\n")

    print("Forward Chaining:")
    print("--------------------")
    answer3 = fol_fc_ask(crime_kb, expr('Spouse(x,y)'))
    print(list(answer3))

    print("Backward Chaining:")
    print("--------------------")
    print("RecursionError: maximum recursion depth exceeded while calling a Python object (uncommented. You said its ok(23/11/2023), cuz relationship is recursive)")
    # RecursionError: maximum recursion depth exceeded while calling a Python object


    # answerB3 = fol_bc_ask(crime_kb, expr('Spouse(x,y)'))
    # print(list(answerB3))

def one_point_two():
    print("\n 1.2 BAYESIAN NETWORKS ")

    # NOTE: Referencing code directly from probability.ipynb from AIMA-PYTHON

    T, F = True, False

    # ('Child', 'Parents', {Probability})

    env_tech_impact = BayesNet([
        ('TechInnovation', '', 0.6),
        ('Urbanization', '', 0.7),
        ('CarbonEmissions', 'TechInnovation Urbanization', {(T, T): 0.8, (T, F): 0.1, (F, T): 0.9, (F, F): 0.7}),
        ('EcologicalFootprint', 'CarbonEmissions', {T: 0.9, F: 0.2}),
        ('CleanEnergyAdoption', 'TechInnovation', {T: 0.7, F: 0.4}),
        ('JobMarket', 'Urbanization', {T: 0.8, F: 0.3})
    ])

    # Variable Elimination
    demo_querying_probability = elimination_ask('CarbonEmissions', dict(TechInnovation=T, Urbanization=T), env_tech_impact).show_approx()
    print("Demo querying the network to obtain probabilities or predictions")
    print("----------------------------------------------------------------")
    print(demo_querying_probability)


def one_point_three():
    print("\n 1.3.1 DATA SELECTION AND PREPROCESSING ")

    # Citation: Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.
    # Also using knowledge from the module Practical Machine learning to read_csv and make it a dataFrame 

    uci_iris_dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    variable_column_name = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    # iris = DataSet(name="iris") <- using this way was giving 'AttributeError: 'DataSet' object has no attribute 'attrnames''

    df_all = pd.read_csv(uci_iris_dataset_url, names=variable_column_name)

    # subset of 10-15 rows of each class
    df = df_all.groupby("class").head(15)

    print("\nDataset of choice 1: Iris Dataset")

    # Calculate the Prior probabilities for the classes - Iris Dataset
    prior_prob = df["class"].value_counts(normalize=True)
    print("\n Prior probabilities for iris dataset:") 
    print("-----------------------------------------")
    print(prior_prob)

    # Estimate the probability of the evidence within the dataset - Iris Dataset

    # Extracting a subset from the dataset via setosa
    setosa_df = df[df["class"] == "Iris-setosa"]
    setosa_sepal_width_3 = setosa_df[setosa_df["sepal_width"] == 3.0]
    prob_of_evidence = len(setosa_sepal_width_3)/len(df)

    print("\nEstimate the probability of the evidence within the iris dataset:") 
    print("------------------------------------------------------------")
    print("Probability that the 'sepal_width' of Iris-setosa is 3.0 in the iris dataset: " + str(prob_of_evidence))

    # Determine the likelihood of the evidence (the numerator of Bayes’ formula) - Iris Dataset

    print("\nDetermine the likelihood of the evidence (the numerator of Bayes formula) - iris dataset")
    print("--------------------------------------------------------------------------")
    likelihood = len(setosa_sepal_width_3)/len(setosa_df)
    print("Probability that the 'sepal_width' of Iris-setosa is 3.0 cm: " + str(prob_of_evidence))
    numerator = likelihood * prior_prob["Iris-setosa"]
    print("Numerator of Bayes formula P(B|A) * P(A):", str(numerator))

    print("\n ________________________________________________________________________")

    print("\nDataset of choice 2: Wine Dataset")

    uci_wine_dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

    wine_variable_column_name = ["class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"]

    # note: using only a sample as discussed of 10-50 rows
    df_all_wine = pd.read_csv(uci_wine_dataset_url, names=wine_variable_column_name)

    # the column to predict must be at the end
    class_column = df_all_wine.pop("class")
    df_all_wine.insert(len(df_all_wine.columns), "class", class_column)

    # subset of 10-15 rows of each class
    df_wine = df_all_wine.groupby("class").head(15)

    # iris = DataSet(name="iris") <- using this way was giving 'AttributeError: 'DataSet' object has no attribute 'attrnames''

    # Calculate the Prior probabilities for the classes - Wine Dataset

    prior_prob_wine = df_wine["class"].value_counts(normalize=True)
    print("\nPrior probabilities for wine dataset:") 
    print("-----------------------------------------")
    print(prior_prob_wine)

    # Estimate the probability of the evidence within the dataset - Wine Dataset

    # Extracting a subset from the dataset via alchocol content > 13
    alcohol_more_13 = df_wine[df_wine["alcohol"] > 13]
    proof_of_evidence_alcohol = len(alcohol_more_13)/len(df_wine)
    print("\nEstimate the probability of the evidence within the wine dataset:") 
    print("---------------------------------------------------------------------")
    print("Probability that the alcohol content is more than 13 in the wine dataset: " + str(proof_of_evidence_alcohol))

    print("\nDetermine the likelihood of the evidence (the numerator of Bayes formula) - Wine dataset")
    print("---------------------------------------------------------------------------------------------")
    evidence = df_wine["alcohol"] >13
    likelihood_wine = df_wine[evidence].groupby("class").size() / df_wine.groupby("class").size()
    print("Probability that the 'alcohol' content is greater than 13 for each class:" + str(likelihood_wine))


    print("\nOverall probability that the 'alcohol' content is greater than 13 for all classes: " + str(likelihood_wine.sum()))
    numerator_wine = likelihood_wine * prior_prob_wine
    print("\nNumerator of Bayes formula P(B|A) * P(A): " +  str(numerator_wine))
    print("\nOverall Numerator of Bayes formula P(B|A) * P(A) for all classes: " +  str(numerator_wine.sum()))


    print("\n ________________________________________________________________________")

    print("\n  1.3.2 NAIVE BAYES CLASSIFICATION")

    # Calculate the full Bayes’ formula 𝑃(𝐴|𝐵)=𝑃(𝐵|𝐴)∗𝑃(𝐴)/𝑃(𝐵)

    print("""
    Bayes formula P(A|B)=P(B|A)*P(A)/P(B) 
    ---------------------------------------
    - P(B|A) is the likelihood,
    - P(A) is the prior probability
    - P(B) is the probability of the evidence
    """)

    # Iris Dataset - implement the Niave Bayes Algorithm

    df_train = df
    df_test = df_all[~df_all.index.isin(df_train.index)]

    print("\n Bayes formula P(A|B) = P(B|A) * P(A) / P(B) - iris dataset")
    print("-------------------------------------------------------------")
    naive_formula_iris = numerator/prob_of_evidence
    print(naive_formula_iris)

    # Wine Dataset - implement the Naive Bayes Algorithm

    print("\n Bayes formula P(A|B) = P(B|A) * P(A) / P(B) - wine dataset")
    print("-------------------------------------------------------------")
    naive_formula_wine = numerator_wine/proof_of_evidence_alcohol
    print(naive_formula_wine)

    # NaiveBayesLearner() Classifier -  Iris dataset

    print("\n NaiveBayesLearner() Classifier -  Iris dataset")
    print("------------------------------------------------")
    nBD_iris = NaiveBayesLearner(DataSet(df_train.values.tolist()), continuous=False)

    correct = 0
    for i, row in df_test.iterrows():
        classification = nBD_iris(row)
        if (classification == row["class"]):
            correct += 1
            
    print("accuracy for iris: " + str(correct / len(df_test)))

    # NaiveBayesLearner() Classifier -  Wine dataset
    print("\n NaiveBayesLearner() Classifier -  Wine dataset")
    print("-------------------------------------------------")

    df_train_wine = df_wine

    df_test_wine = df_all_wine[~df_all_wine.index.isin(df_train_wine.index)]

    nBD_wine = NaiveBayesLearner(DataSet(df_train_wine.values.tolist()), continuous=False)

    correct_wine = 0
    for i, row in df_test_wine.iterrows():
        classification = nBD_wine(row)
        if (classification == row["class"]):
            correct_wine += 1
            
    print("accuracy for wine: " + str(correct_wine / len(df_test_wine)))
 
    # creating the bar plot
    plt.bar(["correct", "incorrect"], [correct, len(df_test) - correct], color ='blue', width = 0.4)

    plt.title("Correct vs Incorrect Predictions (Iris)")
    plt.show()
    
    # creating the bar plot
    plt.bar(["correct", "incorrect"], [correct_wine, len(df_test_wine) - correct_wine], color ='blue', width = 0.4)

    plt.title("Correct vs Incorrect Predictions (Wine)")
    plt.show()
    

def main():
    
    one_point_one()
    one_point_two()
    one_point_three()
    
main()
