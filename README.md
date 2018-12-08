Group Name: Simulnance

Group participants names: Kedir Firehiwot Nesro, Menichelli Alessandro, Wüs Vincent

Project Title: Trade!t

Programming language: Python


# Modeling and Simulation of Social Systems Fall 2018 – Research Plan (Template)
(text between brackets to be removed)

> * Group Name: (be creative!)
> * Group participants names: (alphabetically sorted by last name)
> * Project Title: (can be changed later)
> * Programming language: (Python or MATLAB)

## General Introduction

(States your motivation clearly: why is it important / interesting to solve this problem?)
(Add real-world examples, if any)
(Put the problem into a historical context, from what does it originate? Are there already some proposed solutions?)

## The Model

(Define dependent and independent variables you want to study. Say how you want to measure them.) (Why is your model a good abstraction of the problem you want to study?) (Are you capturing all the relevant aspects of the problem?)


## Fundamental Questions

(At the end of the project you want to find the answer to these questions)
(Formulate a few, clear questions. Articulate them in sub-questions, from the more general to the more specific. )


## Expected Results

(What are the answers to the above questions that you expect to find before starting your research?)


## References 

(Add the bibliographic references you intend to use)
(Explain possible extension to the above models)
(Code / Projects Reports of the previous year)


## Research Methods

(Cellular Automata, Agent-Based Model, Continuous Modeling...) (If you are not sure here: 1. Consult your colleagues, 2. ask the teachers, 3. remember that you can change it afterwards)


## Other

(mention datasets you are going to use)

# Reproducibility

## Light test

In our model we want to simulate a financial market.
For this purpose we used a agent based approach.
The agents represent traders, which trade one single asset.
To simulate how traders place their offers we had to make assumptions on how they behave.
In the order generation process we assume that traders place their offers with respect to a normal distribution.

In this simple reproducibility test we let you simulate the market and plot the estimated price path of the stock.

1. Make sure to have all the packages needed installed. The two packages "matplotlib" and "mesa" are not preinstalled.
2. Run the file "Reproducibility_light" with python without changing any of the code. After say 5 seconds you should receive a plot of the price path. This result is obtained for a small standard deviation of the normal distribution, which corresponds to traders behaving all very simmilar.
3. Search for the global variable "std" which has been set to 0.001 by default. Change it to 0.01 and run the programm again.
The plot which you receive this time should vary stronger around 100 then before, which corresponds to traders behaving very differently. This happens when there's high uncertainty about te actuall stock price.

## Full test
(step by step instructions to reproduce all your results.) 
