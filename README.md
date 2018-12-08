
# Modeling and Simulation of Social Systems Fall 2018 – Research Plan (Template)
(text between brackets to be removed)

> * Group Name: (Simulnance)
> * Group participants names: (Kedir Firehiwot Nesro, Menichelli Alessandro, Wüst Vincent)
> * Project Title: (Trade!t)
> * Programming language: (Python)

## General Introduction

The primary motivation of the paper is to mimic a real financial market using agents based simulation. Secondary motivation is to identify and indicate possible impediments in simulating the market.States your motivation clearly: Although progress is made to simulate the real life behavior of Financial price behavior, due to the complexity of the system, there is still a gap to be filled in representing the financial market.)

## The Model

In the model Price of a stock is a dependent variable and the behaviour of traders is an independent variable that corresponds to the structure of the model. The variables are measured through agents, time and stock parameters. The amount of Agents are set, the amount of days and daily section in which the model is applied is set by authors . In addtion, Intial stock amount is given to agent and money creation is not allowed in the model.) (The model addresed key parameters used in the research of Raberto, 2005.Through using multiple scenarios for the simulation, authors were able to analyze the stability of the outputs.


## Fundamental Questions

The paper addresses three main research questions. Firstly, what is the reproducibility potential of agent based financial market simulation executed in the research of Raberto, 2005? Secondly, what are the similarities and differences in the outputs of the simulations? Is improvement possible? 


## Expected Results

The anticipated result before starting our research was a recreateion of the simulation executed by (Raberto 2005).


## References 

(Pérez, I. (n.d.). High Frequency Trading II: Limit Order Book. Retrieved from https://www.quantstart.com/articles/high-frequency-trading-ii-limit-order-book

Raberto, M., & Cincotti, S. (2005). Modeling and simulation of a double auction artificial financial market. Physica A: Statistical Mechanics and Its Applications, 355(1), 34–45. https://doi.org/10.1016/j.physa.2005.02.061

Raberto, M., Cincotti, S., Focardi, S. M., & Marchesi, M. (2001). Agent-based simulation of a ÿnancial market, 299, 319–327.)



## Research Methods

( Agent-Based Model)


## Other


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
