
# Modeling and Simulation of Social Systems Fall 2018 – Research Plan (Template)


> * Group Name: (Simulnance)
> * Group participants names: (Kedir Firehiwot Nesro, Menichelli Alessandro, Wüst Vincent)
> * Project Title: (Trade!t)
> * Programming language: (Python)

## General Introduction

The primary motivation of the paper is to mimic a real financial market using agents based simulation. Secondary motivation is to identify and indicate possible impediments in simulating the market.States your motivation clearly: Although progress is made to simulate the real life behavior of Financial price behavior, due to the complexity of the system, there is still a gap to be filled in representing the financial market.

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

Agent-Based Modelling


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
The results that we came up with are the visualizations of the estimated price path of the stock and its log returns.
Since we implemented two models we derived all the results two times. Therefore, this test will be split into two parts.
The first one (the simple model) considers the simpler model where the decision making of the traders isn't influenced by price volatility.
The second one (the advanced model) will consider a dependency between the decision making of traders and price volatility.

### The simple model
First some packages must be installed. The mesa package and the matplotlib package aren't installed by default.

from mesa import Agent, Model  
import random  
import matplotlib.pyplot as plt  
import numpy as np  
import math  
from scipy import stats  

Initialization of some global variables.
Apart from m_days they have the same values as in the paper which we replicate.
We chose a smaller number of days to decrease the computational effort when doing the plots. As we did the Jarque-Bera test we made a run without plots. There we increased m_days to 1000, which corresponds to the value of the paper.

m_days = 10  
time_steps_per_day = 25200  
life_span = 600  
avg_order_waiting_time = 20  
num_a = 10000  
init_stock_price = 100  
init_cash_a = 100000  
init_shares_a = 1000  
mean = 1  
std = 0.005  

The fallowing function determines a random amount of shares which both traders can afford.

def trade_amount(a_val, b_val):
    if a_val <= 0 or b_val <= 0:  # To assure that the input of randint corresponds to a valid interval
        return 0
    temp0 = min(a_val, b_val)
    temp1 = random.randint(0, temp0)
    return temp1
