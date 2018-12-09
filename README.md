
# Modeling and Simulation of Social Systems Fall 2018 – Research Plan (Template)


> * Group Name: Simulnance
> * Group participants names: Kedir Firehiwot Nesro, Menichelli Alessandro, Wüst Vincent
> * Project Title: Trade!t
> * Programming language: Python

## General Introduction

The primary motivation of the paper is to mimic a real financial market using agents based simulation. Secondary motivation is to identify and indicate possible impediments in simulating the market.States your motivation clearly: Although progress is made to simulate the real life behavior of Financial price behavior, due to the complexity of the system, there is still a gap to be filled in representing the financial market.

## The Model

In the model Price of a stock is a dependent variable and the behaviour of traders is an independent variable that corresponds to the structure of the model. The variables are measured through agents, time and stock parameters. The amount of Agents are set, the amount of days and daily section in which the model is applied is set by authors . In addtion, Intial stock amount is given to agent and money creation is not allowed in the model.) (The model addresed key parameters used in the research of (Raberto & Cincotti, 2005).Through using multiple scenarios for the simulation, authors were able to analyze the stability of the outputs.


## Fundamental Questions

The paper addresses three main research questions. Firstly, what is the reproducibility potential of agent based financial market simulation executed in the research of (Raberto & Cincotti, 2005)? Secondly, what are the similarities and differences in the outputs of the simulations? Is improvement possible? 


## Expected Results

The anticipated result before starting our research was a recreateion of the simulation executed by (Raberto & Cincotti, 2005).


## References 

(Pérez, I. (n.d.). High Frequency Trading II: Limit Order Book. Retrieved from https://www.quantstart.com/articles/high-frequency-trading-ii-limit-order-book

Raberto, M., & Cincotti, S. (2005). Modeling and simulation of a double auction artificial financial market. Physica A: Statistical Mechanics and Its Applications, 355(1), 34–45. https://doi.org/10.1016/j.physa.2005.02.061

Raberto, M., Cincotti, S., Focardi, S. M., & Marchesi, M. (2001). Agent-based simulation of a ÿnancial market, 299, 319–327.)



## Research Methods

Agent-Based Modelling

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
![some text](https://github.com/alme96/Simulnance/blob/master/small_std.png)
![some more text](https://github.com/alme96/Simulnance/blob/master/large_std.png)

## Full test
The results that we came up with are the visualizations of the estimated price path of the stock and its log returns.
Since we implemented two models we derived all the results two times. Therefore, this test will be split into two parts.
The first one (the simple model) considers the simpler model where the decision making of the traders isn't influenced by price volatility.
The second one (the advanced model) will consider a dependency between the decision making of traders and price volatility.
Next we talk about the implementation of the simple model which is consistent with the file simple_rep.

### The simple model
First some packages must be installed. The mesa package and the matplotlib package aren't installed by default.
```
from mesa import Agent, Model  
import random  
import matplotlib.pyplot as plt  
import numpy as np  
import math  
from scipy import stats  
```
Initialization of some global variables.
Apart from m_days they have the same values as in the paper which we replicate.
We chose a smaller number of days to decrease the computational effort when doing the plots. As we did the Jarque-Bera test we made a run without plots. There we increased `m_days` to 1000, which corresponds to the value of the paper.
```
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
```
The function `trade_amount(a_val, b_val)` determines a random amount of shares which both traders can afford.
```
def trade_amount(a_val, b_val):
    if a_val <= 0 or b_val <= 0:  # To assure that the input of randint corresponds to a valid interval
        return 0
    temp0 = min(a_val, b_val)
    temp1 = random.randint(0, temp0)
    return temp1
```
The function `next_order()` returns the random time interval between consecutive order placements.
```
def next_order():
    return math.ceil(np.random.exponential(avg_order_waiting_time, None))
```
Next the function `price_path(order_list)` is introduced. The price of the stock is calculated using the mid price.
This is done once for every time step and once for every 60th time step
which corresponds to previous tick interpolation of the actual price path (which doesn't change every time step)
with a time window of 60 seconds.
```
def price_path(order_list):
    mid_pr = []
    for iterate in order_list:
        temp = sum(iterate[0])/2
        mid_pr.append([temp, iterate[1]])
    return mid_pr
```
Regarding the function `log_ret(m_price)`. The log returns are deduced form the price path by taking the log of the current price
and subtracting the log of the last price from it.
```
def log_ret(m_price):
    log_r = []
    for i in range(1, len(m_price)):
        temp = math.log(m_price[i][0]) - math.log(m_price[i-1][0])
        log_r.append([temp, m_price[i][1]])
    return log_r
```
The class TradingModel presented below inherits from Model and determines the structure of our model.
Various functions have been implemented.

On behalf of `__init__(self, l_o_b)`. Here all the agents are created.  
Furthermore we need:  
* limit_order_book; to store all the unresolved sell and buy orders.  
* clock; as a reference value for the orders which are cancelled after a while and the order waiting times.  
* order_arrival; as a fixed time step where the next order gets placed  
* last_sell & last_buy; to circumvent an empty limit_order_book  
* agent_list; to store all the agents  

More on behalf of the limit_order_book:  
The limit_order_book is a list with 2 entries, which again are lists. The first one stores the sell order
the second one stores the buy order. Each sell resp. buy order is a tuple with 3 entries. The first entry
corresponds to the price per share offered in the order. The second one corresponds to the time step
when the order gets canceled. The third one corresponds to the unique_id of the agent how placed the order.

The function `get_limit_price(self) ` returns the lowest sell order and the highest buy order.
If the limit_order_book has no sell order resp. no buy order,
the last ask price resp. bid price corresponding to the last_buy resp. last_sell is used.

The function `refresh_lob(self)` iterates through all the elements of the sell order list and  the buy order list and cancels all
orders, which have been in the limit_order_book for more than 600 seconds.

Regarding the function `step(self)`. If the order waiting time has passed, one agent gets randomly chosen to place an order.
Before the selected agent does his step,
the limit_order_book needs to be refreshed and a new order waiting time must be calculated.
To trace the current step, at the end of the step the clock must be updated.

The function `trading_partner(self, key)` returns the agent whose unique_id corresponds to key.
```
class TradingModel(Model):
    def __init__(self, l_o_b):
        super().__init__()
        self.limit_order_book = l_o_b
        self.clock = 0
        self.order_arrival = self.clock + next_order()
        self.last_sell = init_stock_price
        self.last_buy = init_stock_price
        self.agent_list = []
        for i in range(num_a):
            a = TradingAgent(i, self)
            self.agent_list.append(a)

    def get_limit_price(self):
        if len(self.limit_order_book[0]) == 0:
            get_ask = self.last_buy
        else:
            get_ask = min(self.limit_order_book[0])[0]  # minimum with respect to first element of tuples
        if len(self.limit_order_book[1]) == 0:
            get_bid = self.last_sell
        else:
            get_bid = max(self.limit_order_book[1])[0]  # maximum with respect to first element of tuples
        return get_ask, get_bid

    def refresh_lob(self):
        for sell_tuple in self.limit_order_book[0]:
            if sell_tuple[1] < self.clock:
                index_t = self.limit_order_book[0].index(sell_tuple)
                del(self.limit_order_book[0][index_t])
        for buy_tuple in self.limit_order_book[1]:
            if buy_tuple[1] < self.clock:
                index_t = self.limit_order_book[1].index(buy_tuple)
                del(self.limit_order_book[1][index_t])

    def step(self):
        if self.clock == self.order_arrival:
            self.refresh_lob()
            self.order_arrival = self.clock + next_order()
            active_agent = random.choice(self.agent_list)
            active_agent.step()
        self.clock += 1

    def trading_partner(self, key):
        for agent in self.agent_list:
            if key == agent.unique_id:
                return agent
```

Next the class TradingAgent is presented. This class inherits from Agent. Objects from this class only get called by the model.
This class determines how traders i.e. the agents place their orders and how the trading is structured.

In the function `__init__(self, unique_id, model)` the traders get initialized. Every agents i.e. trader has the same amount of shares and cash in the beginning.

Regarding the `step(self)` function. If an agent gets selected to commit a step he either places a sell order or a buy order.
Both equally likely.

Next we look at the `sell_order(self)` function. To form a sell order, the trader i.e. the agent needs to now the current ask_price.
If there are no sell orders in the limit_order_book from which the ask_price could be deduced,
The last ask_price available is chosen, which corresponds to last_buy.
The final sell order (s_order) is normally distributed around the ask_price.
If the s_order is above the bid_price or there aren't any buy orders in the limit_order_book,
s_order gets stored in the limit_order_book,
together with the time when it gets cancelled and the unique_id of the agent (in this order).
Otherwise the current trader will proceed at the bid_price with the trader
that formed the order of the bid_price. A random amount of shares, affordable for both trading partners
gets selected and traded at the bid_price.
After that, the order at the bid_price from the limit_order_book gets canceled.

Similar for the `buy_order(self)` function. To form a buy order, the trader i.e. the agent needs to now the current bid_price.
If there are no buy orders in the limit_order_book from which the bid_price could be deduced,
The last bid_price available is chosen, which corresponds to last_sell.
The final buy order (b_order) is normally distributed around the bid_price.
If the b_order is below the ask_price or there aren't any sell orders in the limit_order_book,
b_order gets stored in the limit_order_book,
together with the time when it gets cancelled and the unique_id of the agent (in this order).
Otherwise the current trader will proceed at the ask_price with the trader
that formed the order of the ask_price. A random amount of shares, affordable for both trading partners
gets selected and traded at the ask_price.
After that, the order at the ask_price from the limit_order_book gets canceled.
```
class TradingAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cash = init_cash_a
        self.shares = init_shares_a

    def step(self):
        coin = round(np.random.uniform(0, 1, None))
        if coin < 0.5:
            self.buy_order()
        else:
            self.sell_order()

    def sell_order(self):
        if len(self.model.limit_order_book[0]) == 0:
            ask_price = self.model.last_buy
        else:
            (ask_price, a_deadline, a_id) = min(self.model.limit_order_book[0])
        s_order_temp = ask_price * (np.random.normal(mean, std, None))  # Offer creation
        s_order = round(s_order_temp, 2)
        if len(self.model.limit_order_book[1]) == 0:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[0].append((s_order, order_deadline, self.unique_id))
            return
        (bid_price, b_deadline, b_id) = max(self.model.limit_order_book[1])
        if s_order > bid_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[0].append((s_order, order_deadline, self.unique_id))
        else:
            self.model.last_sell = bid_price
            buyer = self.model.trading_partner(b_id)
            n_max_sell = self.shares
            n_max_buy = math.floor(buyer.cash / bid_price)
            n_trade = trade_amount(n_max_sell, n_max_buy)
            self.cash = self.cash + bid_price * n_trade
            self.shares = self.shares - n_trade
            buyer.cash = buyer.cash - bid_price * n_trade
            buyer.shares = buyer.shares + n_trade
            index_t = self.model.limit_order_book[1].index((bid_price, b_deadline, b_id))
            del(self.model.limit_order_book[1][index_t])

    def buy_order(self):
        if len(self.model.limit_order_book[1]) == 0:
            bid_price = self.model.last_sell
        else:
            (bid_price, b_deadline, b_id) = max(self.model.limit_order_book[1])
        b_order_temp = bid_price * (np.random.normal(mean, std, None))  # Offer creation
        b_order = round(b_order_temp, 2)
        if len(self.model.limit_order_book[0]) == 0:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[1].append((b_order, order_deadline, self.unique_id))
            return
        (ask_price, a_deadline, a_id) = min(self.model.limit_order_book[0])
        if b_order < ask_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[1].append((b_order, order_deadline, self.unique_id))
        else:
            self.model.last_buy = ask_price
            seller = self.model.trading_partner(a_id)
            n_max_buy = math.floor(self.cash / ask_price)
            n_max_sell = seller.shares
            n_trade = trade_amount(n_max_sell, n_max_buy)
            seller.cash = seller.cash + ask_price * n_trade
            seller.shares = seller.shares - n_trade
            self.cash = self.cash - ask_price * n_trade
            self.shares = self.shares + n_trade
            index_t = self.model.limit_order_book[0].index((ask_price, a_deadline, a_id))
            del(self.model.limit_order_book[0][index_t])
```

In the fallowing we create a model and let it run for m_days consisting of time_steps_per_day
To tract the current ask and bid price in every time step,
we append the values to a list i.e. order_lim in every time step.
To stay consistent with the paper which we replicate by parts, we clear the limit_order_book after every trading day.
```
init_l_o_b = [[], []]
simple_model = TradingModel(init_l_o_b)
order_lim = []

for day in range(m_days):
    for second in range(time_steps_per_day):
        simple_model.step()
        if simple_model.clock % 60 == 0:
            order_lim.append((simple_model.get_limit_price(), simple_model.clock))
    simple_model.limit_order_book = [[], []]
```
Next we calculate the price path and the log returns.
```
interpolate = price_path(order_lim)
log_return = log_ret(interpolate)
y_val = [samples[0] for samples in log_return]
```
With the samples `y_val` given, we can calculate the Jarque-Bera test.
```
test_statistic, p_value = stats.jarque_bera(y_val)
if p_value < 0.005:
    print("The null hypotheses of normal distribution for the log returns is rejected.")
else:
    print("The null hypotheses of normal distribution for the log returns cannot be rejected.")
```
Below the log returns are plotted. The bars need a large width to be visible.
```
x_val = [select_x[1] for select_x in log_return]
y_val = [select_y[0] for select_y in log_return]
plt.bar(x_val, y_val, width=100)
plt.show()
```
As a last step we also plotted the price path.
```
x_val_price = [select_x[1] for select_x in interpolate]
y_val_price = [select_y[0] for select_y in interpolate]
plt.plot(x_val_price, y_val_price)
plt.axis([x_val_price[0], x_val_price[len(x_val_price)-1], 80, 120])
plt.show()
```

Next we talk about the implementation of the advanced model which is consistent with the file advanced_rep.
The main diffrence to the simple model is the use of the function `standard_dev(self)` which is a method of the advanced model and therefore introduced in the chapter below.

### The advanced model
First some packages must be installed. The mesa package and the matplotlib package aren't installed by default.
```
from mesa import Agent, Model  
import random  
import matplotlib.pyplot as plt  
import numpy as np  
import math  
from scipy import stats  
```
Initialization of some global variables.
Apart from m_days they have the same values as in the paper which we replicate.
We chose a smaller number of days to decrease the computational effort when doing the plots. As we did the Jarque-Bera test we made a run without plots. There we increased `m_days` to 1000, which corresponds to the value of the paper.
```
m_days = 10
time_steps_per_day = 25200
life_span = 600
avg_order_waiting_time = 20
num_a = 10000
init_stock_price = 100
init_cash_a = 100000
init_shares_a = 1000
mean = 1
order_lim = []
```
Next the function `price_path(order_list)` is considered. The price of the stock is calculated using the mid price.
This is done once for every time step and once for every 60th time step
which corresponds to previous tick interpolation of the actual price path (which doesn't change every time step)
with a time window of 60 seconds.
```
def price_path(order_list):
    mid_pr = []
    for iterate in order_list:
        temp = sum(iterate[0])/2
        mid_pr.append([temp, iterate[1]])
    return mid_pr
```
Regarding the `log_ret(m_price)`. The log returns are deduced form the price path by taking the log of the current price
and subtracting the log of the last price from it.
```
def log_ret(m_price):
    log_r = []
    for i in range(1, len(m_price)):
        temp = math.log(m_price[i][0]) - math.log(m_price[i-1][0])
        log_r.append([temp, m_price[i][1]])
    return log_r
```
The function `trade_amount(a_val, b_val)` determines a random amount of shares
which both traders can afford.
```
def trade_amount(a_val, b_val):
    if a_val <= 0 or b_val <= 0:  # To assure that the input of randint corresponds to a valid interval
        return 0
    temp0 = min(a_val, b_val)
    temp1 = random.randint(0, temp0)
    return temp1
```
The function `next_order()` returns the random time interval between consecutive order placements.
```
def next_order():
    """This function returns the random time interval between consecutive order placements."""
    return math.ceil(np.random.exponential(avg_order_waiting_time, None))
```
Next, the class TradingModel is introduced. This class inherits from Model and determines the structure of our model.
Various functions have been implemented in this model.

On behalf of `__init__(self, l_o_b)`. Here all the agents are created.
Furthermore we need:
* limit_order_book; to store all the unresolved sell and buy orders.
* clock; as a reference value for the orders which are cancelled after a while and the order waiting times.
* order_arrival; as a fixed time step where the next order gets placed
* last_sell & last_buy; to circumvent an empty limit_order_book
* agent_list; to store all the agents
* order_limits; to tract the current ask and bid price in every time step

More on behalf of the limit_order_book:
The limit_order_book is a list with 2 entries, which again are lists. The first one stores the sell order
the second one stores the buy order. Each sell resp. buy order is a tuple with 3 entries. The first entry
corresponds to the price per share offered in the order. The second one corresponds to the time step
when the order gets canceled. The third one corresponds to the unique_id of the agent how placed the order.

The function `get_limit_price(self)` returns the lowest sell order and the highest buy order.
If the limit_order_book has no sell order resp. no buy order,
the last ask price resp. bid price corresponding to the last_buy resp. last_sell is used.

The function `refresh_lob(self)` Iterates through all the elements of the sell order list and  the buy order list and cancels all
orders, which have been in the limit_order_book for more than 600 seconds.

The function `step(self)` is constructed as fallows. If the order waiting time has passed, one agent gets randomly chosen to place an order. Before the selected agent does his step, the limit_order_book needs to be refreshed and a new order waiting time must be calculated. To trace the current step, at the end of the step the clock must be updated.

The function `trading_partner(self, key)` returns the agent whose unique_id corresponds to key.

Next we disguss how the function `standard_dev(self)` is implemented. Here the standard deviation is formed with respect to the log returns.
First a random amount of the most recent samples of the bid and ask prices
are retained from the order_limits list.
Then those samples are used to calculate the price path and finally the log return samples.
The log return samples are needed to calculate the sample standard deviation.
Besides some exceptional problems which must be considered, we decided to use bounds for the resulting
standard deviation.
```
class TradingModel(Model):
    def __init__(self, l_o_b):
        super().__init__()
        self.limit_order_book = l_o_b
        self.clock = 0
        self.order_arrival = self.clock + next_order()  # time when the next order is executed
        self.last_sell = init_stock_price
        self.last_buy = init_stock_price
        self.agent_list = []
        self.order_limits = order_lim
        for i in range(num_a):
            a = TradingAgent(i, self)
            self.agent_list.append(a)

    def get_limit_price(self):
        if len(self.limit_order_book[0]) == 0:
            get_ask = self.last_buy
        else:
            get_ask = min(self.limit_order_book[0])[0]
        if len(self.limit_order_book[1]) == 0:
            get_bid = self.last_sell
        else:
            get_bid = max(self.limit_order_book[1])[0]
        return get_ask, get_bid

    def refresh_lob(self):
        for sell_tuple in self.limit_order_book[0]:
            if sell_tuple[1] < self.clock:
                index_t = self.limit_order_book[0].index(sell_tuple)
                del(self.limit_order_book[0][index_t])
        for buy_tuple in self.limit_order_book[1]:
            if buy_tuple[1] < self.clock:
                index_t = self.limit_order_book[1].index(buy_tuple)
                del(self.limit_order_book[1][index_t])

    def step(self):
        if self.clock == self.order_arrival:
            self.refresh_lob()
            self.order_arrival = self.clock + next_order()
            active_agent = random.choice(self.agent_list)
            active_agent.step()
        self.clock += 1

    def trading_partner(self, key):
        for agent in self.agent_list:
            if key == agent.unique_id:
                return agent

    def standard_dev(self):
        if self.clock < 600:    # Without this offset the std will always be zero. It makes sense to wait 600 seconds
            return 0.005        # because 600 seconds is the smallest window that anyone would consider.
        t_range = random.randint(10, 100)  # Considering the last 600-6000 time steps corresponds to looking at the last
        samples = self.order_limits  # 10-100 samples.
        t_lim = min(t_range, len(samples))  # within time step 600 and 6000 t_range could be bigger than clock
        end = len(samples)
        relevant_samples = samples[end - t_lim:end - 1]
        price_sample = price_path(relevant_samples)
        log_ret_sample_t = log_ret(price_sample)
        log_ret_sample = [value[0] for value in log_ret_sample_t]
        sigma = np.std(log_ret_sample)
        sigma_r = 4.25 * float(sigma)
        if sigma_r > 0.01:
            return 0.01
        if sigma_r < 0.001:
            return 0.001
        return sigma_r
```
Next the class TradingAgent is introduced.
This class inherits from Agent. Objects from this class only get called by the model.
This class determines how traders i.e. the agents place their orders and how the trading is structured.

On behalf of the function `__init__(self, unique_id, model)`. Every agents i.e. trader has the same amount of shares and cash in the beginning.

The function `step(self)` is constructed as fallows. If an agent gets selected to commit a step he either places a sell order or a buy order. Both equally likely.

Next the function `sell_order(self)` is considered. To form a sell order, the trader i.e. the agent needs to now the current ask_price.
If there are no sell orders in the limit_order_book from which the ask_price could be deduced,
The last ask_price available is chosen, which corresponds to last_buy.
The final sell order (s_order) is normally distributed around the ask_price.
If the s_order is above the bid_price or there aren't any buy orders in the limit_order_book,
s_order gets stored in the limit_order_book,
together with the time when it gets cancelled and the unique_id of the agent (in this order).
Otherwise the current trader will proceed at the bid_price with the trader
that formed the order of the bid_price. A random amount of shares, affordable for both trading partners
gets selected and traded at the bid_price.
After that, the order at the bid_price from the limit_order_book gets canceled.

The function `buy_order(self)` is constructed similarly. To form a buy order, the trader i.e. the agent needs to now the current bid_price.
If there are no buy orders in the limit_order_book from which the bid_price could be deduced,
The last bid_price available is chosen, which corresponds to last_sell.
The final buy order (b_order) is normally distributed around the bid_price.
If the b_order is below the ask_price or there aren't any sell orders in the limit_order_book,
b_order gets stored in the limit_order_book,
together with the time when it gets cancelled and the unique_id of the agent (in this order).
Otherwise the current trader will proceed at the ask_price with the trader
that formed the order of the ask_price. A random amount of shares, affordable for both trading partners
gets selected and traded at the ask_price.
After that, the order at the ask_price from the limit_order_book gets canceled.

```class TradingAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cash = init_cash_a
        self.shares = init_shares_a

    def step(self):
        coin = round(np.random.uniform(0, 1, None))
        if coin < 0.5:
            self.buy_order()
        else:
            self.sell_order()

    def sell_order(self):
        if len(self.model.limit_order_book[0]) == 0:
            ask_price = self.model.last_buy
        else:
            (ask_price, a_deadline, a_id) = min(self.model.limit_order_book[0])
        std = self.model.standard_dev()
        s_order_temp = ask_price * (np.random.normal(mean, std, None))  # Offer creation
        s_order = round(s_order_temp, 2)
        if len(self.model.limit_order_book[1]) == 0:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[0].append((s_order, order_deadline, self.unique_id))
            return
        (bid_price, b_deadline, b_id) = max(self.model.limit_order_book[1])
        if s_order > bid_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[0].append((s_order, order_deadline, self.unique_id))
        else:
            self.model.last_sell = bid_price
            buyer = self.model.trading_partner(b_id)
            n_max_sell = self.shares
            n_max_buy = math.floor(buyer.cash / bid_price)
            n_trade = trade_amount(n_max_sell, n_max_buy)
            self.cash = self.cash + bid_price * n_trade
            self.shares = self.shares - n_trade
            buyer.cash = buyer.cash - bid_price * n_trade
            buyer.shares = buyer.shares + n_trade
            index_t = self.model.limit_order_book[1].index((bid_price, b_deadline, b_id))
            del(self.model.limit_order_book[1][index_t])

    def buy_order(self):
        if len(self.model.limit_order_book[1]) == 0:
            bid_price = self.model.last_sell
        else:
            (bid_price, b_deadline, b_id) = max(self.model.limit_order_book[1])
        std = self.model.standard_dev()
        b_order_temp = bid_price * (np.random.normal(mean, std, None))  # Offer creation
        b_order = round(b_order_temp, 2)
        if len(self.model.limit_order_book[0]) == 0:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[1].append((b_order, order_deadline, self.unique_id))
            return
        (ask_price, a_deadline, a_id) = min(self.model.limit_order_book[0])
        if b_order < ask_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[1].append((b_order, order_deadline, self.unique_id))
        else:
            self.model.last_buy = ask_price
            seller = self.model.trading_partner(a_id)
            n_max_buy = math.floor(self.cash / ask_price)
            n_max_sell = seller.shares
            n_trade = trade_amount(n_max_sell, n_max_buy)
            seller.cash = seller.cash + ask_price * n_trade
            seller.shares = seller.shares - n_trade
            self.cash = self.cash - ask_price * n_trade
            self.shares = self.shares + n_trade
            index_t = self.model.limit_order_book[0].index((ask_price, a_deadline, a_id))
            del(self.model.limit_order_book[0][index_t])
```

In the fallowing we create a model and let it run for m_days consisting of time_steps_per_day
To tract the current ask and bid price in every time step,
we append the values to a list i.e. order_limits, which is an attribute of the model, in every time step.
To stay consistent with the paper which we replicate by parts, we clear the limit_order_book after every trading day.

```
init_l_o_b = [[], []]
advanced_model = TradingModel(init_l_o_b)

for day in range(m_days):
    for second in range(time_steps_per_day):
        advanced_model.step()
        if advanced_model.clock % 60 == 0:
            advanced_model.order_limits.append((advanced_model.get_limit_price(), advanced_model.clock))
    advanced_model.limit_order_book = [[], []]
```
Next the estimated price path and its log returns are calculated.
```
interpolate = price_path(advanced_model.order_limits)
log_return = log_ret(interpolate)
y_val = [samples[0] for samples in log_return]
```
With the samples `y_val` given, we can calculate the Jarque-Bera test.
```
test_statistic, p_value = stats.jarque_bera(y_val)
if p_value < 0.005:
    print("The null hypotheses of normal distribution for the log returns is rejected.")
else:
    print("The null hypotheses of normal distribution for the log returns cannot be rejected.")
```
Below we plot the log returns. The bars need a large width to be visible.
```
x_val = [select_x[1] for select_x in log_return]
y_val = [select_y[0] for select_y in log_return]
plt.bar(x_val, y_val, width=100)
plt.show()
```
As a last step we plot the estimated price path.
```
x_val_price = [select_x[1] for select_x in interpolate]
y_val_price = [select_y[0] for select_y in interpolate]
plt.plot(x_val_price, y_val_price)
plt.axis([x_val_price[0], x_val_price[len(x_val_price)-1], 80, 120])
plt.show()
```
