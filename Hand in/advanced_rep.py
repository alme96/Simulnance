"""
This module describes an experiment witch models a financial market.
This module differs from the module simple in the way that traders construct their orders.
In both modules the traders place their selling order resp. buying order normally distributed
around the current ask price resp. bid price. Whereas in this module the standard deviation
of the normal distribution is formed with respect to the log returns from the past,
in the module simple the standard deviation corresponds to a constant.
"""

# The mesa package and the matploblib package are not installed by default.
from mesa import Agent, Model
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats

# Initialize some global variables
# Apart from m_days they have the same values as in the paper which we replicate.
# We chose a smaller number of days to decrease the computational effort.
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


def price_path(order_list):
    """
    The price of the stock is calculated using the mid price.
    This is done once for every time step and once for every 60th time step
    which corresponds to previous tick interpolation of the actual price path (which doesn't change every time step)
    with a time window of 60 seconds.
    """
    mid_pr = []
    for iterate in order_list:
        temp = sum(iterate[0])/2
        mid_pr.append([temp, iterate[1]])
    return mid_pr


def log_ret(m_price):
    """
    The log returns are deduced form the price path by taking the log of the current price
    and subtracting the log of the last price from it.
    """
    log_r = []
    for i in range(1, len(m_price)):
        temp = math.log(m_price[i][0]) - math.log(m_price[i-1][0])
        log_r.append([temp, m_price[i][1]])
    return log_r


def trade_amount(a_val, b_val):
    """
    This function determines a random amount of shares
    which both traders can afford
    """
    if a_val <= 0 or b_val <= 0:  # To assure that the input of randint corresponds to a valid interval
        return 0
    temp0 = min(a_val, b_val)
    temp1 = random.randint(0, temp0)
    return temp1


def next_order():
    """This function returns the random time interval between consecutive order placements."""
    return math.ceil(np.random.exponential(avg_order_waiting_time, None))


class TradingModel(Model):
    """This class inherits from Model and determines the structure of our model."""
    def __init__(self, l_o_b):
        """
        Initialisation of the model. Here all the agents are created.
        Furthermore we need:
            - limit_order_book; to store all the unresolved sell and buy orders.
            - clock; as a reference value for the orders which are cancelled after a while and the order waiting times.
            - order_arrival; as a fixed time step where the next order gets placed
            - last_sell & last_buy; to circumvent an empty limit_order_book
            - agent_list; to store all the agents
            - order_limits; to tract the current ask and bid price in every time step
        More on behalf of the limit_order_book:
        The limit_order_book is a list with 2 entries, which again are lists. The first one stores the sell order
        the second one stores the buy order. Each sell resp. buy order is a tuple with 3 entries. The first entry
        corresponds to the price per share offered in the order. The second one corresponds to the time step
        when the order gets canceled. The third one corresponds to the unique_id of the agent how placed the order.
        """
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
        """
        Returns the lowest sell order and the highest buy order.
        If the limit_order_book has no sell order resp. no buy order,
        the last ask price resp. bid price corresponding to the last_buy resp. last_sell is used.
        """
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
        """
        Iterates through all the elements of the sell order list and  the buy order list and cancels all
        orders, which have been in the limit_order_book for more than 600 seconds.
        """
        for sell_tuple in self.limit_order_book[0]:
            if sell_tuple[1] < self.clock:
                index_t = self.limit_order_book[0].index(sell_tuple)
                del(self.limit_order_book[0][index_t])
        for buy_tuple in self.limit_order_book[1]:
            if buy_tuple[1] < self.clock:
                index_t = self.limit_order_book[1].index(buy_tuple)
                del(self.limit_order_book[1][index_t])

    def step(self):
        """
        If the order waiting time has passed, one agent gets randomly chosen to place an order.
        Before the selected agent does his step,
        the limit_order_book needs to be refreshed and a new order waiting time must be calculated.
        To trace the current step, at the end of the step the clock must be updated.
        """
        if self.clock == self.order_arrival:
            self.refresh_lob()
            self.order_arrival = self.clock + next_order()
            active_agent = random.choice(self.agent_list)
            active_agent.step()
        self.clock += 1

    def trading_partner(self, key):
        """Returns the agent whose unique_id corresponds to key."""
        for agent in self.agent_list:
            if key == agent.unique_id:
                return agent

    def standard_dev(self):
        """
        Here the standard deviation is formed with respect to the log returns.
        First a random amount of the most recent samples of the bid and ask prices
        are retained from the order_limits list.
        Then those samples are used to calculate the price path and finally the log return samples.
        The log return samples are needed to calculate the sample standard deviation.
        Besides some exceptional problems which must be considered, we decided to use bounds for the resulting
        standard deviation.
        """
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


class TradingAgent(Agent):
    """
    This class inherits from Agent. Objects from this class only get called by the model.
    This class determines how traders i.e. the agents place their orders and how the trading is structured.
    """
    def __init__(self, unique_id, model):
        """Every agents i.e. trader has the same amount of shares and cash in the beginning."""
        super().__init__(unique_id, model)
        self.cash = init_cash_a
        self.shares = init_shares_a

    def step(self):
        """
        If an agent gets selected to commit a step he either places a sell order or a buy order.
        Both equally likely.
        """
        coin = round(np.random.uniform(0, 1, None))
        if coin < 0.5:
            self.buy_order()
        else:
            self.sell_order()

    def sell_order(self):
        """
        To form a sell order, the trader i.e. the agent needs to now the current ask_price.
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
        """
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
        """
        To form a buy order, the trader i.e. the agent needs to now the current bid_price.
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
        """
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


# In the fallowing we create a model and let it run for m_days consisting of time_steps_per_day
# To tract the current ask and bid price in every time step,
# we append the values to a list i.e. order_limits, which is an attribute of the model, in every time step.
# To stay consistent with the paper which we replicate, we clear the limit_order_book after every trading day.

init_l_o_b = [[], []]
advanced_model = TradingModel(init_l_o_b)


for day in range(m_days):
    for second in range(time_steps_per_day):
        advanced_model.step()
        if advanced_model.clock % 60 == 0:
            advanced_model.order_limits.append((advanced_model.get_limit_price(), advanced_model.clock))
    advanced_model.limit_order_book = [[], []]


# Calculation of the price path and the log returns.
interpolate = price_path(advanced_model.order_limits)
log_return = log_ret(interpolate)
y_val = [samples[0] for samples in log_return]

test_statistic, p_value = stats.jarque_bera(y_val)
if p_value < 0.005:
    print("The null hypotheses of normal distribution for the log returns is rejected.")
else:
    print("The null hypotheses of normal distribution for the log returns cannot be rejected.")

# Plotting the log returns. The bars need a large width to be visible.
x_val = [select_x[1] for select_x in log_return]
y_val = [select_y[0] for select_y in log_return]
plt.bar(x_val, y_val, width=100)
plt.show()

# Plotting the price path.
x_val_price = [select_x[1] for select_x in interpolate]
y_val_price = [select_y[0] for select_y in interpolate]
plt.plot(x_val_price, y_val_price)
plt.axis([x_val_price[0], x_val_price[len(x_val_price)-1], 80, 120])
plt.show()
