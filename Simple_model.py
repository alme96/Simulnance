from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import numpy as np
import math

# Initialize some global variables

m_days = 20  # should be at 1000. increasing this value will lead to an increased processing time. 300 works (within 1h)
time_steps_per_day = 25200
life_span = 600
avg_order_waiting_time = 20
num_a = 10000
init_stock_price = 100
init_cash_a = 100000
init_shares_a = 1000
mean = 1
std = 0.005  # 0.005 probably too small, 0.01 already works.


def trade_amount(a_val, b_val):
    if a_val <= 0 or b_val <= 0:
        return 0
    temp0 = min(a_val, b_val)
    temp1 = random.randint(0, temp0)
    return temp1


def next_order():  # function that produces time interval between orders
    return math.ceil(np.random.exponential(avg_order_waiting_time, None))


class TradingModel(Model):

    def __init__(self, l_o_b):
        super().__init__()
        self.limit_order_book = l_o_b
        self.clock = 0
        self.order_arrival = self.clock + next_order()  # time when the next order is executed
        self.last_sell = init_stock_price
        self.last_buy = init_stock_price
        self.schedule = RandomActivation(
            self)  # subclass from mesa with the function "add" and the list of agents schedule.agents
        for i in range(num_a):
            a = TradingAgent(i, self)
            self.schedule.add(a)
        self.data_collector_1 = DataCollector(
            model_reporters={"Limits": TradingModel.get_limit_price}
        )

    def get_limit_price(self):
        if len(self.limit_order_book[0]) == 0:
            get_ask = self.last_sell
        else:
            get_ask = min(self.limit_order_book[0])[0]
        if len(self.limit_order_book[1]) == 0:
            get_bid = self.last_buy
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
        self.data_collector_1.collect(self)
        if self.clock == self.order_arrival:
            self.refresh_lob()
            self.order_arrival = self.clock + next_order()
            active_agent = random.choice(self.schedule.agents)
            active_agent.step()
        self.clock += 1

    def trading_partner(self, key):
        for agent in self.schedule.agents:
            if key == agent.unique_id:
                return agent


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
            ask_price = self.model.last_sell
        else:
            (ask_price, a_deadline, a_id) = min(self.model.limit_order_book[0])
        s_order = round(ask_price * (np.random.normal(mean, std, None)))  # Offer creation
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
            bid_price = self.model.last_buy
        else:
            (bid_price, b_deadline, b_id) = max(self.model.limit_order_book[1])
        b_order = round(bid_price * (np.random.normal(mean, std, None)))  # Offer creation
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


# The limit_order_book is a list with 2 entries, which again are lists. The first one stores the sell_order's
# the second one stores the buy_order's.

init_l_o_b = [[], []]
simple_model = TradingModel(init_l_o_b)

for day in range(m_days):
    for second in range(time_steps_per_day):
        simple_model.step()
    simple_model.limit_order_book = [[], []]

limits = simple_model.data_collector_1.get_model_vars_dataframe()
order_limits = limits.as_matrix()


def price_path(order_list):
    i = 0
    mid_pr = []
    interp = []
    for iterate in order_list:
        temp = sum(iterate[0])/2
        mid_pr.append(temp)
        if i % 60 == 0:
            interp.append(temp)
        i += 1
    return mid_pr, interp


def log_ret(m_price):
    log_r = []
    for i in range(1, len(m_price)):
        temp = math.log(m_price[i]) - math.log(m_price[i-1])
        log_r.append(temp)
    return log_r


mid_price, interpolate = price_path(order_limits)
log_return = log_ret(interpolate)

plt.bar(range(len(log_return)), log_return)
plt.plot(range(len(log_return)), log_return)
plt.savefig('log_returns.png')
plt.close()

# plt.bar(range(len(interpolate)), interpolate)
plt.plot(range(len(interpolate)), interpolate)
plt.axis([10, len(interpolate)-1, 80, 120])
plt.savefig('price_path.png')
plt.close()

auto_corr = np.correlate(log_return, log_return, mode="full")
half = math.floor(len(auto_corr)/2)
auto_corr_p = auto_corr[half:]
plt.bar(range(len(auto_corr_p)), auto_corr_p)
plt.plot(range(len(auto_corr_p)), auto_corr_p)
plt.savefig('correlation.png')
plt.close()

abs_val = [abs(number) for number in log_return]
auto_corr_abs = np.correlate(abs_val, abs_val, mode="full")
half_abs = math.floor(len(auto_corr_abs)/2)
auto_corr_abs_p = auto_corr_abs[half_abs:]
plt.bar(range(len(auto_corr_abs_p)), auto_corr_abs_p)
plt.plot(range(len(auto_corr_abs_p)), auto_corr_abs_p)
plt.savefig('correlation_abs.png')
plt.close()

# Assumptions been made, that don't fallow directly from the paper:

#   - The orders of the traders are price per share. The actual amount of shares and money traded
#   only depends on the assets which buyer and seller own (when a trade occurs, we just take a random fraction of that).

#   - Normally the traders consider ask resp. bid prices when they want to place a sell resp. buy order.
#   If the Limit Order Book is empty the traders consider the last successful sell resp. buy order.


# Results:
#
#   - We are interested in log price returns. Whit price the price of the stock is meant.
#   Since our orders only depend on price per share offers we have to calculate the stock price by the mid price.
#
#   - Inhomogeneous time steps because offers are exponentially distributed. But with previous tick interpolation
#   this is equivalent to just consider every 60th time step.
