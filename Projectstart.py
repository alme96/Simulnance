# To get this model running, some moduls have to be integrated first.
# Mesa and matplotlib must be installed first.
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import numpy as np
import math


# Initialize some global variables

m_days = 1
time_steps_per_day = 25
life_span = 6
avg_order_waiting_time = 2
num_a = 10
init_stock_price = 10000
init_cash_a = 100000
init_shares_a = 1000
mean = 1
std = 0.005


def next_order():
    return math.ceil(np.random.exponential(avg_order_waiting_time, size=1)[0])


class TradingModel(Model):

    def __init__(self, l_o_b, time):
        super().__init__()
        self.limit_order_book = l_o_b
        self.clock = time
        self.order_arrival = next_order()
        self.schedule = RandomActivation(self)
        for i in range(num_a):
            a = TradingAgent(i, self)
            self.schedule.add(a)

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
            self.order_arrival = self.order_arrival + next_order()
            active_agent = random.choice(self.schedule.agents)
            active_agent.step()
        self.clock += 1


class TradingAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.cash = init_cash_a
        self.shares = init_shares_a

    def step(self):
        coin = round(np.random.uniform(0, 1, 1)[0])
        if coin < 0.5:
            self.buy_order()
        else:
            self.sell_order()

    def sell_order(self):
        (ask_price, a_deadline) = min(self.model.limit_order_book[0])  # Ev. problematic to always calculate min/max.
        (bid_price, b_deadline) = max(self.model.limit_order_book[1])  # maybe easier to have an ordered l_o_b!
        s_order = round(ask_price * (np.random.normal(mean, std, 1)[0]))
        if s_order > bid_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[0].append((s_order, order_deadline))
        else:
            index_t = self.model.limit_order_book[1].index((bid_price, b_deadline))
            del(self.model.limit_order_book[1][index_t])

    def buy_order(self):
        (ask_price, a_deadline) = min(self.model.limit_order_book[0])
        (bid_price, b_deadline) = max(self.model.limit_order_book[1])
        b_order = round(bid_price * (np.random.normal(mean, std, 1)[0]))
        if b_order < ask_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[1].append((b_order, order_deadline))
        else:
            index_t = self.model.limit_order_book[0].index((ask_price, a_deadline))
            del(self.model.limit_order_book[0][index_t])


# The limit_order_book is a list with 2 entries, which again are lists. The first one stores the sell_order's
# the second one stores the buy_order's.
# for the initialization of the limit_order_book we should include one very high sell_order and one very low
# buy order such that the limit order book never is empty
init_l_o_b = [[(init_stock_price * 100, time_steps_per_day * m_days), (init_stock_price, life_span)],
              [(0, time_steps_per_day * m_days), (init_stock_price, life_span)]]
first_try = TradingModel(init_l_o_b, 0)
for j in range(time_steps_per_day):
    first_try.step()
print(first_try.limit_order_book)


# Yet missing: - The transaction of money and shares
#              - Generalization to daily sections
#              - A Data collector
