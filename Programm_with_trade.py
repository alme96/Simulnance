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

m_days = 5
time_steps_per_day = 40
life_span = 10
avg_order_waiting_time = 1
num_a = 10
init_stock_price = 1000
init_cash_a = 100000
init_shares_a = 1000
mean = 1
std = 0.010


def trade_amount(a, b):
    if a <= b:
        return random.randint(0, a)
    else:
        return random.randint(0, b)


def next_order():  # function that produces time intervall between orders
    return math.ceil(np.random.exponential(avg_order_waiting_time, size=1)[0])


class TradingModel(Model):

    def __init__(self, l_o_b, time):
        super().__init__()
        self.limit_order_book = l_o_b
        self.clock = time
        self.order_arrival = self.clock + next_order()  # time when the next order is executed
        self.last_sell = init_stock_price
        self.last_buy = init_stock_price
        self.schedule = RandomActivation(
            self)  # subclass from mesa with the function "add" and the list of agents schedule.agents
        self.price_book = []
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
                del (self.limit_order_book[0][index_t])
        for buy_tuple in self.limit_order_book[1]:
            if buy_tuple[1] < self.clock:
                index_t = self.limit_order_book[1].index(buy_tuple)
                del (self.limit_order_book[1][index_t])

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
        coin = round(np.random.uniform(0, 1, 1)[0])
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
            print("No trade at time", self.model.clock, ":")
            print(self.model.limit_order_book)
            print()
            return
        (bid_price, b_deadline, b_id) = max(self.model.limit_order_book[1])
        if s_order > bid_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[0].append((s_order, order_deadline, self.unique_id))
            print("No trade at time", self.model.clock, ":")
            print(self.model.limit_order_book)
            print()
        else:
            self.model.last_sell = bid_price
            self.model.price_book.append(bid_price)
            buyer = self.model.trading_partner(b_id)
            N_max_sell = self.shares
            N_max_buy = int(buyer.cash / bid_price)
            N_trade = trade_amount(N_max_sell, N_max_buy)
            self.cash = self.cash + bid_price * N_trade
            self.shares = self.shares - N_trade
            buyer.cash = buyer.cash - bid_price * N_trade
            buyer.shares = buyer.shares + N_trade
            print("Trade at time", self.model.clock, ":")
            print("Seller", self.unique_id, "meets Buyer", buyer.unique_id, "and sells", N_trade, "shares at",
                  bid_price, "each.")
            print("Agent", self.unique_id, " now has", self.cash, " amount of cash and ", self.shares,
                  "amount of shares.")
            print("Agent", buyer.unique_id, "now has", buyer.cash, "amount of cash and", buyer.shares,
                  "amount of shares.")
            print(self.model.limit_order_book)
            print()
            index_t = self.model.limit_order_book[1].index((bid_price, b_deadline, b_id))
            del (self.model.limit_order_book[1][index_t])

    def buy_order(self):
        if len(self.model.limit_order_book[1]) == 0:
            bid_price = self.model.last_buy
        else:
            (bid_price, b_deadline, b_id) = max(self.model.limit_order_book[1])
        b_order = round(bid_price * (np.random.normal(mean, std, None)))  # Offer creation
        if len(self.model.limit_order_book[0]) == 0:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[1].append((b_order, order_deadline, self.unique_id))
            print("No trade at time", self.model.clock, ":")
            print(self.model.limit_order_book)
            print()
            return
        (ask_price, a_deadline, a_id) = min(self.model.limit_order_book[0])
        if b_order < ask_price:
            order_deadline = self.model.clock + life_span
            self.model.limit_order_book[1].append((b_order, order_deadline, self.unique_id))
            print("No trade at time", self.model.clock, ":")
            print(self.model.limit_order_book)
            print()
        else:
            self.model.last_buy = ask_price
            self.model.price_book.append(ask_price)
            seller = self.model.trading_partner(a_id)
            N_max_buy = int(self.cash / ask_price)
            N_max_sell = seller.shares
            N_trade = trade_amount(N_max_sell, N_max_buy)
            seller.cash = self.cash + ask_price * N_trade
            seller.shares = self.shares - N_trade
            self.cash = self.cash - ask_price * N_trade
            self.shares = self.shares + N_trade
            print("Trade at time", self.model.clock, ":")
            print("Buyer", self.unique_id, "meets Seller", seller.unique_id, "and buys", N_trade, "shares at",
                  ask_price, "each.")
            print("Agent", self.unique_id, "now has", self.cash, "amount of cash and", self.shares, "amount of shares.")
            print("Agent", seller.unique_id, "now has", seller.cash, "amount of cash and", seller.shares,
                  "amount of shares.")
            print(self.model.limit_order_book)
            print()
            index_t = self.model.limit_order_book[0].index((ask_price, a_deadline, a_id))
            del (self.model.limit_order_book[0][index_t])


# The limit_order_book is a list with 2 entries, which again are lists. The first one stores the sell_order's
# the second one stores the buy_order's.
# for the initialization of the limit_order_book we should include one very high sell_order and one very low
# buy order such that the limit order book never is empty
# init_l_o_b = [[(init_stock_price * 100, time_steps_per_day * m_days), (init_stock_price, life_span)],
#               [(0, time_steps_per_day * m_days), (init_stock_price, life_span)]]
init_l_o_b = [[], []]
first_try = TradingModel(init_l_o_b, 0)
# for j in range(time_steps_per_day):
#     first_try.step()
# print()
# print("Final limit order book:")
# print(first_try.limit_order_book)

# Yet missing: - The transaction of money and shares
#              - Generalization to daily sections
#              - A Data collector

# Going from a single day to multiple:
#
for day in range(m_days):
    for j in range(time_steps_per_day):
        first_try.step()
    print()
    print("Limit Order Book after ", day + 1, " days:")
    print(first_try.limit_order_book, "\n")
    first_try.limit_order_book = [[], []]

limits = first_try.data_collector_1.get_model_vars_dataframe()
print("The ask_price and the bid_price at every step:")
print(limits, "\n")
print("The history of price returns:")
print(first_try.price_book)
