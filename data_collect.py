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
time_steps_per_day = 4000
life_span = 10
avg_order_waiting_time = 1
num_a = 10
init_stock_price = 100
init_cash_a = 100000
init_shares_a = 1000
mean = 1
std = 0.010


def trade_amount(a_val, b_val):
    if a_val <= 0 or b_val <= 0:
        return 0
    temp0 = min(a_val, b_val)
    temp1 = random.randint(0, temp0)
    return temp1


def next_order():  # function that produces time interval between orders
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
            n_max_sell = self.shares
            n_max_buy = math.floor(buyer.cash / bid_price)
            n_trade = trade_amount(n_max_sell, n_max_buy)
            self.cash = self.cash + bid_price * n_trade
            self.shares = self.shares - n_trade
            buyer.cash = buyer.cash - bid_price * n_trade
            buyer.shares = buyer.shares + n_trade
            print("Trade at time", self.model.clock, ":")
            print("Seller", self.unique_id, "meets Buyer", buyer.unique_id, "and sells", n_trade, "shares at",
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
            n_max_buy = math.floor(self.cash / ask_price)#3
            n_max_sell = seller.shares
            n_trade = trade_amount(n_max_sell, n_max_buy)
            seller.cash = seller.cash + ask_price * n_trade
            seller.shares = seller.shares - n_trade
            self.cash = self.cash - ask_price * n_trade
            self.shares = self.shares + n_trade
            print("Trade at time", self.model.clock, ":")
            print("Buyer", self.unique_id, "meets Seller", seller.unique_id, "and buys", n_trade, "shares at",
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
# print(first_try.price_book)
order_limits = limits.as_matrix()


def price_path(order_list):
    i = 0
    mid_price = []
    interpolate = []
    for iterate in order_list:
        temp = sum(iterate[0])/2
        mid_price.append(temp)
        if i % 60 == 0:
            interpolate.append(temp)
        i += 1
    return mid_price, interpolate


def log_ret(m_price):
    log_r = []
    for i in range(1, len(m_price)):
        temp = math.log(m_price[i]) - math.log(m_price[i-1])
        log_r.append(temp)
    return log_r


# def pdf(x_val):
#     n_val = len(x_val)
#     x_round = [round(elem, 2) for elem in x_val]
#     x_new = sorted(x_round)
#     print(x_val, "\n\n")
#     print(x_new)
#     output_y = [1/n_val]
#     output_x = [x_new[0]]
#     for element in x_new:
#         index = len(output_y)
#         if element == output_x[index - 1]:
#             output_y[index - 1] = output_y[index - 1] + 1/n_val
#         else:
#             output_y.append(1/n_val)
#             output_x.append(element)
#     output_y[0] = output_y[0] - 1/n_val
#     print(output_x)
#     return output_x, output_y


mid_price, interpolate = price_path(order_limits)
log_return = log_ret(mid_price)

plt.bar(range(len(log_return)), log_return)
plt.plot(range(len(log_return)), log_return)
plt.show()

plt.bar(range(len(mid_price)), mid_price)
plt.plot(range(len(mid_price)), mid_price)
plt.show()

auto_corr = np.correlate(log_return, log_return, mode="full")
half = math.floor(len(auto_corr)/2)
auto_corr_p = auto_corr[half:]
plt.bar(range(len(auto_corr_p)), auto_corr_p)
plt.plot(range(len(auto_corr_p)), auto_corr_p)
plt.show()

abs_val = [abs(number) for number in log_return]
auto_corr_abs = np.correlate(abs_val, abs_val, mode="full")
half_abs = math.floor(len(auto_corr_abs)/2)
auto_corr_abs_p = auto_corr_abs[half_abs:]
plt.bar(range(len(auto_corr_abs_p)), auto_corr_abs_p)
plt.plot(range(len(auto_corr_abs_p)), auto_corr_abs_p)
plt.show()

# distr_x, distr_y = pdf(log_return)
# plt.bar(distr_x, distr_y)
# plt.semilogy(distr_x, distr_y)
# plt.show()

# Assumptions been made, that don't fallow directly from the paper:

#   - The orders of the traders are price per share. The actual amount of shares and money traded
#   only depends on the assets which buyer and seller own (when a trade occurs, we just take a random fraction of that).

#   - Normally the traders consider ask resp. bid prices when they want to place a sell resp. buy order.
#   If the Limit Order Book is empty the traders consider the last successful sell resp. buy order.


# Results:
#
#   - We are intrested in log price returns. Whit price the price of the stock is meanr.
#   Since our orders only depend on price per share offers we have to calculate the stockprice by the midprice.
#
#   - Inhomogeneous timesteps because offers are exponentially distr. But with previous tick interpolation
#   this is equivallent to just consider every 60th timestep.

# Don't need:
#   - price book
