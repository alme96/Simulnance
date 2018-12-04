from mesa import Agent, Model
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
order_lim = []


def price_path(order_list):
    i = 0
    mid_pr = []
    interp = []
    for iterate in order_list:
        temp = sum(iterate[0])/2
        mid_pr.append([temp, iterate[1]])
        if i % 60 == 0:
            interp.append([temp, iterate[1]])
        i += 1
    return mid_pr, interp


def log_ret(m_price):
    log_r = []
    for i in range(1, len(m_price)):
        temp = math.log(m_price[i][0]) - math.log(m_price[i-1][0])
        log_r.append([temp, m_price[i][1]])
    return log_r


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
        self.agent_list = []
        self.order_limits = order_lim
        for i in range(num_a):
            a = TradingAgent(i, self)
            self.agent_list.append(a)

    def get_limit_price(self):
        if len(self.limit_order_book[0]) == 0:
            get_ask = self.last_buy#sell
        else:
            get_ask = min(self.limit_order_book[0])[0]
        if len(self.limit_order_book[1]) == 0:
            get_bid = self.last_sell#buy
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
        t_range = random.randint(600, 6000)
        t_lim = min(t_range, self.clock)
        samples = self.order_limits
        end = len(samples)
        relevant_samples = samples[end - t_lim:end - 1]
        not_relevant, price_sample = price_path(relevant_samples)
        log_ret_sample_t = log_ret(price_sample)
        log_ret_sample = [value[0] for value in log_ret_sample_t]
        sigma = np.std(log_ret_sample)
        sigma_r = 4.25 * float(sigma)  # 4.25
        if sigma_r > 0.01:
            return 0.01
        if sigma_r < 0.001:
            return 0.001
        return sigma_r


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
            ask_price = self.model.last_buy#sell
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
            bid_price = self.model.last_sell#buy
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


# The limit_order_book is a list with 2 entries, which again are lists. The first one stores the sell_order's
# the second one stores the buy_order's.

init_l_o_b = [[], []]
simple_model = TradingModel(init_l_o_b)


for day in range(m_days):
    for second in range(time_steps_per_day):
        simple_model.step()
        simple_model.order_limits.append((simple_model.get_limit_price(), simple_model.clock))
    simple_model.limit_order_book = [[], []]


mid_price, interpolate = price_path(simple_model.order_limits)
log_return = log_ret(interpolate)

x_val = [select_x[1] for select_x in log_return]
y_val = [select_y[0] for select_y in log_return]
plt.bar(x_val, y_val, width=100)
#plt.plot(x_val, y_val)
plt.savefig('log_returns_a.png')
plt.close()

# plt.bar(range(len(interpolate)), interpolate)
x_val_price = [select_x[1] for select_x in interpolate]
y_val_price = [select_y[0] for select_y in interpolate]
plt.plot(x_val_price, y_val_price)
plt.axis([x_val_price[0], x_val_price[len(x_val_price)-1], 80, 120])
plt.savefig('price_path_a.png')
plt.close()

auto_corr = np.correlate(y_val, y_val, mode="full")
half = math.floor(len(auto_corr)/2)
auto_corr_p = auto_corr[half:]
plt.bar(range(len(auto_corr_p)), auto_corr_p)
#plt.plot(range(len(auto_corr_p)), auto_corr_p)
plt.savefig('correlation_a.png')
plt.close()

abs_val = [abs(number) for number in y_val]
auto_corr_abs = np.correlate(abs_val, abs_val, mode="full")
half_abs = math.floor(len(auto_corr_abs)/2)
auto_corr_abs_p = auto_corr_abs[half_abs:]
plt.bar(range(len(auto_corr_abs_p)), auto_corr_abs_p)
#plt.plot(range(len(auto_corr_abs_p)), auto_corr_abs_p)
plt.savefig('correlation_abs_a.png')
plt.close()
