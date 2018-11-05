
# To get this model running, some moduls have to be integrated first.
# Mesa and matplotlib must be installed first.
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt
import numpy as np


# 2 subclasses will be defined next. MoneyModel inherits from Model, MoneyAgent inherits from Agent.

class MoneyModel(Model):
    """A model with some number of agents."""
    def __init__(self, n, width, height):
        super().__init__()  # ev take out
        self.num_agents = n
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        """Create agents"""
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

            """Add agents to a random grid cell"""
            x = random.randrange(self.grid.width)
            y = random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.datacollector = DataCollector(model_reporters={"Gini": MoneyAgent.compute_gini}, agent_reporters={"Wealth": "wealth"})

    def step(self):
        """Advance the model by one step"""
        self.datacollector.collect(self)
        self.schedule.step()


class MoneyAgent(Agent):

    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    def compute_gini(model):
        agent_wealths = [agent.wealth for agent in model.schedule.agents]
        x = sorted(agent_wealths)
        n_agents = model.num_agents
        b = sum(xi * (n_agents-i) for i, xi in enumerate(x))/(n_agents*sum(x))
        return 1 + 1/n_agents - 2*b

    def step(self):
        """The agents step will go here."""
        self.move()
        if self.wealth > 0:
            self.give_money()

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

        # print(self.unique_id)


third_model = MoneyModel(50, 10, 10)
for index in range(100):
    third_model.step()

gini = third_model.datacollector.get_model_vars_dataframe()
gini.plot()
plt.show()

agents_wealth = third_model.datacollector.get_agent_vars_dataframe()
agents_wealth.head()
print(agents_wealth)

agent_counts = np.zeros((third_model.grid.width, third_model.grid.height))
for cell in third_model.grid.coord_iter():
    cell_content, px, py = cell
    agent_counts[px][py] = len(cell_content)

plt.imshow(agent_counts, interpolation='nearest')
plt.colorbar()
plt.show()

