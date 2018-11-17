import datetime
import parameters
from enum import Enum
import pandas

class State:
    def __init__(self):
        self.time = 0
        self.date = datetime.datetime(2018, 1, 1)
        self.battery_charge = 0
        self.net_gain = 0
        self.curr_load = 0
        self.__get_next_load = get_system_load()
        self.curr_energy_gen = 0
        self.__get_next_energy = get_energy_generated()

    def get_next_energy_gen(self):
        self.curr_energy_gen = self.__get_next_energy()

    def get_next_system_load(self):
        self.curr_load = self.__get_next_load
    
    def increment_time(self):
        self.time += parameters.TIME_STEP

class Action(Enum):
    charge = 1
    discharge = 2

def get_energy_generated():
    df = pandas.read_csv(parameters.SOLAR_GENERATION_FILE_LOCATION)
    index = 0

    def get_energy():
        nonlocal index
        energy_generated = df[index, parameters.ENERGY_COL_INDEX]
        index += 1
        return energy_generated
    
    return get_energy

def get_system_load():
    df = pandas.read_csv(parameters.HOME_ENERGY_FILE_LOCATION)
    index = 0

    def get_load():
        nonlocal index
        energy_generated = df[index, parameters.LOAD_COL_INDEX]
        index += 1
        return energy_generated
    
    return get_load

def get_battery_wear(delta_energy):
    return delta_energy**2 # temp function until Select sends the real one.

def get_reward(State): # Finish this on Monday.
    return 0

def simulator():
    get_energy = get_energy_generated()
    get_load = get_system_load()

    def time_step(state, action):
        state.increment_time()
        cur_energy = get_energy()
        cur_load = get_load()
        if (action == Action.charge):
            x=1
            # Charge the battery
            # Check if we need to buy to charge the battery
                # Decrement net_gain
        else:
            x = 1
            # Discharge the battery
            # Check if we need to sell any energy
                # Increment net_gain

    return time_step

if __name__ == "__main__":
    cur_state = State()
    cur_action = Action.charge
    my_simulator = simulator()
    print(cur_state.time)
    my_simulator(cur_state, cur_action)
    print(cur_state.time)
