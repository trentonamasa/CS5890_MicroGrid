import datetime
import parameters
from enum import Enum
import pandas
import math

class State:
    def __init__(self):
        self.time = 0
        self.date = 0
        self.battery_charge = 0
        self.net_gain = 0
        # self.max_load = 0
        self.cur_load = 0
        self.__get_next_load = get_system_load()
        self.cur_energy_gen = 0
        self.__get_next_energy = get_energy_generated()
        self.max_battery_capacity = parameters.MAX_BATTERY_CAPACITY

    def get_next_energy_gen(self):
        self.cur_energy_gen = self.__get_next_energy(self.time)

    def get_next_system_load(self):
        self.cur_load = self.__get_next_load()
    
    def increment_time(self):
        seconds_in_a_day = 86400
        self.time += parameters.TIME_STEP
        if (self.time > seconds_in_a_day):
            self.time -= seconds_in_a_day

def get_energy_generated():
    #df = pandas.read_csv(parameters.SOLAR_GENERATION_FILE_LOCATION)
    row = 0

    def get_energy(time):
        begin_solar_gen_time = 39600
        end_solar_gen_time = 64800
        scalar = 900
        if (time > begin_solar_gen_time and time < end_solar_gen_time):
            return math.sin((time - begin_solar_gen_time) * math.pi / (end_solar_gen_time - begin_solar_gen_time)) * scalar
        return 0
    '''def get_energy():
        nonlocal row
        energy_generated = df.iloc[row, parameters.ENERGY_COL_INDEX]
        row += 1
        return energy_generated'''
    
    return get_energy

def get_system_load():
    df = pandas.read_csv(parameters.HOME_ENERGY_FILE_LOCATION, header=None)
    row = 0

    def get_load():
        nonlocal row
        energy_generated = df.iloc[row, parameters.LOAD_COL_INDEX]
        row += 1
        row %= 1000
        return energy_generated
    
    return get_load
def get_battery_wear(delta_energy):
    return delta_energy**2 # temp function until Select sends the real one.

def get_reward(State): # Finish this on Monday.
    return 0

def initialize_v_table():
    v_table = []
    # Initialize v_table
    for time_step_bin in range(parameters.NUM_TIME_STEP_BINS):
        v_table.append([])
        for battery_bin in range(parameters.NUM_BATTERY_CAPACITY_BINS):
            v_table[time_step_bin].append(0)
    
    state = State()
    state.time = parameters.TIME_STEP * (parameters.NUM_TIME_STEP_BINS - 1)
    state.cur_load = 50 # TODO not sure what to put here so I just put 50... probably should look into this a bit more. Maybe there's someway we can use the get_next_system_load?
    state.get_next_energy_gen
    # Fill in final column of v_table
    for battery_level in range(parameters.NUM_BATTERY_CAPACITY_BINS):
        v_table[parameters.NUM_TIME_STEP_BINS - 1][battery_level] = get_reward(state)
    
    # Fill v_table
    delta = float("inf")
    while delta > parameters.MIN_ACCEPTABLE_DELTA:
        delta = 0
        state.time = 0
        for cur_time in range(parameters.NUM_TIME_STEP_BINS - 1, -1 , -1):
            # update state.cur_load; maybe using get_next_system_load?
            state.get_next_energy_gen
            for cur_battery_level in range(parameters.NUM_BATTERY_CAPACITY_BINS):
                v = v_table[cur_time][cur_battery_level]
                best = float("-inf")
                for delta_battery_level in range(-cur_battery_level, parameters.MAX_BATTERY_CAPACITY - cur_battery_level):
                    state.battery_charge = cur_battery_level + delta_battery_level
                    best = max(best, get_reward(state) + v_table[cur_time + 1][cur_battery_level + delta_battery_level])
                delta = max(delta, abs(v - best))
                v_table[cur_time][cur_battery_level] = best
    
    return v_table

def simulate_time_step(state, action):
    if (action > 0):
        x=1
        # Charge the battery
        state.battery_charge += state.cur_energy_gen
        # Check if we need to buy to charge the battery
            # Increment max_load
            # Decrement net_gain
        # else if we have excess
            # Increment net_gain
    else:
        x = 1        
        # Discharge the battery
        state.battery_charge -= state.cur_load
        # Check if we need to sell any energy
            # Increment net_gain
        # else if we need to buy energy
            # Update max_load
            # Decrement net_gain
    state.increment_time()
    state.get_next_energy_gen()
    state.get_next_system_load()

if __name__ == "__main__":
    cur_state = State()
    cur_action = 3
    print("Time: ",cur_state.time,"  Energy: ",cur_state.cur_energy_gen,"   Load: ", cur_state.cur_load,"  Charge in Battery: ", cur_state.battery_charge)
    for i in range(5760):
        simulate_time_step(cur_state, cur_action)
        if (cur_state.cur_load > 0):
            print("Time: ",cur_state.time,"  Energy: ",cur_state.cur_energy_gen,"   Load: ",cur_state.cur_load,"  Charge in Battery: ", cur_state.battery_charge)
