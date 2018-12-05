import datetime
import parameters
from enum import Enum
import pandas
import math
import matplotlib.pyplot as plt

BATTERY_SCALAR = parameters.MAX_BATTERY_CAPACITY / parameters.NUM_BATTERY_CAPACITY_BINS

class State:
    def __init__(self, is_v_table_initializer = False):
        self.time = 0
        self.date = 0
        self.battery_charge = 0
        self.net_gain = 0
        # self.max_load = 0
        self.cur_load = 0
        self.__get_next_load = get_system_load(is_v_table_initializer)
        self.cur_energy_gen = 0
        self.__get_next_energy = get_energy_generated()
        self.max_battery_capacity = parameters.MAX_BATTERY_CAPACITY

    def get_next_energy_gen(self):
        self.cur_energy_gen = self.__get_next_energy(self.time)

    def get_next_system_load(self, row = None):
        self.cur_load = self.__get_next_load(row)

    def get_difference_battery_level(self, delta_energy):
        flux = 0
        if self.battery_charge + delta_energy > parameters.MAX_BATTERY_CAPACITY:
            flux = self.battery_charge - parameters.MAX_BATTERY_CAPACITY
        elif self.battery_charge + delta_energy < 0:
            flux = self.battery_charge
        return flux

    def change_battery_level(self, delta_energy):
        self.battery_charge += delta_energy
        if self.battery_charge > parameters.MAX_BATTERY_CAPACITY:
            self.battery_charge = parameters.MAX_BATTERY_CAPACITY
        elif self.battery_charge < 0:
            self.battery_charge = 0
    
    def increment_time(self):
        self.time += parameters.TIME_STEP
        if (self.time > parameters.TIME_STEP * parameters.NUM_TIME_STEP_BINS):
            self.time -= parameters.TIME_STEP * parameters.NUM_TIME_STEP_BINS

def get_energy_generated():
    #df = pandas.read_csv(parameters.SOLAR_GENERATION_FILE_LOCATION)
    row = 0

    def get_energy(time):
        begin_solar_gen_time = 11
        end_solar_gen_time = 18
        scalar = 13000
        if (time > begin_solar_gen_time and time < end_solar_gen_time):
            return math.sin((time - begin_solar_gen_time) * math.pi / (end_solar_gen_time - begin_solar_gen_time)) * scalar
        return 0
    '''def get_energy():
        nonlocal row
        energy_generated = df.iloc[row, parameters.ENERGY_COL_INDEX]
        row += 1
        return energy_generated'''
    
    return get_energy

def get_system_load(is_v_table_initializer = False):
    if is_v_table_initializer:
        df = pandas.read_csv(parameters.LOAD_V_TABLE_INITIALIZER_FILE_LOCATION, header=None)
    else:
        df = pandas.read_csv(parameters.HOME_ENERGY_FILE_LOCATION, header=None)
    row = 0

    def get_load(cur_time_bin = None):
        nonlocal row
        if cur_time_bin != None: row = cur_time_bin
        energy_generated = df.iloc[row, parameters.LOAD_COL_INDEX]
        row += 1
        row %= 1000
        return energy_generated
    
    return get_load

def get_battery_wear(delta_energy):
    return -abs(delta_energy**2) # TODO Replace this function with the real one from Select

def get_reward(state, action): # TODO Finish this.
    return gain_changer(state, state.get_difference_battery_level(action)) + get_battery_wear(action - state.get_difference_battery_level(action))

def gain_changer(state, flux):
    flux += state.cur_energy_gen - state.cur_load
    if state.time < parameters.PEAK_TIME_BEGIN and state.time < parameters.PEAK_TIME_END:
        if flux < 0: return abs(flux)*parameters.PEAK_TIME_COST
        else: return abs(flux)*parameters.PEAK_TIME_SELL
    else:
        if flux < 0: return abs(flux)*parameters.COST
        else: return abs(flux)*parameters.SELL

def arg_max(state, v_table):
    for cur_battery_level in range(parameters.NUM_BATTERY_CAPACITY_BINS):
        best = float("-inf")
        for delta_battery_level in range(-cur_battery_level, parameters.NUM_BATTERY_CAPACITY_BINS - cur_battery_level):
            state.battery_charge = (cur_battery_level + delta_battery_level) * BATTERY_SCALAR
            cur_score = get_reward(state, delta_battery_level * BATTERY_SCALAR) + v_table[(state.time + 1) % parameters.NUM_TIME_STEP_BINS][cur_battery_level + delta_battery_level]
            if best < cur_score:
                best = cur_score
                action = delta_battery_level
    
    return action


def initialize_v_table():
    v_table = []
    # Initialize v_table
    for time_step_bin in range(parameters.NUM_TIME_STEP_BINS):
        v_table.append([])
        for battery_bin in range(parameters.NUM_BATTERY_CAPACITY_BINS):
            v_table[time_step_bin].append(0)
    
    is_v_table_initializer = True
    state = State(is_v_table_initializer)
    state.time = parameters.TIME_STEP * (parameters.NUM_TIME_STEP_BINS - 1)
    state.get_next_system_load(parameters.NUM_TIME_STEP_BINS - 1)
    state.get_next_energy_gen()
    state.cur_energy_gen = 0
    state.cur_load = 0
    # Fill in final column of v_table
    for battery_level in range(parameters.NUM_BATTERY_CAPACITY_BINS):
        print("Battery Level Bins:",battery_level * BATTERY_SCALAR)
        v_table[parameters.NUM_TIME_STEP_BINS - 1][battery_level] = gain_changer(state, battery_level * BATTERY_SCALAR)
    print(v_table[parameters.NUM_TIME_STEP_BINS-1])
    
    # Fill v_table
    delta = float("inf")
    while delta > parameters.MIN_ACCEPTABLE_DELTA:
        delta = 0
        for cur_time_bin in range(parameters.NUM_TIME_STEP_BINS - 2, -1 , -1):
            # print("cur_time_bin: ", cur_time_bin)
            state.time = cur_time_bin * parameters.TIME_STEP
            state.get_next_system_load(cur_time_bin)
            state.get_next_energy_gen()
            for cur_battery_level in range(parameters.NUM_BATTERY_CAPACITY_BINS):
                v = v_table[cur_time_bin][cur_battery_level]
                best = float("-inf")
                for delta_battery_level in range(-cur_battery_level, parameters.NUM_BATTERY_CAPACITY_BINS - cur_battery_level):
                    #print("Action:", delta_battery_level, " Battery Level:", cur_battery_level," Max Capacity:", parameters.MAX_BATTERY_CAPACITY)
                    state.battery_charge = (cur_battery_level + delta_battery_level) * BATTERY_SCALAR
                    best = max(best, get_reward(state, delta_battery_level * BATTERY_SCALAR) + v_table[cur_time_bin + 1][cur_battery_level + delta_battery_level])
                #print("v:",v,"   best:",best,"   v - best =",v-best)
                delta = max(delta, abs(v - best))
                # print(delta) # This is much greater than the parameter one, looping through the while loop and getting bigger.
                v_table[cur_time_bin][cur_battery_level] = best
        print(delta)
    
    return v_table

def simulate_time_step(state, action):
    flux = state.get_difference_battery_level(action)
    state.change_battery_level(action)
    state.net_gain += gain_changer(state, flux) + get_battery_wear(action - flux)

    # Increment max_load
    # Decrement net_gain
    state.increment_time()
    state.get_next_energy_gen()
    state.get_next_system_load()

if __name__ == "__main__":
    cur_state = State()
    cur_action = 3
    print("I am printing something")
    v_table = initialize_v_table()
    print('done with v_table')

    times = []
    energy_gens = []
    loads = []
    battery_charges = []
    gains = []

    times.append(cur_state.time)
    energy_gens.append(cur_state.cur_energy_gen)
    loads.append(cur_state.cur_load)
    battery_charges.append(cur_state.battery_charge)
    gains.append(cur_state.net_gain)
    # print("Time: ",cur_state.time,"  Energy: ",cur_state.cur_energy_gen,"   Load: ", cur_state.cur_load,"  Charge in Battery: ", cur_state.battery_charge)
    
    for i in range(72):
        simulate_time_step(cur_state, cur_action)
        cur_action = arg_max(cur_state, v_table)

        times.append(cur_state.time)
        energy_gens.append(cur_state.cur_energy_gen)
        loads.append(cur_state.cur_load)
        battery_charges.append(cur_state.battery_charge)
        gains.append(cur_state.net_gain)
   
    plt.plot(energy_gens, label = 'Energy gen')
    plt.plot(loads, label = 'Load')
    plt.plot(battery_charges, label = 'Battery charge')
    plt.plot(gains, label = 'Net gain/loss')
    plt.ylabel("Watt")
    plt.xlabel("Hours")
    plt.legend()
    plt.show()        
