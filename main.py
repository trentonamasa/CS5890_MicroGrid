import datetime
import parameters
from enum import Enum
import pandas
import math
import matplotlib.pyplot as plt

BATTERY_SCALAR = parameters.MAX_BATTERY_CAPACITY / parameters.NUM_BATTERY_CAPACITY_BINS

class Buy_Sell_Amount:
    def __init__(self):
        self.amount_to_buy = 0
        self.amount_to_sell = 0

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
        self.get_next_energy_gen()
        self.get_next_system_load()

    def get_next_energy_gen(self):
        self.cur_energy_gen = self.__get_next_energy(self.time)

    def get_next_system_load(self, row = None):
        self.cur_load = self.__get_next_load(row)

    def get_difference_battery_level(self, delta_energy):
        buy_sell = Buy_Sell_Amount()
        total_energy = self.battery_charge + delta_energy
        if total_energy > parameters.MAX_BATTERY_CAPACITY:
            buy_sell.amount_to_sell = total_energy - parameters.MAX_BATTERY_CAPACITY
        elif total_energy < 0:
            buy_sell.amount_to_buy = abs(total_energy)
        return buy_sell

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

class plot_info:
    def __init__(self):
        self.time = []
        self.battery_charges = []
        self.loads = []
        self.gains = []
        self.energy_gens = []

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
    return -abs((delta_energy/BATTERY_SCALAR)**2) # TODO Replace this function with the real one from Select

def get_reward(state, action):
    buy_sell = state.get_difference_battery_level(action)
    return gain_changer(state, action) + get_battery_wear(action - buy_sell.amount_to_buy - buy_sell.amount_to_sell)

def gain_changer(state, action):
    buy_sell = Buy_Sell_Amount()
    energy_difference = -action - state.cur_load + state.cur_energy_gen
    if energy_difference < 0:
        buy_sell.amount_to_buy = abs(energy_difference)
    else:
        buy_sell.amount_to_sell = abs(energy_difference)

    if state.time > parameters.PEAK_TIME_BEGIN and state.time < parameters.PEAK_TIME_END:
        return buy_sell.amount_to_buy * parameters.PEAK_TIME_COST + buy_sell.amount_to_sell * parameters.PEAK_TIME_SELL
    else:
        return buy_sell.amount_to_buy * parameters.COST + buy_sell.amount_to_sell * parameters.SELL

def arg_max(state, v_table):
    cur_battery_level = int(state.battery_charge / BATTERY_SCALAR)
    best = float("-inf")
    for delta_battery_level in range(-cur_battery_level, parameters.NUM_BATTERY_CAPACITY_BINS - cur_battery_level):
        cur_score = get_reward(state, delta_battery_level * BATTERY_SCALAR) + v_table[(state.time + 1) % parameters.NUM_TIME_STEP_BINS][cur_battery_level + delta_battery_level]
        if best < cur_score:
            best = cur_score
            action = delta_battery_level
        
    return action * BATTERY_SCALAR

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
        #print("Battery Level Bins:",battery_level * BATTERY_SCALAR)
        v_table[parameters.NUM_TIME_STEP_BINS - 1][battery_level] = get_reward(state, -battery_level * BATTERY_SCALAR)
    
    # Fill v_table
    delta = float("inf")
    while delta > parameters.MIN_ACCEPTABLE_DELTA:
        delta = 0
        for cur_time_bin in range(parameters.NUM_TIME_STEP_BINS - 2, -1 , -1):
            # Update State for current time
            state.time = cur_time_bin * parameters.TIME_STEP
            state.get_next_system_load(cur_time_bin)
            state.get_next_energy_gen()

            # Loop through all battery states
            for cur_battery_level in range(parameters.NUM_BATTERY_CAPACITY_BINS):
                state.battery_charge = cur_battery_level * BATTERY_SCALAR
                v = v_table[cur_time_bin][cur_battery_level]

                # Loop through all possible actions (empty battery to charge fully)
                best = float("-inf")
                for delta_battery_level in range(-cur_battery_level, parameters.NUM_BATTERY_CAPACITY_BINS - cur_battery_level):
                    #print("\nBatt:", cur_battery_level, "Action:", delta_battery_level, "Reward:", get_reward(state, delta_battery_level * BATTERY_SCALAR), "Next State Value:", v_table[cur_time_bin + 1][cur_battery_level + delta_battery_level])
                    best = max(best, get_reward(state, delta_battery_level * BATTERY_SCALAR) + v_table[cur_time_bin + 1][cur_battery_level + delta_battery_level])
                delta = max(delta, abs(v - best))
                v_table[cur_time_bin][cur_battery_level] = best
    
    return v_table

def simulate_time_step(state, action):
    state.net_gain += get_reward(state, action)
    state.change_battery_level(action)

    # Increment max_load
    # Decrement net_gain
    state.increment_time()
    state.get_next_energy_gen()
    state.get_next_system_load()

def print_v_table(v_table):
    print("--- V_Table ---")
    for i, time_bin in enumerate(v_table):
        print(i)
        print(time_bin)
    print("-------------")

def get_action_for_select_function(state):
    if state.cur_load >= parameters.MAX_ACCEPTABLE_LOAD_FOR_SELECT:
        action = -min(state.cur_load - parameters.MAX_ACCEPTABLE_LOAD_FOR_SELECT, state.battery_charge)
    elif (state.time < 23 or state.time < 5) and state.battery_charge < parameters.MAX_BATTERY_CAPACITY:
        action = (parameters.MAX_BATTERY_CAPACITY - state.battery_charge) / 6
    else:
        action = 0
    return action

def plot_results(info, title):
    plt.figure(1)
    plt.title(title)
    plt.subplot(211)
    plt.plot(info.time, info.energy_gens, label = 'Energy gen')
    plt.plot(info.time, info.loads, label = 'Load')
    plt.ylabel("Watts")
    plt.xlabel("Hours")
    plt.legend()
    plt.plot(info.time, info.battery_charges, label = 'Battery charge')

    plt.subplot(212)
    plt.plot(info.time, info.gains, label = 'Net gain/loss')
    plt.ylabel("Net gain/loss ($)")
    plt.xlabel("Hours")
    plt.legend()
    plt.show()  

def run_machine_learning():
    cur_state = State(True)
    cur_action = 3
    num_days_to_simulate = 3 * parameters.NUM_TIME_STEP_BINS
    v_table = initialize_v_table()

    graph_info = plot_info()

    graph_info.time.append(cur_state.time)
    graph_info.energy_gens.append(cur_state.cur_energy_gen)
    graph_info.loads.append(cur_state.cur_load)
    graph_info.battery_charges.append(cur_state.battery_charge)
    graph_info.gains.append(cur_state.net_gain)
    # print("Time: ",cur_state.time,"  Energy: ",cur_state.cur_energy_gen,"   Load: ", cur_state.cur_load,"  Charge in Battery: ", cur_state.battery_charge)
    
    for i in range(num_days_to_simulate):
        #print("\nTime:", parameters.TIME_STEP * i, "  cur_time:", cur_state.time, "  Net Score:", cur_state.net_gain)
        #print("Load:", cur_state.cur_load, "  Energy Gen:", cur_state.cur_energy_gen, "  Batt:", cur_state.battery_charge)
        graph_info.time.append(parameters.TIME_STEP * i)
        graph_info.energy_gens.append(cur_state.cur_energy_gen)
        graph_info.loads.append(cur_state.cur_load)
        graph_info.battery_charges.append(cur_state.battery_charge)
        graph_info.gains.append(cur_state.net_gain)

        cur_action = arg_max(cur_state, v_table)
        #print("Action:", cur_action)
        simulate_time_step(cur_state, cur_action)
   
    # plot_results(graph_info, 'Machine Learning')

def run_select_style():
    cur_state = State(False)
    cur_action = 3
    num_days_to_simulate = 3*parameters.NUM_TIME_STEP_BINS

    graph_info = plot_info()

    graph_info.time.append(cur_state.time)
    graph_info.energy_gens.append(cur_state.cur_energy_gen)
    graph_info.loads.append(cur_state.cur_load)
    graph_info.battery_charges.append(cur_state.battery_charge)
    graph_info.gains.append(cur_state.net_gain)
    # print("Time: ",cur_state.time,"  Energy: ",cur_state.cur_energy_gen,"   Load: ", cur_state.cur_load,"  Charge in Battery: ", cur_state.battery_charge)
    
    for i in range(num_days_to_simulate):
        #print("\nTime:", parameters.TIME_STEP * i, "  cur_time:", cur_state.time, "  Net Score:", cur_state.net_gain)
        #print("Load:", cur_state.cur_load, "  Energy Gen:", cur_state.cur_energy_gen, "  Batt:", cur_state.battery_charge)
        graph_info.time.append(parameters.TIME_STEP * i)
        graph_info.energy_gens.append(cur_state.cur_energy_gen)
        graph_info.loads.append(cur_state.cur_load)
        graph_info.battery_charges.append(cur_state.battery_charge)
        graph_info.gains.append(cur_state.net_gain)

        cur_action = get_action_for_select_function(cur_state)
        #print("Action:", cur_action)
        simulate_time_step(cur_state, cur_action)
   
    plot_results(graph_info, 'SELECT Function')    

if __name__ == "__main__":
    run_machine_learning()
    run_select_style()
   