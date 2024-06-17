############### 动作空间缩小为36 ###############
path = './utils/'
import random
import numpy as np
from matplotlib import pyplot as plt
import gym
import gym.spaces as spaces
import math

# Default parameters

# if not point out day apply DEFALT_DAY
DEFAULT_DAY0 = 3
DEFAULT_DAYN = 1
# Power generated in the microgrid
DEFAULT_POWER_GENERATED = np.genfromtxt(path + "wind_generation_fortum.csv", delimiter=',', skip_header=0,
                                        usecols=[-1]) / 100
DEFAULT_WIND_POWER_COST = 3.2
# Balancing market prices
DEFAULT_DOWN_REG = np.genfromtxt(path + "down_regulation.csv", delimiter=',', skip_header=1, usecols=[-1]) / 10
DEFAULT_UP_REG = np.genfromtxt(path + "up_regulation.csv", delimiter=',', skip_header=1, usecols=[-1]) / 10
DEFAULT_TRANSFER_PRICE_IMPORT = 0.97
DEFAULT_TRANSFER_PRICE_EXPORT = 0.09
# Length of one episode(one day)
DEFAULT_ITERATIONS = 24
# TCLs HVAC
DEFAULT_NUM_TCLS = 100
DEFAULT_AVGTCLPOWER = 1.5
DEFAULT_TEMPERATURS = np.genfromtxt(path + "temperatures.csv", usecols=[5], skip_header=1, delimiter=',')
DEFAULT_TCL_SALE_PRICE = 3.2
DEFAULT_TCL_TMIN = 19
DEFAULT_TCL_TMAX = 25
# Price responsive loads
DEFAULT_NUM_LOADS = 150
DEFAULT_BASE_LOAD = np.array(
    [.4, .3, .2, .2, .2, .2, .3, .5, .6, .6, .5, .5, .5, .4, .4, .6, .8, 1.4, 1.2, .9, .8, .6, .5, .4])
DEFAULT_MARKET_PRICE = 5.48
DEFAULT_PRICE_TIERS = np.array([-2.0, 0.0, 2.0])  # Battery characteristics (kwh)
DEFAULT_BAT_CAPACITY = 1000
DEFAULT_MAX_CHARGE = 250
DEFAULT_MAX_DISCHARGE = 250
DEFAULT_BATTERY_DEMAND = np.array([100, 0.0, 100])

# 缩小奖励
MAX_R = 100

# 用于演示的列表
SOCS_RENDER = []
LOADS_RENDER = []
BATTERY_RENDER = []
PRICE_RENDER = []
ENERGY_SOLD_RENDER = []
ENERGY_BOUGHT_RENDER = []
GRID_PRICES_BUY_RENDER = []
GRID_PRICES_SELL_RENDER = []
ENERGY_GENERATED_RENDER = []
TCL_CONTROL_RENDER = []
TCL_CONSUMPTION_RENDER = []
TOTAL_CONSUMPTION_RENDER = []
TEMP_RENDER = []
AVG_SOC = []
REWAED_DAY = []
ENERGY_DAY = []

ACTIONS = [[i, j, k] for i in range(4) for j in range(3) for k in range(3)]


# 4*5*5=100
# i=1 j =2 k=3
# 1*25+2*5+3=38


class TCL:
    """
    Simulates an invidual TCL
    """

    def __init__(self, ca, cm, q, P, Tmin=DEFAULT_TCL_TMIN, Tmax=DEFAULT_TCL_TMAX):
        self.ca = ca
        self.cm = cm
        self.q = q
        self.P = P
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.ON = 0
        self.OFF = 0
        # Added for clarity
        self.u = 0

    def set_T(self, T, Tm):
        self.T = T
        self.Tm = Tm

    def control(self, ui=0):
        # control TCL using u with respect to the backup controller
        if self.T < self.Tmin - 3:
            self.u = 1
        elif self.Tmin - 3 < self.T < self.Tmax + 3:
            self.u = ui
        else:
            self.u = 0

    def control_ON(self, soc=0):
        # control TCL using u with respect to the backup controller
        if soc < 0:
            self.ON = 1
            self.OFF = 0
        elif soc > 1:
            self.ON = 0
            self.OFF = 1
        if self.ON == 1:
            self.u = 1
        else:
            self.u = 0

    def back_control(self):
        self.T = random.uniform(15, 24)
        self.Tm = self.T

    def update_state(self, T0):
        # update the indoor and mass temperatures according to (22)
        for _ in range(2):
            self.T += self.ca * (T0 - self.T) + self.cm * (self.Tm - self.T) + self.P * self.u + self.q
            self.Tm += self.cm * (self.T - self.Tm)
            if self.T >= self.Tmax:
                break

    """ 
    @property allows us to write "tcl.SoC", and it will
    run this function to get the value
    """

    @property
    def SoC(self):
        return (self.T - self.Tmin) / (self.Tmax - self.Tmin)


class Battery:
    # Simulates the battery system of the microGrid
    def __init__(self, capacity, useD, dissipation, rateC, maxDD, chargeE):
        self.capacity = capacity  # full charge battery capacity
        self.useD = useD  # useful discharge coefficient
        self.dissipation = dissipation  # dissipation coefficient of the battery
        self.rateC = rateC  # charging rate
        self.maxDD = maxDD  # maximum power that the battery can deliver per timestep
        self.chargeE = chargeE  # max Energy given to the battery to charge
        self.RC = 0  # remaining capacity

    def charge(self, E):
        empty = self.capacity - self.RC
        if empty <= 0:
            return E
        else:
            self.RC += self.rateC * min(E, self.chargeE)
            leftover = self.RC - self.capacity + max(E - self.chargeE, 0)  # 不能超过电池容量
            self.RC = min(self.capacity, self.RC)
            return max(leftover, 0)

    def supply(self, E):
        remaining = self.RC
        self.RC -= min(E, remaining, self.maxDD)  # e应该是正数
        self.RC = max(self.RC, 0)
        return min(E, remaining, self.maxDD) * self.useD

    def dissipate(self):
        self.RC = self.RC * math.exp(- self.dissipation)

    @property
    def SoC(self):
        return self.RC / self.capacity

    def reset(self):
        self.RC = 0


class Grid:
    def __init__(self, down_reg, up_reg, exp_fees, imp_fees):
        self.sell_prices = down_reg
        self.buy_prices = up_reg
        self.exp_fees = exp_fees
        self.imp_fees = imp_fees
        self.time = 0

    def sell(self, E):
        return (self.sell_prices[self.time] + self.exp_fees) * E

    def buy(self, E):
        return -(self.buy_prices[self.time] + self.imp_fees) * E

    # def get_price(self,time):
    #     return self.prices[time]

    def set_time(self, time):
        self.time = time

    def total_cost(self, prices, energy):
        return sum(prices * energy / 100 + self.imp_fees * energy)


class Generation:
    def __init__(self, generation):
        self.power = generation

    def current_generation(self, time):
        # We consider that we have 2 sources of power a constant source and a variable source
        return self.power[time]


class Load:
    def __init__(self, price_sens, base_load, max_v_load, patience):
        self.price_sens = max(0, price_sens)
        self.orig_price_sens = max(0, price_sens)
        self.base_load = base_load
        self.max_v_load = max_v_load
        self.response = 0
        self.shifted_loads = {}
        self.patience = max(patience, 1)
        self.dr_load = 0

    def react(self, price_tier, time_day):
        self.dr_load = self.base_load[time_day]
        response = self.price_sens * (price_tier - 2)
        if response != 0:
            self.dr_load -= self.base_load[time_day] * response
            self.shifted_loads[time_day] = self.base_load[time_day] * response
        for k in list(self.shifted_loads):
            probability_of_execution = -self.shifted_loads[k] * (price_tier - 2) + (time_day - k) / self.patience
            if random.random() <= probability_of_execution:
                self.dr_load += self.shifted_loads[k]
                del self.shifted_loads[k]

    def load(self):
        return max(self.dr_load, 0)


class MicroGridEnv(gym.Env):
    def __init__(self, **kwargs):
        # Get number of iterations and TCLs from the
        # parameters (we have to define it through kwargs because
        # of how Gym works...)
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.num_tcls = kwargs.get("num_tcls", DEFAULT_NUM_TCLS)
        self.avg_tcl_power = kwargs.get("tcl_power", DEFAULT_AVGTCLPOWER)
        self.tcl_sale_price = kwargs.get("tcl_price", DEFAULT_TCL_SALE_PRICE)
        self.num_loads = kwargs.get("num_loads", DEFAULT_NUM_LOADS)
        self.typical_load = kwargs.get("base_load", DEFAULT_BASE_LOAD)
        self.market_price = kwargs.get("normal_price", DEFAULT_MARKET_PRICE)
        self.temperatures = kwargs.get("temperatures", DEFAULT_TEMPERATURS)
        self.price_tiers = kwargs.get("price_tiers", DEFAULT_PRICE_TIERS)
        self.day0 = kwargs.get("day0", DEFAULT_DAY0)
        self.dayn = kwargs.get("dayn", DEFAULT_DAYN)
        self.power_cost = kwargs.get("power_cost", DEFAULT_WIND_POWER_COST)
        self.down_reg = kwargs.get("down_reg", DEFAULT_DOWN_REG)
        self.up_reg = kwargs.get("up_reg", DEFAULT_UP_REG)
        self.imp_fees = kwargs.get("imp_fees", DEFAULT_TRANSFER_PRICE_IMPORT)
        self.exp_fees = kwargs.get("exp_fees", DEFAULT_TRANSFER_PRICE_EXPORT)
        self.bat_capacity = kwargs.get("battery_capacity", DEFAULT_BAT_CAPACITY)
        self.max_discharge = kwargs.get("max_discharge", DEFAULT_MAX_DISCHARGE)
        self.max_charge = kwargs.get("max_charge", DEFAULT_MAX_CHARGE)
        self.day = self.day0
        # The current timestep
        self.time_step = 0
        # reward record
        self.reward_pv = 0
        self.reward_soc = 0
        self.reward_tcl = 0
        self.reward_tcl = 0
        self.reward_buy = 0
        self.reward_sell = 0
        self.reward_load = 0
        self.avg_tcl_soc = 0
        self.energy_pv = 0
        self.energy_tcl = 0
        self.energy_load = 0
        self.energy_sell = 0
        self.energy_buy = 0
        self.energy_battery = 0
        self.elc_available_energy = 0
        # The cluster of TCLs to be controlled.
        # These will be created in reset()
        self.tcls_parameters = []
        # The cluster of loads.
        # These will be created in reset()
        self.loads_parameters = []
        self.generation = Generation(kwargs.get("generation_data", DEFAULT_POWER_GENERATED))
        self.grid = Grid(down_reg=self.down_reg, up_reg=self.up_reg, exp_fees=self.exp_fees, imp_fees=self.imp_fees)
        self.battery = Battery(capacity=self.bat_capacity, useD=0.99, dissipation=0.001, rateC=0.99,
                               maxDD=self.max_discharge, chargeE=self.max_charge)

        self.tcls = [self._create_tcl(*self._create_tcl_parameters()) for _ in range(self.num_tcls)]
        self.loads = [self._create_load(*self._create_load_parameters()) for _ in range(self.num_loads)]
        # env parameter
        self.action_space_sep = spaces.Box(low=0, high=1, dtype=np.float32, shape=(14,))
        self.action_space = spaces.Discrete(36)

        # Observations:
        self.observation_space = spaces.Box(low=-100, high=100, dtype=np.float32,
                                            shape=(7,))

    def _create_tcl_parameters(self):
        """
                Initialize one TCL randomly with given T_0,
                and return it. Copy/paste from Taha's code
                """
        # Hardcoded initialization values to create
        # bunch of different TCLs
        ca = random.normalvariate(0.004, 0.0008)
        cm = random.normalvariate(0.3, 0.004)
        q = random.normalvariate(0, 0.01)
        P = random.normalvariate(self.avg_tcl_power, 0.01)
        init_temp = random.uniform(15, 24)
        return [ca, cm, q, P, init_temp]

    def _create_tcl(self, ca, cm, q, P, init_temp):
        tcl = TCL(ca, cm, q, P)
        tcl.set_T(init_temp, init_temp)
        return tcl

    def _create_load_parameters(self):
        """
        Initialize one load randomly,
        and return it.
        """
        # Hardcoded initialization values to create
        # bunch of different loads

        price_sensitivity = random.normalvariate(0.4, 0.3)
        max_v_load = random.normalvariate(0.4, 0.01)
        patience = int(random.normalvariate(10, 6))
        return [price_sensitivity, max_v_load, patience]

    def _create_load(self, price_sensitivity, max_v_load, patience):
        load = Load(price_sensitivity, base_load=self.typical_load, max_v_load=max_v_load, patience=patience)
        return load

    def _build_state(self):
        """
        Return current state representation as one vector.
        Returns:
            state: 1D state vector, containing state-of-charges of all TCLs, Loads, current battery soc, current power generation,
                   current temperature, current price and current time (hour) of day
        """
        # SoCs of all TCLs binned + current temperature + current price + time of day (hour)
        socs = np.array([tcl.SoC for tcl in self.tcls])
        # Scaling between 0 and 1
        # We need to standardize the generation and the price
        # Minimum soc is -1
        socs = (socs + np.ones(shape=socs.shape)) / 2
        loads = self.typical_load[(self.time_step) % self.iterations]
        loads = (loads - min(self.typical_load)) / (max(self.typical_load) - min(self.typical_load))

        current_generation = self.generation.current_generation(self.day * self.iterations + self.time_step)
        current_generation = (current_generation -
                              np.average(self.generation.power[
                                         self.day * self.iterations:self.day * self.iterations + self.iterations])) \
                             / np.std(
            self.generation.power[self.day * self.iterations:self.day * self.iterations + self.iterations])
        temperature = self.temperatures[self.day * self.iterations + self.time_step]
        temperature = (temperature -
                       min(self.temperatures[self.iterations:9 * self.iterations + self.iterations])) \
                      / (max(self.temperatures[self.iterations:9 * self.iterations + self.iterations])
                         - min(self.temperatures[self.iterations:9 * self.iterations + self.iterations]))
        price_grid_buy = self.grid.buy_prices[self.time_step]
        price_grid_buy = (price_grid_buy -
                          np.average(self.grid.buy_prices[0:self.iterations])) \
                         / np.std(self.grid.buy_prices[0:self.iterations])

        # 买卖电价都是阶梯电价的形式所以可以省去卖电价
        # price_grid_sell = self.grid.sell_prices[self.time_step]
        # price_grid_sell = (price_grid_sell -
        #          np.average(self.grid.sell_prices[0:self.iterations])) \
        #         / np.std(self.grid.sell_prices[0:self.iterations])
        # high_price = min(self.high_price/4,1)

        time_step = (self.time_step) / (self.iterations - 1)
        avage_soc = np.mean(socs)
        # print([i-avage_soc for i in socs])
        std_soc = np.std(socs)
        # print(std_soc)
        state = [avage_soc, loads, time_step, self.battery.SoC, current_generation, price_grid_buy, temperature]
        return state

    # 暂时没用
    def _build_info(self):
        """
        Return dictionary of misc. infos to be given per state.
        Here this means providing forecasts of future
        prices and temperatures (next 24h)
        """
        temp_forecast = np.array(self.temperatures[self.time_step + 1:self.time_step + self.iterations + 1])
        return {"temperature_forecast": temp_forecast,
                "forecast_times": np.arange(0, self.iterations)}

    # 用于实际的tcl计算
    def _compute_tcl_power(self):
        """
        Return the total power consumption of all TCLs
        """
        return sum([tcl.u * tcl.P for tcl in self.tcls])

    def step(self, action, test=False):
        """
        Arguments:
            action: A num need tansform one-hot.

        Returns:
            state: Current state
            reward: How much reward was obtained on last action
            terminal: Boolean on if the game ended (maximum number of iterations)
            info: None (not used here)
        """
        # tansform to one-hot.
        if type(action) is not list:
            action = ACTIONS[action]

        self.grid.set_time(self.time_step)
        reward = 0
        # 分别表示每一个动作
        tcl_action = action[0]
        price_action = action[1]
        # 减一是为了让价格响应有正有负之后避免一直高价收取利润的操作
        # self.high_price += price_action - 1
        # if self.high_price > 4:
        #     price_action = 1
        #     self.high_price = 4
        battery_action = action[2]

        # 从风能发电中获取产能
        available_energy = self.generation.current_generation(self.day * self.iterations + self.time_step)
        # Calculate the cost of energy produced from wind turbines
        reward -= available_energy * self.power_cost / 100
        # print("wind:", available_energy * self.power_cost / 100)
        self.reward_pv += available_energy * self.power_cost / 100
        self.energy_pv += available_energy
        # 从价格响应负载获取利润
        # We implement the pricing action and we calculate the total load in response to the price
        for load in self.loads:
            load.react(price_tier=price_action, time_day=self.time_step)
        total_loads = sum([l.load() for l in self.loads])
        available_energy -= total_loads
        self.energy_load += total_loads

        # We calculate the return based on the sale price.
        self.sale_price = self.price_tiers[price_action] + self.market_price
        # We increment the reward by the amount of return
        # Division by 100 to transform from cents to euros
        reward += total_loads * (self.sale_price) / 100
        # print("price:", total_loads * (self.sale_price) / 100)
        self.reward_load += total_loads * (self.sale_price) / 100

        # 从温度负载获取利润
        avage_soc = np.mean([tcl.SoC for tcl in self.tcls])
        # Distributing the energy according to priority
        # HVAC
        sortedTCLs = sorted(self.tcls, key=lambda x: x.SoC)
        control = max(tcl_action * self.num_tcls * self.avg_tcl_power / 12, 0)
        self.control = control
        for tcl in sortedTCLs:
            if control > 0:
                tcl.control(1)
                control -= tcl.P * tcl.u
            else:
                tcl.control(0)
            # tcl.control_ON(avage_soc)
            tcl.update_state(self.temperatures[self.day * self.iterations + self.time_step])
        available_energy -= self._compute_tcl_power()
        self.energy_tcl += self._compute_tcl_power()
        reward += self._compute_tcl_power() * self.tcl_sale_price / 100
        # print('tcl*price',self._compute_tcl_power() * self.tcl_sale_price / 100)
        self.reward_tcl += self._compute_tcl_power() * self.tcl_sale_price / 100
        self.elc_available_energy += available_energy
        # 从舒适度中获取奖励
        punish = 0
        avage_soc = np.mean([tcl.SoC for tcl in self.tcls])
        # for tcl in self.tcls:
        #     if tcl.SoC>1:
        #         punish+=(tcl.T-tcl.Tmax)
        #     elif tcl.SoC<0:
        #         punish+=(tcl.Tmin-tcl.T)
        # if punish<20:
        #     self.reward_soc+=1
        #     reward+=2
        # elif punish<200:
        #     reward+=2*(-(punish/200)**3+0.5)
        #     self.reward_soc+=2*(-(punish/200)**3+0.5)
        # else:
        #     self.reward_soc-=0.8
        #     reward-=0.8

        self.avg_tcl_soc = avage_soc
        if avage_soc <= 1 and avage_soc >= 0:
            reward += 1
            self.reward_soc += 4 * (1 - avage_soc)
        elif avage_soc > 1:
            reward += max(-8 * (avage_soc - 1), -1.5)/2
            self.reward_soc += max(-8 * (avage_soc - 1), -1.5)
        elif avage_soc < 0:
            reward += max(8 * (avage_soc), -1.5)/2
            self.reward_soc += max(8 * (avage_soc), -1.5)
        # 电池操作
        # print(self.time_step,self.battery.SoC,"电池动作",battery_action,self.grid.buy_prices[self.time_step])
        # print(available_energy)
        if battery_action < 1:
            available_energy -= DEFAULT_BATTERY_DEMAND[battery_action]
            self.energy_battery += DEFAULT_BATTERY_DEMAND[battery_action]
            return_battery = self.battery.charge(DEFAULT_BATTERY_DEMAND[battery_action])
            available_energy += return_battery
            self.energy_battery -= return_battery
            # print(available_energy)
        elif battery_action == 1:
            pass
        elif battery_action > 1:
            temp = self.battery.supply(DEFAULT_BATTERY_DEMAND[battery_action])
            available_energy += temp
            self.energy_battery -= temp

            # print(available_energy)
        # 最后与主电网的交互补充差额
        if available_energy > 0:
            # print("电力多余")
            reward += self.grid.sell(available_energy) / 100
            # print("电力多余:", self.grid.sell(available_energy) / 100)
            self.reward_sell += self.grid.sell(available_energy) / 100
            self.energy_sell += available_energy
            self.energy_sold = available_energy
            self.energy_bought = 0
        else:
            # print("电力不足")
            self.energy_bought = -available_energy
            reward += self.grid.buy(self.energy_bought) / 100
            # print("电力不足:", self.grid.buy(self.energy_bought) / 100)
            self.energy_buy += self.energy_bought
            self.reward_buy -= self.grid.buy(self.energy_bought) / 100
            self.energy_sold = 0

        # Proceed to next timestep.
        self.time_step += 1
        # Build up the representation of the current state (in the next timestep)
        state = self._build_state()
        terminal = self.time_step == self.iterations
        if terminal:
            if test == True:
                print(self.reward_sell)
                REWAED_DAY.append(self.reward_sell)
                ENERGY_DAY.append(self.reward_buy)
                # print( self.day,"comfortable",self.reward_soc,"tcl",self.reward_tcl,"sell",
                # self.reward_sell-self.reward_buy, "load",self.reward_load,"avaible",self.elc_available_energy)
                # print( self.day,"pv",self.energy_pv,"tcl",self.energy_tcl,"load",self.energy_load,"sell",
                # self.energy_sell,"buy",self.energy_buy,"battery",self.energy_battery)
                if self.day == 9:
                    # self.reward_sell += self.grid.sell(self.battery.SoC*1000) / 100
                    # reward+= self.grid.sell(self.battery.SoC*1000) / 100
                    # print("111",self.grid.sell(self.battery.SoC*1000) / 100)
                    # print(self.day, "sell", self.energy_sell, "buy", self.energy_buy, )
                    np.save("result/" + 'QEMS_energy_sellnum.npy', REWAED_DAY)
                    print(REWAED_DAY)
                    np.save("result/" + 'QEMS_energy_buynum.npy', ENERGY_DAY)
                    print(ENERGY_DAY)
                    np.save("result/" + 'QEMS_total_sell.npy', np.array(REWAED_DAY) - np.array(ENERGY_DAY))
                # print(self.reward_pv/100)
                # self.reward_pv=0
                # self.reward_soc=0
                # self.reward_tcl=0
            self.reward_sell = 0
            self.reward_buy = 0
            self.energy_sell = 0
            self.energy_buy = 0
            # self.elc_available_energy = 0
            # self.reward_load =0

        info = self._build_info()
        return state, reward / MAX_R, terminal, info

    def reset(self, day=None):
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        if day == None:
            self.day = self.day0
        else:
            self.day = day
        # print("Day:", self.day)
        self.time_step = 0
        self.high_price = 0
        return self._build_state()

    def reset_all(self, seed=None):
        """
        Create new TCLs, and return initial state.
        Note: Overrides previous TCLs
        """
        if seed == None:
            self.day = self.seed
        else:
            self.day = seed
        self.time_step = 0
        self.battery.reset()
        self.high_price = 0
        for load in self.loads:
            load.shifted_loads.clear()
        for tcl in self.tcls:
            tcl.back_control()
        # self.tcls.clear()
        # self.loads.clear()
        # self.tcls = [self._create_tcl(*self._create_tcl_parameters()) for _ in range(self.num_tcls)]
        # self.loads = [self._create_load(*self._create_load_parameters()) for _ in range(self.num_loads)]
        return self._build_state()

    def render(self, test_day, name='', display=[0, 1, 0, 0]):
        SOCS_RENDER.append([tcl.SoC * 6 + 19 for tcl in self.tcls])
        LOADS_RENDER.append([l.load() for l in self.loads])
        PRICE_RENDER.append(self.sale_price)
        BATTERY_RENDER.append(self.battery.SoC)
        ENERGY_GENERATED_RENDER.append(
            self.generation.current_generation(self.day * self.iterations + self.time_step - 1))
        ENERGY_SOLD_RENDER.append(self.energy_sold)
        ENERGY_BOUGHT_RENDER.append(self.energy_bought)
        GRID_PRICES_BUY_RENDER.append(self.grid.buy_prices[self.time_step - 1])
        GRID_PRICES_SELL_RENDER.append(self.grid.sell_prices[self.time_step - 1])
        TCL_CONTROL_RENDER.append(self.control)
        TCL_CONSUMPTION_RENDER.append(self._compute_tcl_power())
        TOTAL_CONSUMPTION_RENDER.append(self._compute_tcl_power() + np.sum([l.load() for l in self.loads]))
        TEMP_RENDER.append(self.temperatures[self.day * self.iterations + self.time_step - 1])
        AVG_SOC.append(self.avg_tcl_soc)
        if self.time_step == self.iterations:
            # 1tcl控制信息
            if display[1] == 1 or display[0] == 1:
                # tcl的温度控制情况
                fig = plt.figure(figsize=(6, 3))
                # ax = pyplot.axes()
                ax = plt.subplot(1, 1, 1)
                plt.axhspan(19, 25, facecolor='#228B22', alpha=0.5)
                if test_day == 0:
                    ax.set_ylabel("indoor temperature (°C)", fontdict={'size': 13})
                # ax.set_facecolor("silver")
                # ax.yaxis.grid(True)
                ax.set_xlabel("Time (h)", fontdict={'size': 13})
                plt.yticks(np.arange(15, 30, step=2), )
                ax.boxplot(SOCS_RENDER, positions=range(24), patch_artist=True, boxprops={'facecolor': "#1FA6BB"},
                           showfliers=False)
                print("Day:", test_day)
                if test_day < 3:
                    plt.tight_layout()
                    # plt.savefig('resultfig/QRL1/cu_temper{}.pdf'.format(test_day) , format='pdf')
                    np.save("result/" + 'QEMS_SOCS_RENDER{}.npy'.format(test_day), SOCS_RENDER)
                    plt.close()  # 关闭图形
                # ax1 = ax.twinx()
                # ax1.set_ylabel("Temperatures °C")
                # ax1.plot(np.array(TEMP_RENDER), '--')
                # plt.title("Comparison of indoor temperature change")
                # plt.xlabel("Time (h)")
                # plt.legend(["Outdoor Temperatures"], loc='lower right')
                # plt.show()
                # ax = plt.subplot(2, 1, 2)
                # ax.set_facecolor("silver")
                # ax.set_ylabel("kW")
                # ax.set_xlabel("Time (h)")
                # ax.yaxis.grid(True)
                # ax.plot(GRID_PRICES_BUY_RENDER, color='k')
                # ax.bar(x=np.array(np.arange(self.iterations)) - 0.2, height=TCL_CONTROL_RENDER, width=0.2)
                # ax.bar(x=np.array(np.arange(self.iterations)), height=TCL_CONSUMPTION_RENDER, width=0.2)
                # plt.xticks( np.array(np.arange(self.iterations)))
                # plt.title("Energy allocated to and consumed by TCLs and energy generated")
                # plt.legend(['GRID_PRICES_BUY','Energy allocated for TCLs', 'Energy consumed by TCLs'],
                # loc='upper right')
                # plt.xlabel("Time (h)")
                # plt.ylabel("kW")
                # plt.show()
                # plt.close()  # 关闭图形
                # 2 价格响应负载信息
            if display[2] == 1 or display[0] == 1:
                # 定价变换图
                ax = plt.axes()
                ax.set_facecolor("silver")
                ax.yaxis.grid(True)
                plt.plot(PRICE_RENDER, color='k')
                plt.title("SALE PRICES")
                plt.xlabel("Time (h)")
                plt.ylabel("€ cents")
                plt.show()
                plt.close()
                # if test_day<10:  # 保存测试的图
                #     plt.savefig('resultfig/QRL1/SALE_PRICES{}.png'.format(test_day), format='png')
                #     plt.close()
                # 价格响应负载的供应情况
                ax = plt.axes()
                ax.set_facecolor("silver")
                ax.set_ylabel("kW")
                ax.set_xlabel("Time (h)")
                ax.yaxis.grid(True)
                plt.boxplot(np.array(LOADS_RENDER).T)
                plt.title("Hourly residential loads")
                plt.xlabel("Time (h)")
                plt.show()
                plt.close()
                # if test_day<10:
                #     plt.savefig('resultfig/QRL1/Hourly_residential_loads{}.png'.format(test_day), format='png')
                #     plt.close()

            if display[3] == 1 or display[0] == 1:
                # 电池变换图
                # ax = plt.axes()
                # # ax.set_facecolor("silver")
                # ax.set_xlabel("Time (h)")
                # ax.yaxis.grid(True)
                # plt.plot(np.array(BATTERY_RENDER),color='k')
                # plt.title("ESS SOC")
                # plt.xlabel("Time (h)")
                # # ax4.set_ylabel("BATTERY SOC")
                # plt.show()
                #
                # #总消耗变化图
                # ax = plt.axes()
                # ax.set_facecolor("silver")
                # ax.set_xlabel("Time (h)")
                # ax.set_ylabel("kWh")
                # ax.yaxis.grid(True)
                # plt.plot(np.array(TOTAL_CONSUMPTION_RENDER), color='k')
                # plt.title("Demand")
                # plt.xlabel("Time (h)")
                # plt.show()

                # 每日消耗模型
                # ax = plt.axes()
                # ax.set_facecolor("silver")
                # ax.set_xlabel("Time (h)")
                # ax.yaxis.grid(True)
                # plt.plot(np.array(self.typical_load), color='k')
                # plt.title("Expected Individual basic load (L_b)")
                # plt.xlabel("Time (h)")
                # plt.ylabel("kWh")
                # plt.show()

                # 能源产生情况
                # ax = plt.axes()
                # ax.set_facecolor("silver")
                # ax.yaxis.grid(True)
                # plt.plot(np.array(ENERGY_GENERATED_RENDER),color='k')
                # plt.title("ENERGY GENERATED")
                # plt.xlabel("Time (h)")
                # plt.ylabel("kW")
                # plt.show()

                # #能源售卖图和电价图
                ax = plt.axes()
                # ax.set_facecolor("silver")
                ax.yaxis.grid(True)
                ax.set_axisbelow(True)
                # ax.axis(ymin=0,ymax=610)
                ax.bar(x=np.array(np.arange(self.iterations)), height=np.array(ENERGY_SOLD_RENDER), color='#D65F5F',
                       width=0.8)
                ax.bar(x=np.array(np.arange(self.iterations)), height=np.array(ENERGY_BOUGHT_RENDER), color='#6ACC64',
                       width=0.8)
                if test_day == 0:
                    ax.set_ylabel("Energy Exchanged (kWh)", fontdict={'size': 13})
                ax.set_xlabel("Time (h)", fontdict={'size': 13})
                ax.legend(['Energy sold', 'Energy purchased'], loc='upper left', prop={'size': 13})
                # pyplot.show()

                ax1 = ax.twinx()
                ax1.plot(np.array(GRID_PRICES_BUY_RENDER), linestyle='solid', color='#4878D0', linewidth=2.0)
                ax1.plot(np.array(GRID_PRICES_SELL_RENDER), linestyle='dashed', color='#4878D0', linewidth=2.0)
                plt.tick_params(axis='y', colors='#4878D0')
                ax.spines['left'].set_color('#4878D0')
                if test_day == 2:
                    ax1.set_ylabel("GRID PRICES € cents", fontdict={'size': 13})
                ax1.legend(['Buying prices', 'Selling prices'], loc='upper right', prop={'size': 13})
                if test_day < 10:
                    # plt.savefig('resultfig/QRL1/cu_sell{}.pdf'.format(test_day), format='pdf')
                    np.save("result/" + 'QEMS_ENERGY_SOLD_RENDER{}.npy'.format(test_day), ENERGY_SOLD_RENDER)
                    np.save("result/" + 'QEMS_ENERGY_BOUGHT_RENDER{}.npy'.format(test_day), ENERGY_BOUGHT_RENDER)
                    plt.close()
                # plt.show()

            SOCS_RENDER.clear()
            LOADS_RENDER.clear()
            PRICE_RENDER.clear()
            BATTERY_RENDER.clear()
            GRID_PRICES_BUY_RENDER.clear()
            GRID_PRICES_SELL_RENDER.clear()
            ENERGY_BOUGHT_RENDER.clear()
            ENERGY_SOLD_RENDER.clear()
            ENERGY_GENERATED_RENDER.clear()
            TCL_CONTROL_RENDER.clear()
            TCL_CONSUMPTION_RENDER.clear()
            TOTAL_CONSUMPTION_RENDER.clear()
            TEMP_RENDER.clear()

        if test_day == 9 and self.time_step == self.iterations:
            ma_rewards = []
            fig = plt.axes()
            fig.set_facecolor("silver")
            fig.yaxis.grid(True)
            x = list(np.arange(1, 241))
            for av_soc in AVG_SOC:
                if ma_rewards:
                    ma_rewards.append(ma_rewards[-1] * 0.9 + av_soc * 0.1)
                else:
                    ma_rewards.append(av_soc)
            plt.plot(x, AVG_SOC, c='b', alpha=0.8)
            plt.plot(x, ma_rewards, c='r', alpha=0.8)
            plt.title("Comfort SOC")
            plt.xlabel("Time (h)")
            plt.ylabel("average soc")
            plt.legend(['Comfort', 'Smooth comfort'], loc='upper left')
            plt.show()
            np.save("result/" + 'QEMS_avg_soc.npy', AVG_SOC)
            print(AVG_SOC)

            # ax = plt.axes()
            # ax.set_facecolor("silver")
            # ax.yaxis.grid(True)
            # # ax.axis(ymin=0,ymax=610)
            # #ax.plot(BATTERY_RENDER, c='b', alpha=0.8)
            # ax.bar(x=np.array(np.arange(self.iterations*4)), height=np.array(BATTERY_RENDER), color='navy',
            #        )
            # ax.set_xlabel("Time (h)")
            # ax.set_ylabel("SOC")
            # ax.legend(['BATTERY SOC'], loc='upper left')
            # # pyplot.show()
            #
            # ax1 = ax.twinx()
            # ax1.plot( GRID_PRICES_BUY_RENDER, c='r', alpha=0.8)
            # ax1.plot(np.array(GRID_PRICES_BUY_RENDER), color='red')
            # ax1.set_ylabel("GRID PRICES € cents")
            # ax1.legend(['Buying prices'], loc='upper right')
            # plt.show()
            # ax = plt.axes()
            # ax.set_facecolor("silver")
            # ax.yaxis.grid(True)
            # # ax.axis(ymin=0,ymax=610)
            # #ax.plot(BATTERY_RENDER, c='b', alpha=0.8)
            # ax.bar(x=np.array(np.arange(self.iterations*4)), height=np.array(BATTERY_RENDER), color='navy',
            #        )
            # ax.set_xlabel("Time (h)")
            # ax.set_ylabel("SOC")
            # ax.legend(['BATTERY SOC'], loc='upper left')
            # # pyplot.show()
            #
            # ax1 = ax.twinx()
            # ax1.plot( GRID_PRICES_BUY_RENDER, c='r', alpha=0.8)
            # ax1.plot(np.array(GRID_PRICES_BUY_RENDER), color='red')
            # ax1.set_ylabel("GRID PRICES € cents")
            # ax1.legend(['Buying prices'], loc='upper right')
            # plt.show()

    def close(self):
        """
        Nothing to be done here, but has to be defined
        """
        return

    def seedy(self, s):
        """
        Set the random seed for consistent experiments
        """
        random.seed(s)
        np.random.seed(s)


if __name__ == '__main__':
    # Testing the environment
    # Initialize the environment
    env = MicroGridEnv()
    env.seedy(1)
    # Save the rewards in a list
    rewards = []
    # reset the environment to the initial state
    state = env.reset()
    # Call render to prepare the visualization

    # Interact with the environment (here we choose random actions) until the terminal state is reached
    while True:
        # Pick an action from the action space (here we pick an index between 0 and 80)
        # action = env.action_space.sample()
        # action =[np.argmax(action[0:4]),np.argmax(action[4:9]),np.argmax(action[9:11]),np.argmax(action[11:])]
        action = [1, 2, 0, 0]
        # Using the index we get the actual action that we will send to the environment
        # print(ACTIONS[action])
        print(action)
        # Perform a step in the environment given the chosen action
        # state, reward, terminal, _ = env.step(action)
        state, reward, terminal, _ = env.step(list(action))
        env.render()
        print(reward)
        rewards.append(reward)
        if terminal:
            break
    print("Total Reward:", sum(rewards))
    # Plot the TCL SoCs
    states = np.array(rewards)
    plt.plot(rewards)
    plt.title("rewards")
    plt.xlabel("Time")
    plt.ylabel("rewards")
    plt.show()

