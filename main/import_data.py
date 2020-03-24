# -*- coding=utf-8 -*-

import numpy as np

def get_test_data(input_size=100):
    mass_input_size = 1000
    Mass = np.random.choice(mass_input_size,input_size)

    stiff_input_size = 2000
    Stiffness = np.random.choice(stiff_input_size,input_size)

    x_test = np.array([ [x,y] for x,y in zip(Mass,Stiffness)])
    t_test = np.array([ [2*np.pi*(mass/stiff)**0.5] for mass,stiff in zip(Mass,Stiffness)])
    return x_test, t_test

def get_train_data(input_size=100):
    mass_input_size = 100
    Mass = np.linspace(1,mass_input_size,100)
    np.random.shuffle(Mass)

    stiff_input_size = 200
    Stiffness = np.linspace(1,stiff_input_size,100)
    np.random.shuffle(Stiffness)

    # #random
    # mass_input_size = 1000
    # Mass = np.random.choice(mass_input_size,input_size)

    # stiff_input_size = 2000
    # Stiffness = np.random.choice(stiff_input_size,input_size)


    x_train = np.array([ [x,y] for x,y in zip(Mass,Stiffness)])
    t_train = np.array([ [2*np.pi*(mass/stiff)**0.5] for mass,stiff in zip(Mass,Stiffness)])
    return x_train, t_train


if __name__ == "__main__":
    x_test, t_test = get_test_data()
    print(t_test)
    

