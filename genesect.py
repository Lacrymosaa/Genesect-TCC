from __future__ import division, print_function
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import matplotlib.lines as lines
from drive import drive_food, drive_organism
import numpy as np
import operator
from math import atan2, cos, degrees, floor,radians, sin, sqrt
from random import random, randint, sample, uniform

settings = {}

# Configurações de evolução
settings['pop_size'] = 50       # número de organismos
settings['food_amo'] = 100      # número de particulas de comida
settings['gens'] = 50           # número de gerações
settings['elitism'] = 0.20      # elitismo (selection bias)
settings['mutate'] = 0.10       # taxa de mutação

# Configurações de simulação
settings['gen_time'] = 100      # tamanho da geração (segundos)
settings['dt'] = 0.04           # passos de tempo da simulação (delta time (quantidade de tempo entre calculos dentro da simulação))
settings['dr_max'] = 720        # velocidade máxima de rotação (graus por segundo)
settings['v_max'] = 0.5         # velocidade máxima (unidades por segundo)
settings['dv_max'] =  0.25      # aceleração máxima (+/-) (unidades por segundo^2)

# Configurações da arena -> todas as medidas são partem de um ponto central expandindo para os determinados lados
settings['x_min'] = -2.0        # borda oeste
settings['x_max'] =  2.0        # borda leste
settings['y_min'] = -2.0        # borda sul
settings['y_max'] =  2.0        # borda norte

settings['plot'] = False        # plot final generation?

# Configuração de rede neural de organismos
settings['inodes'] = 1          # número de nós de entrada
settings['hnodes'] = 5          # número de nós ocultos
settings['onodes'] = 2          # número de nós de saída


#--- Classes ---#

class food():
    def __init__(self, settings): # A função uniforme pegará um número aleatório entre o máximo X/Y e o minimo X/Y para o alimento ser inserido
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1


    def respawn(self,settings):
        self.x = uniform(settings['x_min'], settings['x_max'])
        self.y = uniform(settings['y_min'], settings['y_max'])
        self.energy = 1


class organism(): # Classe que armazenará todas as informações dos indivíduos
    def __init__(self, settings, wih=None, who=None, name=None):

        self.x = uniform(settings['x_min'], settings['x_max'])  # Posição (x)
        self.y = uniform(settings['y_min'], settings['y_max'])  # Posição (y)

        self.r = uniform(0,360)                 # Orientação [0, 360] -> Direção que irá seguir
        self.v = uniform(0,settings['v_max'])   # Velocidade [0, v_max] -> Entre 0 e o máximo permitido
        self.dv = uniform(-settings['dv_max'], settings['dv_max']) # Aceleração 

        self.d_food = 100   # Distancia do alimento mais perto
        self.r_food = 0     # Direção para a comida mais próxima
        self.fitness = 0    # Fitness (contador de quanto se alimentou)

        self.wih = wih # Pesos da camada oculta da rede neural
        self.who = who # Pesos da camada de saída da rede neural

        self.name = name # Nome, ele merece!


    # Rede neural
    def think(self): # Calcula as saídas de rede neural com base na entrada, nesse caso sendo a distância do alimento mais próximo

        # SIMPLE MLP
        af = lambda x: np.tanh(x)               # Função de ativação -> Calcula a tangente hiperbólica de X
        h1 = af(np.dot(self.wih, self.r_food))  # Camada oculta
        out = af(np.dot(self.who, h1))          # Camada de saída

        # UPDATE dv AND dr WITH MLP RESPONSE
        self.nn_dv = float(out[0])   # [-1, 1]  (acelerar = 1, desacelerar = -1)
        self.nn_dr = float(out[1])   # [-1, 1]  (esquerda = 1, direita = -1)


    # UPDATE direção
    def update_r(self, settings): # Atualiza a orientação com base nas respostas de nn_dr 
        self.r += self.nn_dr * settings['dr_max'] * settings['dt'] # Multiplica valor da resposta da rede pela a configuração de distância máxima e pelo intervalo de tempo
        self.r = self.r % 360 # Ajusta o resultaod para estar dentro do intervalo em graus


    # UPDATE velocidade
    def update_vel(self, settings): # Atualiza a velocidade com base nas respostas de nn_dv
        self.v += self.nn_dv * settings['dv_max'] * settings['dt'] # Multiplica valor da resposta da rede pelas configurações de velocidade máxima e pelo intervalo de tempo 
        if self.v < 0: self.v = 0 # Limita o resultado para que ele não seja menor que zero e nem maior que o máximo
        if self.v > settings['v_max']: self.v = settings['v_max']


    # UPDATE posição
    def update_pos(self, settings):
        dx = self.v * cos(radians(self.r)) * settings['dt']
        dy = self.v * sin(radians(self.r)) * settings['dt']
        self.x += dx
        self.y += dy



#--- Funções ---#

def dist(x1,y1,x2,y2):
    return sqrt((x2-x1)**2 + (y2-y1)**2) # Retorne a raiz quadrada da conta para entregar a posição na arena


def calc_heading(org, food): # Calcular distância entre o organismo e o alimento
    d_x = food.x - org.x
    d_y = food.y - org.y
    theta_d = degrees(atan2(d_y, d_x)) - org.r #  Calcula a diferença entre o ângulo do alimento e o ângulo em que o indivíduo se dirige.
    if abs(theta_d) > 180: theta_d += 360 # Recalibra o ângulo desejado
    return theta_d / 180


def plot_frame(settings, organisms, foods, gen, time):
    # Criação da figura e seus pontos
    fig, ax = plt.subplots()
    fig.set_size_inches(9.6, 5.4)

    plt.xlim([settings['x_min'] + settings['x_min'] * 0.25, settings['x_max'] + settings['x_max'] * 0.25])
    plt.ylim([settings['y_min'] + settings['y_min'] * 0.25, settings['y_max'] + settings['y_max'] * 0.25])

    # Gera os organismos
    for organism in organisms:
        drive_organism(organism.x, organism.y, organism.r, ax)

    # Gera comida
    for food in foods:
        drive_food(food.x, food.y, ax)

    # Configura a figura
    ax.set_aspect('equal')
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])

    plt.figtext(0.025, 0.95,r'GENERATION: '+str(gen))
    plt.figtext(0.025, 0.90,r'T_STEP: '+str(time))

    # Salva a figura
    plt.savefig(str(gen)+'-'+str(time)+'.png', dpi=100)
##    plt.show()


def evolve(settings, organisms_old, gen):

    elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_orgs = settings['pop_size'] - elitism_num

    #--- Pega os status da geração atual
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness

        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']


    #--- Elitismo (manterá os individuos com melhor performance)
    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = []
    for i in range(0, elitism_num):
        organisms_new.append(organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name))


    # Geração de novos organismos
    for w in range(0, new_orgs):

        # Seleção (truncamento)
        canidates = range(0, elitism_num)
        random_index = sample(canidates, 2)
        org_1 = orgs_sorted[random_index[0]]
        org_2 = orgs_sorted[random_index[1]]

        # Crossover: selecionar individuos da selação anterior para gerar um novo conjunto de individuos misturando suas caracteristicas
        crossover_weight = random()
        wih_new = (crossover_weight * org_1.wih) + ((1 - crossover_weight) * org_2.wih)
        who_new = (crossover_weight * org_1.who) + ((1 - crossover_weight) * org_2.who)

        # Mutação: A mutação ocorre caso o valor de mutação do individuo seja menor que o limite.
        # A mutação fará com que uma das duas matrizes de peso da rede neural será substituído por um valor aleatório que esteja dentro de +/- 10% do valor original.
        # Isso criará a possibilidade de gerar um individuo mais apto, porém impedindo uma mutação brusca para não causar falhas as redes neurais.
        mutate = random()
        if mutate <= settings['mutate']:

            # Escolhe a matriz que será modificada
            mat_pick = randint(0,1)

            # Mutação: peso WIH
            if mat_pick == 0:
                index_row = randint(0,settings['hnodes']-1)
                wih_new[index_row] = wih_new[index_row] * uniform(0.9, 1.1)
                if wih_new[index_row] >  1: wih_new[index_row] = 1
                if wih_new[index_row] < -1: wih_new[index_row] = -1

            # Mutação: peso WHO
            if mat_pick == 1:
                index_row = randint(0,settings['onodes']-1)
                index_col = randint(0,settings['hnodes']-1)
                who_new[index_row][index_col] = who_new[index_row][index_col] * uniform(0.9, 1.1)
                if who_new[index_row][index_col] >  1: who_new[index_row][index_col] = 1
                if who_new[index_row][index_col] < -1: who_new[index_row][index_col] = -1

        organisms_new.append(organism(settings, wih=wih_new, who=who_new, name='gen['+str(gen)+']-org['+str(w)+']'))

    return organisms_new, stats


def simulate(settings, organisms, foods, gen):

    total_time_steps = int(settings['gen_time'] / settings['dt'])

    #--- Ciclo através de cada passo de tempo ---------------------+
    for t_step in range(0, total_time_steps, 1):

        # PLOT SIMULATION FRAME
        if settings['plot']==True and gen==settings['gens']-1:
            plot_frame(settings, organisms, foods, gen, t_step)

        # UPDATE FITNESS FUNCTION
        for food in foods:
            for org in organisms:
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # UPDATE FITNESS FUNCTION
                if food_org_dist <= 0.075:
                    org.fitness += food.energy
                    food.respawn(settings)

                # RESET DISTANCE AND HEADING TO NEAREST FOOD SOURCE
                org.d_food = 100
                org.r_food = 0

        # CALCULATE HEADING TO NEAREST FOOD SOURCE
        for food in foods:
            for org in organisms:

                # CALCULATE DISTANCE TO SELECTED FOOD PARTICLE
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE
                if food_org_dist < org.d_food:
                    org.d_food = food_org_dist
                    org.r_food = calc_heading(org, food)

        # GET ORGANISM RESPONSE
        for org in organisms:
            org.think()

        # UPDATE ORGANISMS POSITION AND VELOCITY
        for org in organisms:
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)

    return organisms


#--- Main ---#

def run(settings):

    #--- POPULATE THE ENVIRONMENT WITH FOOD ---------------+
    foods = []
    for i in range(0,settings['food_amo']):
        foods.append(food(settings))

    #--- POPULATE THE ENVIRONMENT WITH ORGANISMS ----------+
    organisms = []
    for i in range(0,settings['pop_size']):
        wih_init = np.random.uniform(-1, 1, (settings['hnodes'], settings['inodes']))     # mlp weights (input -> hidden)
        who_init = np.random.uniform(-1, 1, (settings['onodes'], settings['hnodes']))     # mlp weights (hidden -> output)

        organisms.append(organism(settings, wih_init, who_init, name='gen[x]-org['+str(i)+']'))

    #--- CYCLE THROUGH EACH GENERATION --------------------+
    for gen in range(0, settings['gens']):

        # SIMULATE
        organisms = simulate(settings, organisms, foods, gen)

        # EVOLVE
        organisms, stats = evolve(settings, organisms, gen)
        print('> GEN:',gen,'BEST:',stats['BEST'],'AVG:',stats['AVG'],'WORST:',stats['WORST'])

    pass


run(settings)