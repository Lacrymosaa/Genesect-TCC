import random
from deap import base, creator, tools

class Ser:
    def __init__(self, codigo_genetico, x, y):
        self.codigo_genetico = codigo_genetico
        self.x = x
        self.y = y

class Ambiente:
    def __init__(self, tamanho):
        self.tamanho = tamanho #Tamanho em largura e altura é o mesmo
        self.zona_reproducao = None #Instância a zona de reprodução dos seres
        self.seres = [] #Cria a lista de seres da classe

    def criar_seres_iniciais(self, quantidade):
        self.seres = [] #Usa a lista como inicial
        for _ in range(quantidade):
            codigo_genetico = CodigoGenetico()
            x, y = self.gerar_posicao_aleatoria()
            ser = Ser(codigo_genetico, x, y)
            self.seres.append(ser)

    def gerar_posicao_aleatoria(self):
        while True:
            x = random.randint(0, self.tamanho)
            y = random.randint(0, self.tamanho)
            if not self.verificar_colisao(x, y):
                return x, y

    def verificar_colisao(self, x, y):
        for ser in self.seres:
            if ser.x == x and ser.y == y:
                return True
        return False
    
    def definir_zona_reproducao(self, x, y, largura, altura):
        self.zona_reproducao = (x, y, largura, altura)
        # O X e Y são os pontos do canto superior esquerdo da área

    def seres_na_zona_reproducao(self):
        seres_na_zona = []
        for ser in self.seres:
            if self.zona_reproducao is not None:
                x, y, largura, altura = self.zona_reproducao
                if x <= ser.x < x + largura and y <= ser.y < y + altura:
                    seres_na_zona.append(ser)
        return seres_na_zona


class CodigoGenetico:
    def __init__(self):
        self.codigo = self.gerar_codigo_genetico()

    def gerar_codigo_genetico(self):
        codigo = ""
        for _ in range(30):
            codigo += random.choice("ABCDEF")
        return codigo

    def mutacao(self):
        for i in range(len(self.codigo)):
            if random.random() < 1/128:
                novo_valor = random.choice("ABCDEF")
                self.codigo = self.codigo[:i] + novo_valor + self.codigo[i+1:]

