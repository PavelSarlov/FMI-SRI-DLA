#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2021/2022
#############################################################################
###
### Домашно задание 2
###
#############################################################################

import numpy as np

#############################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def lossAndGradient(u_w, Vt, v):
    ###  Векторът u_w е влагането на целевата дума. shape(u_w) = M.
    ###  Матрицата Vt представя влаганията на контекстните думи. shape(Vt) = (n+1)xM.
    ###  Векторът v е параметър. shape(v) = M.
    ###  Първият ред на Vt е влагането на коректната контекстна дума, а
    ###  следващите n реда са влаганията на извадката от негативни контекстни думи
    ###
    ###  функцията връща J -- загубата в тази точка;
    ###                  du_w -- градиентът на J спрямо u_w;
    ###                  dVt --  градиентът на J спрямо Vt;
    ###                  dv -- градиентът на J спрямо v.
    #############################################################################
    #### Забележка: За по-добра числена стабилност използвайте np.tanh,
    ####            вместо сами да имплементирате хиперболичен тангенс.
    #### Начало на Вашия код. На мястото на pass се очакват 7-15 реда
    
    print(u_w.shape, Vt.shape, v.shape)
    delta = np.array([1 if x==0 else 0 for x in range(Vt.shape[0])])
    dot = np.dot(delta, Vt)

    J = -(np.log(sigmoid(np.dot(v, np.tanh(u_w + dot)))) 
            +np.sum(np.log(sigmoid(-np.dot(v, np.tanh(u_w + Vt).T))))
            -np.log(sigmoid(-np.dot(v, np.tanh(u_w + dot)))))
    du_w = -((1-sigmoid(np.dot(v, np.tanh(u_w + dot))))*(np.dot(v, (1-np.tanh(u_w + dot)**2)))
            +np.sum((1-sigmoid(-np.dot(v, np.tanh(u_w + Vt).T)))*(-np.dot(v, (1-np.tanh(u_w + Vt).T**2))))
            -(1-sigmoid(-np.dot(v, np.tanh(u_w + dot))))*(-np.dot(v, (1-np.tanh(u_w + dot)**2))))
    dVt = np.ones(5)
    dv = -((1-sigmoid(np.dot(v, np.tanh(u_w + dot))))*np.tanh(u_w + dot)
            +np.sum((1-sigmoid(-np.dot(v, np.tanh(u_w + Vt).T)))*(-np.tanh(u_w + Vt).T), 1)
            -(1-sigmoid(-np.dot(v, np.tanh(u_w + dot))))*(-np.tanh(u_w + dot)))
    
    #### Край на Вашия код
    #############################################################################

    return J, du_w, dVt, dv


def lossAndGradientCumulative(u_w, Vt, v):
    ###  Изчисляване на загуба и градиент за цяла партида
    ###  Тук за всяко от наблюденията се извиква lossAndGradient
    ###  и се акумулират загубата и градиентите за S-те наблюдения
    Cdu_w = []
    CdVt = []
    Cdv = []
    CJ = 0
    S = u_w.shape[0]
    for i in range(S):
        J, du_w, dVt, dv = lossAndGradient(u_w[i],Vt[i], v)
        Cdu_w.append(du_w/S)
        CdVt.append(dVt/S)
        Cdv.append(dv/S)
        CJ += J/S
    return CJ, Cdu_w, CdVt, Cdv


def lossAndGradientBatched(u_w, Vt, v):
    ###  Изчисляване на загуба и градиент за цяла партида.
    ###  Тук едновременно се изчислява загубата и градиентите за S наблюдения.
    ###  Матрицата u_w представя влаганията на целевите думи и shape(u_w) = SxM.
    ###  Тензорът Vt представя S матрици от влагания на контекстните думи и shape(Vt) = Sx(n+1)xM.
    ###  Параметричният вектор v; shape(v) = M.
    ###  Във всяка от S-те матрици на Vt в първия ред е влагането на коректната контекстна дума, а
    ###  следващите n реда са влаганията на извадката от негативни контекстни думи.
    ###
    ###  Функцията връща J -- загубата за цялата партида;
    ###                  du_w -- матрица с размерност SxM с градиентите на J спрямо u_w за всяко наблюдение;
    ###                  dVt --  с размерност Sx(n+1)xM -- S градиента на J спрямо Vt;
    ###                  dv -- с размерност SxM -- S градиента на J спрямо v.
    #############################################################
    ###  От вас се очаква вместо да акумулирате резултатите за отделните наблюдения,
    ###  да използвате тензорни операции, чрез които наведнъж да получите
    ###  резултата за цялата партида. Очаква се по този начин да получите над 2 пъти по-бързо изпълнение.
    #############################################################

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-20 реда
    
    pass
    
    #### Край на Вашия код
    #############################################################################
    return J, du_w, dVt, dv
    