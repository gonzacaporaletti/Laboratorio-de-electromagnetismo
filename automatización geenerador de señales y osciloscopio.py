
import pyvisa as visa
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import time



rm = visa.ResourceManager()


instrumentos = rm.list_resources()
print(instrumentos)

# Este string determina el intrumento que vamos a usar.
osci_name = 'USB0::0x0699::0x0363::C108012::INSTR'
gen_name = 'USB0::0x0699::0x0353::1603553::INSTR'


osci = rm.open_resource(osci_name)
my_gen = rm.open_resource(gen_name)




osci.query('*IDN?')
my_gen.query('*IDN?')

#%%

osci.write("DAT:SOU CH1")
xze, xin = osci.query_ascii_values('WFMPRE:XZE?;XIN?', separator=';')


xze, xin


osci.write("DAT:SOU CH1")
yze1, ymu1, yoff1 = osci.query_ascii_values('WFMPRE:YZE?;YMU?;YOFF?;', separator=';')



yze1, ymu1, yoff1



osci.write("DAT:SOU CH2")
yze2, ymu2, yoff2 = osci.query_ascii_values('WFMPRE:YZE?;YMU?;YOFF?;', separator=';')



# Modo de transmision: Binario
osci.write('DAT:ENC RPB')
osci.write('DAT:WID 1')



osci.write("DAT:SOU CH1")
data1 = osci.query_binary_values('CURV?', datatype='B', container=np.array)
print(data1.shape)


osci.write("DAT:SOU CH2")
data2 = osci.query_binary_values('CURV?', datatype='B', container=np.array)
print(data2.shape)

osci.query('MEASUrement:MEAS1:VAlUe?')
osci.query('MEASUrement:MEAS2:VAlUe?')


dato = float(my_gen.query('SOURCE1:FREQ?'))
my_gen.write('SOURCE1:FREQ 500')

#%%

#barrido y recoleccion de datos automática.

frecuencias = np.geomspace(1,100e3,100)

mis_valores = []
Vpp_filtrado=[]
Vpp_sinfiltral=[]
for i, frec in enumerate(frecuencias):
    my_gen.write(f'SOURCE1:FREQ {frec}')
    tb=(1/(4*frec))
    osci.write(f'HORizontal:MAIn:SCAle {tb}')
    espera=14*tb
    time.sleep(espera)
    osci.write('MEASUrement:IMMed:TYPe PK2pk')
    osci.write('MEASUrement:IMMed:SOUrce CH1')
    Vpp_filtrado.append(float(osci.query('MEASUrement:IMMed:VAlUe?')))
    osci.write('MEASUrement:IMMed:SOUrce CH2')
    Vpp_sinfiltral.append(float(osci.query('MEASUrement:IMMed:VAlUe?')))
    mis_valores.append(frec)

f=np.array(mis_valores)
V0=np.array(Vpp_filtrado)
Vi=np.array(Vpp_sinfiltral)

plt.plot(f, (V0/Vi))
# data1

#%% Grafico

tiempo = xze + np.arange(len(data2)) * xin

data1v = (data1 - yoff1) * ymu1 + yze1
data2v = (data2 - yoff2) * ymu2 + yze2

plt.figure('Grafico')
plt.plot(tiempo, data1v, color='black')
plt.plot(tiempo, data2v, color= 'blue', ls='dotted')
plt.grid()
plt.xlabel('Tiempo [s]')
plt.ylabel('Voltaje [V]')

#%% Guardo

## exportar la medición como archivo de datos: tiempo, CH1, CH2
salida=np.column_stack((tiempo,data1v,data2v))
np.savetxt("datos_osci.csv", salida, delimiter = ",")


## importar medición vieja: tiempo, CH1, CH2
datos=np.loadtxt('datos_osci.csv',delimiter=',')
tiempo=datos[:,0]
CH1=datos[:,1]
CH2=datos[:,2]

#%%ajuste
plt.rcParams["figure.figsize"] = (11,6)
plt.rcParams.update({'font.size': 14})

def exp (r, c):
    exp

def ajustar(func, x, y, yerr=None, p0=None):
    """Ajusta la función `func` a los datos `x` e `y`,
    ponderando opcionalmente por `yerr`.

    Devuelve los parametros optimos y sus incertezas.
    """
    if yerr is not None:
        # Si pasamos un numero, en lugar de un arrray,
        # lo convierte al tamaño del array.
        yerr = np.broadcast_to(yerr, y.shape)
    # Realizamos el ajuste ponderado.
    p, cov = curve_fit(func, x, y, sigma=yerr, p0=p0, maxfev= 200000)
    # Los errores de los parametros son la raiz de la diagonal de `cov`.
    p_err = np.sqrt(np.diag(cov))
    return p, p_err

def graficar_ajuste(x, y, func, params, *, xerr=None, yerr=None, x_eval=None, colorD=None, colorA=None, ylabel=None, xlabel=None):
    """Grafica las mediciones, el ajuste y los residuos.

    x, y : array de mediciones
    func : funcion ajustada
    params : parametros optimos
    xerr, yerr : errores en x e y (opcionales)
    x_eval : valores de x para graficar el modelo (opcional)
    """
    if x_eval is None:
        # Si no elegimos (otros) puntos para graficar el modelo,
        # usamos los x medidos.
        x_eval = x

    if colorD is None:
        # Si no elegimos (otros) puntos para graficar el modelo,
        # usamos los x medidos.
        colorD = 'black'

    if colorA is None:
        # Si no elegimos (otros) puntos para graficar el modelo,
        # usamos los x medidos.
        colorA = 'red'
    plt.figure()
    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    # Datos
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=5, color=colorD)
    # Modelo
    plt.plot(x_eval, func(x_eval, *params), color=colorA)

    plt.grid(True)
    plt.show()


