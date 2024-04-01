# Guardar los sismogramas en formato de texto ASCII
def escribir_sismogramas(sispressure, nt, nrec, DELTAT, t0):
    for irec in range(1, nrec + 1):
        nombre_archivo = "pressure_file_{:03d}.dat".format(irec)
        with open(nombre_archivo, "w") as archivo:
            for it in range(1, nt + 1):
                tiempo = sngl((dble(it - 1) * DELTAT - t0 + DELTAT / 2))
                presion = sngl(sispressure(it, irec))
                archivo.write("{:.6f} {:.6f}\n".format(tiempo, presion))

import numpy as np
import os

def create_color_image(image_data_2D, NX, NY, it, ISOURCE, JSOURCE, ix_rec, iy_rec, nrec,
                       NPOINTS_PML, USE_PML_XMIN, USE_PML_XMAX, USE_PML_YMIN, USE_PML_YMAX, field_number):
    # Display no lineal para mejorar las pequeñas amplitudes para gráficos
    POWER_DISPLAY = 0.30

    # Umbral de amplitud por encima del cual dibujamos el punto de color
    cutvect = 0.01

    # Usar fondo negro o blanco para puntos que están por debajo del umbral
    WHITE_BACKGROUND = True

    # Tamaño del cruce y del cuadrado en píxeles para representar la fuente y los receptores
    width_cross = 5
    thickness_cross = 1
    size_square = 3

    R, G, B = 0, 0, 0

    # Archivo de imagen y comando del sistema para convertir la imagen a un formato más conveniente
    if field_number == 1:
        nombre_archivo = "image{:06d}_Vx.pnm".format(it)
        comando_sistema = "convert image{:06d}_Vx.pnm image{:06d}_Vx.gif ; rm image{:06d}_Vx.pnm".format(it, it, it)
    elif field_number == 2:
        nombre_archivo = "image{:06d}_Vy.pnm".format(it)
        comando_sistema = "convert image{:06d}_Vy.pnm image{:06d}_Vy.gif ; rm image{:06d}_Vy.pnm".format(it, it, it)
    elif field_number == 3:
        nombre_archivo = "image{:06d}_pressure.pnm".format(it)
        comando_sistema = "convert image{:06d}_pressure.pnm image{:06d}_pressure.gif ; rm image{:06d}_pressure.pnm".format(it, it, it)

    with open(nombre_archivo, "w") as archivo:
        archivo.write("P3\n")
        archivo.write("{} {}\n".format(NX, NY))
        archivo.write("255\n")

        # Calcular amplitud máxima
        max_amplitud = np.max(np.abs(image_data_2D))

        # Imagen comienza en la esquina superior izquierda en formato PNM
        for iy in range(NY, 0, -1):
            for ix in range(1, NX + 1):
                # Definir datos como componente vectorial normalizada a [-1:1] y redondeada al entero más cercano
                valor_normalizado = image_data_2D[ix-1, iy-1] / max_amplitud

                # Suprimir valores que están fuera de [-1:+1] para evitar pequeños efectos de borde
                if valor_normalizado < -1:
                    valor_normalizado = -1
                if valor_normalizado > 1:
                    valor_normalizado = 1

                # Dibujar una cruz naranja para representar la fuente
                if (ix >= ISOURCE - width_cross and ix <= ISOURCE + width_cross and
                        iy >= JSOURCE - thickness_cross and iy <= JSOURCE + thickness_cross) or (
                        ix >= ISOURCE - thickness_cross and ix <= ISOURCE + thickness_cross and
                        iy >= JSOURCE - width_cross and iy <= JSOURCE + width_cross):
                    R, G, B = 255, 157, 0

                # Dibujar un marco negro de dos píxeles de grosor alrededor de la imagen
                elif ix <= 2 or ix >= NX - 1 or iy <= 2 or iy >= NY - 1:
                    R, G, B = 0, 0, 0

                # Mostrar bordes de las capas PML
                elif (USE_PML_XMIN and ix == NPOINTS_PML) or (USE_PML_XMAX and ix == NX - NPOINTS_PML) or (
                        USE_PML_YMIN and iy == NPOINTS_PML) or (USE_PML_YMAX and iy == NY - NPOINTS_PML):
                    R, G, B = 255, 150, 0

                # Suprimir todos los valores que están por debajo del umbral
                elif abs(image_data_2D[ix-1, iy-1]) <= max_amplitud * cutvect:
                    if WHITE_BACKGROUND:
                        R, G, B = 255, 255, 255
                    else:
                        R, G, B = 0, 0, 0

                # Representar puntos de imagen regulares usando rojo si el valor es positivo, azul si es negativo
                elif valor_normalizado >= 0:
                    R = int(255 * valor_normalizado ** POWER_DISPLAY)
                    G, B = 0, 0
                else:
                    R, G = 0, 0
                    B = int(255 * abs(valor_normalizado) ** POWER_DISPLAY)

                # Dibujar un cuadrado verde para representar los receptores
                for irec in range(nrec):
                    if (ix >= ix_rec[irec] - size_square and ix <= ix_rec[irec] + size_square and
                            iy >= iy_rec[irec] - size_square and iy <= iy_rec[irec] + size_square):
                        R, G, B = 30, 180, 60  # Usar color verde oscuro

                archivo.write("{} {} {}\n".format(R, G, B))

    # Llamar al sistema para convertir la imagen a GIF (se puede comentar si "call system" está ausente en su compilador)
    os.system(comando_sistema)

import numpy as np

# Flags para agregar capas PML a los bordes de la malla
USE_PML_XMIN = True
USE_PML_XMAX = True
USE_PML_YMIN = True
USE_PML_YMAX = True

# Número total de puntos de la malla en cada dirección de la malla
NX = 301
NY = 301

# Tamaño de una celda de la malla
DELTAX = 5.0
DELTAY = DELTAX

# Grosor de la capa PML en puntos de la malla
NPOINTS_PML = 10

# Velocidad de P y densidad
cp_unrelaxed = 2500.0
density = 2200.0

# Número total de pasos de tiempo
NSTEP = 1250

# Paso de tiempo en segundos
DELTAT = 0.0002  

# Parámetros para la fuente
f0 = 20.0
t0 = 0.0
factor = 1.0

# Fuente (en presión)
xsource = 750.0
ysource = 750.0
ISOURCE = int(xsource / DELTAX) + 1
JSOURCE = int(ysource / DELTAY) + 1

# Receptores
NREC = 1
xdeb = 2301.0  # primer receptor x en metros
ydeb = 2301.0  # primer receptor y en metros
xfin = 2301.0  # último receptor x en metros
yfin = 2301.0  # último receptor y en metros

# Información de visualización en la pantalla de vez en cuando
IT_DISPLAY = 100

# Valor de PI
PI = np.pi

# Cero
ZERO = 0.0

# Valor grande para máximo
HUGEVAL = 1.0e+30

# Umbral por encima del cual consideramos que el código se vuelve inestable
STABILITY_THRESHOLD = 1.0e+25

# Matrices principales
pressure_past = np.zeros((NX, NY))
pressure_present = np.zeros((NX, NY))
pressure_future = np.zeros((NX, NY))
pressure_xx = np.zeros((NX, NY))
pressure_yy = np.zeros((NX, NY))
dpressurexx_dx = np.zeros((NX, NY))
dpressureyy_dy = np.zeros((NX, NY))
kappa_unrelaxed = np.zeros((NX, NY))
rho = np.zeros((NX, NY))
Kronecker_source = np.zeros((NX, NY))

# Para interpolar parámetros del material o velocidad en la ubicación correcta en la celda de la malla escalonada
rho_half_x = 0.0
rho_half_y = 0.0

# Potencia para calcular el perfil d0
NPOWER = 2.0

# de las notas de clase no publicadas de Stephen Gedney para la clase EE699, conferencia 8, diapositiva 8-11
K_MAX_PML = 1.0
ALPHA_MAX_PML = 2.0 * PI * (f0 / 2.0)


# arrays for the memory variables
# could declare these arrays in PML only to save a lot of memory, but proof of concept only here
memory_dpressure_dx = np.zeros((NX, NY))
memory_dpressure_dy = np.zeros((NX, NY))
memory_dpressurexx_dx = np.zeros((NX, NY))
memory_dpressureyy_dy = np.zeros((NX, NY))

value_dpressure_dx = 0.0
value_dpressure_dy = 0.0
value_dpressurexx_dx = 0.0
value_dpressureyy_dy = 0.0

# 1D arrays for the damping profiles
d_x = np.zeros(NX)
K_x = np.ones(NX)
alpha_x = np.zeros(NX)
a_x = np.zeros(NX)
b_x = np.zeros(NX)
d_x_half = np.zeros(NX)
K_x_half = np.ones(NX)
alpha_x_half = np.zeros(NX)
a_x_half = np.zeros(NX)
b_x_half = np.zeros(NX)

d_y = np.zeros(NY)
K_y = np.ones(NY)
alpha_y = np.zeros(NY)
a_y = np.zeros(NY)
b_y = np.zeros(NY)
d_y_half = np.zeros(NY)
K_y_half = np.ones(NY)
alpha_y_half = np.zeros(NY)
a_y_half = np.zeros(NY)
b_y_half = np.zeros(NY)

thickness_PML_x = NPOINTS_PML * DELTAX
thickness_PML_y = NPOINTS_PML * DELTAY

Rcoef = 0.001
if NPOWER < 1:
    raise ValueError("NPOWER must be greater than 1")

d0_x = - (NPOWER + 1) * cp_unrelaxed * np.log(Rcoef) / (2.0 * thickness_PML_x)
d0_y = - (NPOWER + 1) * cp_unrelaxed * np.log(Rcoef) / (2.0 * thickness_PML_y)

print('d0_x =', d0_x)
print('d0_y =', d0_y)



d_x[:] = 0.0
d_x_half[:] = 0.0
K_x[:] = 1.0
K_x_half[:] = 1.0
alpha_x[:] = 0.0
alpha_x_half[:] = 0.0
a_x[:] = 0.0
a_x_half[:] = 0.0

d_y[:] = 0.0
d_y_half[:] = 0.0
K_y[:] = 1.0
K_y_half[:] = 1.0
alpha_y[:] = 0.0
alpha_y_half[:] = 0.0
a_y[:] = 0.0
a_y_half[:] = 0.0

# damping in the X direction

# origin of the PML layer (position of right edge minus thickness, in meters)
xoriginleft = thickness_PML_x
xoriginright = (NX - 1) * DELTAX - thickness_PML_x

for i in range(0, NX):

    # abscissa of current grid point along the damping profile
    xval = DELTAX * float(i - 1)

    # ---------- left edge
    if USE_PML_XMIN:

        # define damping profile at the grid points
        abscissa_in_PML = xoriginleft - xval
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_x
            d_x[i] = d0_x * abscissa_normalized**NPOWER
            K_x[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        # define damping profile at half the grid points
        abscissa_in_PML = xoriginleft - (xval + DELTAX / 2.0)
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_x
            d_x_half[i] = d0_x * abscissa_normalized**NPOWER
            K_x_half[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_x_half[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

    # ---------- right edge
    if USE_PML_XMAX:

        # define damping profile at the grid points
        abscissa_in_PML = xval - xoriginright
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_x
            d_x[i] = d0_x * abscissa_normalized**NPOWER
            K_x[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_x[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        # define damping profile at half the grid points
        abscissa_in_PML = xval + DELTAX / 2.0 - xoriginright
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_x
            d_x_half[i] = d0_x * abscissa_normalized**NPOWER
            K_x_half[i] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_x_half[i] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

    # just in case, for -5 at the end
    if alpha_x[i] < ZERO:
        alpha_x[i] = ZERO
    if alpha_x_half[i] < ZERO:
        alpha_x_half[i] = ZERO

    b_x[i] = np.exp(- (d_x[i] / K_x[i] + alpha_x[i]) * DELTAT)
    b_x_half[i] = np.exp(- (d_x_half[i] / K_x_half[i] + alpha_x_half[i]) * DELTAT)

    # this to avoid division by zero outside the PML
    if abs(d_x[i]) > 1.0e-6:
        a_x[i] = d_x[i] * (b_x[i] - 1.0) / (K_x[i] * (d_x[i] + K_x[i] * alpha_x[i]))
    if abs(d_x_half[i]) > 1.0e-6:
        a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (K_x_half[i] * (d_x_half[i] + K_x_half[i] * alpha_x_half[i]))

# damping in the Y direction

# origin of the PML layer (position of right edge minus thickness, in meters)
yoriginbottom = thickness_PML_y
yorigintop = (NY - 1) * DELTAY - thickness_PML_y

for j in range(0, NY):

    # abscissa of current grid point along the damping profile
    yval = DELTAY * float(j - 1)

    # ---------- bottom edge
    if USE_PML_YMIN:

        # define damping profile at the grid points
        abscissa_in_PML = yoriginbottom - yval
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_y
            d_y[j] = d0_y * abscissa_normalized**NPOWER
            K_y[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_y[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        # define damping profile at half the grid points
        abscissa_in_PML = yoriginbottom - (yval + DELTAY / 2.0)
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_y
            d_y_half[j] = d0_y * abscissa_normalized**NPOWER
            K_y_half[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_y_half[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

    # ---------- top edge
    if USE_PML_YMAX:

        # define damping profile at the grid points
        abscissa_in_PML = yval - yorigintop
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_y
            d_y[j] = d0_y * abscissa_normalized**NPOWER
            K_y[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_y[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

        # define damping profile at half the grid points
        abscissa_in_PML = yval + DELTAY / 2.0 - yorigintop
        if abscissa_in_PML >= ZERO:
            abscissa_normalized = abscissa_in_PML / thickness_PML_y
            d_y_half[j] = d0_y * abscissa_normalized**NPOWER
            K_y_half[j] = 1.0 + (K_MAX_PML - 1.0) * abscissa_normalized**NPOWER
            alpha_y_half[j] = ALPHA_MAX_PML * (1.0 - abscissa_normalized)

    b_y[j] = np.exp(- (d_y[j] / K_y[j] + alpha_y[j]) * DELTAT)
    b_y_half[j] = np.exp(- (d_y_half[j] / K_y_half[j] + alpha_y_half[j]) * DELTAT)

    # this to avoid division by zero outside the PML
    if abs(d_y[j]) > 1.0e-6:
        a_y[j] = d_y[j] * (b_y[j] - 1.0) / (K_y[j] * (d_y[j] + K_y[j] * alpha_y[j]))
    if abs(d_y_half[j]) > 1.0e-6:
        a_y_half[j] = d_y_half[j] * (b_y_half[j] - 1.0) / (K_y_half[j] * (d_y_half[j] + K_y_half[j] * alpha_y_half[j]))


# compute the Lame parameter and density
for j in range(0, NY):
    for i in range(1, NX):
        rho[i, j] = density
        kappa_unrelaxed[i, j] = density * cp_unrelaxed * cp_unrelaxed

# print position of the source
print('Position of the source:')
print('')
print('x =', xsource)
print('y =', ysource)
print('')

# define location of the source
Kronecker_source[:, :] = 0.0
Kronecker_source[ISOURCE, JSOURCE] = 1.0

# define location of receivers
print('There are', nrec, 'receivers')
print('')
if NREC > 1:
    myNREC = NREC
    xspacerec = (xfin - xdeb) / float(myNREC - 1)
    yspacerec = (yfin - ydeb) / float(myNREC - 1)
else:
    xspacerec = 0.0
    yspacerec = 0.0

# para receptores
xspacerec = 0.0
yspacerec = 0.0
distval = 0.0
dist = 0.0
ix_rec = np.zeros(NREC, dtype=int)
iy_rec = np.zeros(NREC, dtype=int)
xrec = np.zeros(NREC)
yrec = np.zeros(NREC)
myNREC = 0

for irec in range(0, nrec):
    xrec[irec] = xdeb + float(irec - 1) * xspacerec
    yrec[irec] = ydeb + float(irec - 1) * yspacerec

# find closest grid point for each receiver
for irec in range(0, nrec):
    dist = HUGEVAL
    for j in range(1, NY + 1):
        for i in range(1, NX + 1):
            distval = np.sqrt((DELTAX * float(i - 1) - xrec[irec])**2 + (DELTAY * float(j - 1) - yrec[irec])**2)
            if distval < dist:
                dist = distval
                ix_rec[irec] = i
                iy_rec[irec] = j
    print('receiver', irec, 'x_target,y_target =', xrec[irec], yrec[irec])
    print('closest grid point found at distance', dist, 'in i,j =', ix_rec[irec], iy_rec[irec])
    print('')

# check the Courant stability condition for the explicit time scheme
# R. Courant et K. O. Friedrichs et H. Lewy (1928)
Courant_number = cp_unrelaxed * DELTAT * np.sqrt(1.0 / DELTAX**2 + 1.0 / DELTAY**2)
print('Courant number is', Courant_number)
print('')
if Courant_number > 1.0:
    raise ValueError('time step is too large, simulation will be unstable')

# suppress old files (can be commented out if "call system" is missing in your compiler)
# call system('rm -f Vx_*.dat Vy_*.dat image*.pnm image*.gif')

# initialize arrays
pressure_present[:, :] = 0.0
pressure_past[:, :] = 0.0

# PML
memory_dpressure_dx[:, :] = 0.0
memory_dpressure_dy[:, :] = 0.0
memory_dpressurexx_dx[:, :] = 0.0
memory_dpressureyy_dy[:, :] = 0.0

# beginning of time loop
for it in range(0, NSTEP):
    # compute the first spatial derivatives divided by density
    for j in range(0, NY-1):
        for i in range(0, NX-2):
            value_dpressure_dx = (pressure_present[i + 1, j] - pressure_present[i, j]) / DELTAX

            memory_dpressure_dx[i, j] = b_x_half[i] * memory_dpressure_dx[i, j] + a_x_half[i] * value_dpressure_dx

            value_dpressure_dx = value_dpressure_dx / K_x_half[i] + memory_dpressure_dx[i, j]

            rho_half_x = 0.5 * (rho[i + 1, j] + rho[i, j])
            pressure_xx[i, j] = value_dpressure_dx / rho_half_x

    for j in range(0, NY-2):
        for i in range(0, NX-1):
            value_dpressure_dy = (pressure_present[i, j + 1] - pressure_present[i, j]) / DELTAY

            memory_dpressure_dy[i, j] = b_y_half[j] * memory_dpressure_dy[i, j] + a_y_half[j] * value_dpressure_dy

            value_dpressure_dy = value_dpressure_dy / K_y_half[j] + memory_dpressure_dy[i, j]

            rho_half_y = 0.5 * (rho[i, j + 1] + rho[i, j])
            pressure_yy[i, j] = value_dpressure_dy / rho_half_y

    # compute the second spatial derivatives
    for j in range(0, NY-1):
        for i in range(1, NX-1):
            value_dpressurexx_dx = (pressure_xx[i, j] - pressure_xx[i - 1, j]) / DELTAX

            memory_dpressurexx_dx[i, j] = b_x[i] * memory_dpressurexx_dx[i, j] + a_x[i] * value_dpressurexx_dx

            value_dpressurexx_dx = value_dpressurexx_dx / K_x[i] + memory_dpressurexx_dx[i, j]

            dpressurexx_dx[i, j] = value_dpressurexx_dx

    for j in range(1, NY-1):
        for i in range(0, NX-1):
            value_dpressureyy_dy = (pressure_yy[i, j] - pressure_yy[i, j - 1]) / DELTAY

            memory_dpressureyy_dy[i, j] = b_y[j] * memory_dpressureyy_dy[i, j] + a_y[j] * value_dpressureyy_dy

            value_dpressureyy_dy = value_dpressureyy_dy / K_y[j] + memory_dpressureyy_dy[i, j]

            dpressureyy_dy[i, j] = value_dpressureyy_dy

    # add the source (pressure located at a given grid point)
    a = np.pi * np.pi * f0 * f0
    t = float(it - 1) * DELTAT

    # Ricker source time function (second derivative of a Gaussian)
    source_term = factor * (1.0 - 2.0 * a * (t - t0)**2) * np.exp(-a * (t - t0)**2)

    # apply the time evolution scheme
    pressure_future[:, :] = -pressure_past[:, :] + 2.0 * pressure_present[:, :] + \
                             DELTAT * DELTAT * ((dpressurexx_dx[:, :] + dpressureyy_dy[:, :]) * kappa_unrelaxed[:, :] + \
                             4.0 * np.pi * cp_unrelaxed**2 * source_term * Kronecker_source[:, :])

    # apply Dirichlet conditions at the edges of the domain
    pressure_future[0, :] = 0.0  # Dirichlet condition for pressure on the left boundary
    pressure_future[NX - 1, :] = 0.0  # Dirichlet condition for pressure on the right boundary
    pressure_future[:, 0] = 0.0  # Dirichlet condition for pressure on the bottom boundary
    pressure_future[:, NY - 1] = 0.0  # Dirichlet condition for pressure on the top boundary

    # store seismograms
    for irec in range(0, NREC-1):
        sispressure[it - 1, irec - 1] = pressure_future[ix_rec[irec - 1], iy_rec[irec - 1]]

    # output information
    if it % IT_DISPLAY == 0 or it == 5:
        # print maximum of pressure and of norm of velocity
        pressurenorm = np.max(np.abs(pressure_future))
        print('Time step #', it, 'out of', NSTEP)
        print('Time:', float(it - 1) * DELTAT, 'seconds')
        print('Max absolute value of pressure =', pressurenorm)
        print('')
        # check stability of the code, exit if unstable
        if pressurenorm > STABILITY_THRESHOLD:
            raise ValueError('code became unstable and blew up')

        #crear_imagen_color(pressure_future, NX, NY, it, ISOURCE, JSOURCE, ix_rec, iy_rec, nrec,NPOINTS_PML, USE_PML_XMIN, USE_PML_XMAX, USE_PML_YMIN, USE_PML_YMAX, 3)

    # move new values to old values (the present becomes the past, the future becomes the present)
    pressure_past[:, :] = pressure_present[:, :]
    pressure_present[:, :] = pressure_future[:, :]
    plt.imshow(pressure_present)
    plt.show()


# guardar sismogramas
escribir_sismogramas(sispressure, NSTEP, NREC, DELTAT, t0)

print('')
print('Fin de la simulación')
print('')