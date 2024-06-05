# -*- coding: utf-8 -*-

# Sergio Gonzalez Montero

import numpy as np
from numpy import cos as cos, sin as sin, pi as pi
import matplotlib.pyplot as plt
from matplotlib import animation


def sph2cart(phi,theta):
    """
    phi : float; longitud de [0, 2*pi)
    theta : float; latitud de [-pi/2, pi/2]
    De coordenadas esfericas a cartesianas
    """
    x = cos(theta)*cos(phi)
    y = cos(theta)*sin(phi)
    z = sin(theta)
    
    tocart = np.array([x,y,z])
    return tocart

def cartTgPlanePt(o, o_phi, o_theta, v_phi, v_theta):
    """
    o : lst; punto en cartesianas
    o_phi : float; longitud de o
    o_theta : float; latitud de o
    v_phi, v_theta : float; componentes tangenciales
    Coordenadas cartesianas del punto o en el plano tangente
    """
    
    a = np.array([-cos(o_theta)*sin(o_phi),
                   cos(o_theta)*cos(o_phi),
                   0])
    b = np.array([-sin(o_theta)*cos(o_phi),
                   -sin(o_theta)*sin(o_phi),
                   cos(o_theta)])
    cart_coord = o + v_phi * a + v_theta * b
    
    return cart_coord

#Familia parametrica
def paramFam(t,phi,theta0,v02):
    """
    t : float; parametro temporal
    phi : float; angulo acimutal
    theta0 : float; angulo polar
    Familia paramétrica dependiente de t^2
    """
    p_phi = v02/cos(theta0) * sin(sin(theta0)*phi*t**2)
    p_theta = v02 * cos(sin(theta0)*phi*t**2)
    
    return p_phi, p_theta


def animate(t, theta01, theta02, v02, xsph, ysph, zsph):
    ax = plt.axes(projection = '3d')
    
    # Vector blanco
    o_phi1 = 2*pi*t**2
    o_theta1 = theta01
    v_phi1,v_theta1 = paramFam(t, 2*pi, theta01, c0)
    
    o1 = sph2cart(o_phi1,o_theta1)
    p1 = cartTgPlanePt(o1, o_phi1,o_theta1,v_phi1,v_theta1)
    X1,Y1,Z1,U1,V1,W1 = np.concatenate((o1,p1-o1))
    
    # Vector rojo
    o_phi2 = 2*pi*t**2
    o_theta2 = theta02
    v_phi2,v_theta2 = paramFam(t,2*pi,theta02,c0)
    
    o2 = sph2cart(o_phi2,o_theta2)
    p2 = cartTgPlanePt(o2, o_phi2,o_theta2,v_phi2,v_theta2)
    X2,Y2,Z2,U2,V2,W2 = np.concatenate((o2,p2-o2))
    
    # Esfera y vectores
    ax.plot_surface(xsph, ysph, zsph, cmap='winter',
                    edgecolor='none',alpha=0.5)
    ax.quiver(X1,Y1,Z1,U1,V1,W1,colors="white", zorder=3,
              arrow_length_ratio=0.4)
    ax.quiver(X2,Y2,Z2,U2,V2,W2,colors="red", zorder=3,
              arrow_length_ratio=0.4)
   
    ax.set_axis_off()
    ax.set_facecolor('black')
    
    
# Sistema de referencia
phi = np.linspace(0, 2*pi, 3000)
theta = np.linspace(-pi/2, pi/2, 2000)
r = 1
x = r*np.outer(cos(theta), cos(phi))
y = r*np.outer(cos(theta), sin(phi))
z = r*np.outer(sin(theta), np.ones_like(phi))


# Animacion
theta01 = 0
theta02 = pi/6
v02 = pi/5
c0 = np.sqrt(pi/5)

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig,animate,
                              frames=np.arange(0,2.01,0.025),
                              fargs=(theta01,theta02,v02,x,y,z), 
                              interval=20)
ani.save('transporte paralelo.gif',fps=10)
plt.show()

