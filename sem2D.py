import numpy as np               # Ofrece funciones para operaciones basicas con arreglos
import scipy.linalg as lin       # Ofrece funciones para operaciones de algebra lineal

#=================================================
# Funciones traducidas y adaptadas de FSELIB
# http://dehesa.freeshell.org/FSELIB/
#=================================================

def lobatto(i=6):

#=================================================
# Zeros of the ith-degree Lobatto polynomial
#
# This table contains values for i = 1, 2, ..., 6
# The default value is i=6
#=================================================

    if(i>6):
        print('Data is not available; Will take i=6')
        i=6

    Z = np.zeros(i)

    if(i==1):
        Z[0] = 0.0

    elif(i==2):
        Z[0] = -1.0/np.sqrt(5.0)
        Z[1] = -Z[0]

    elif(i==3):
        Z[0] = -np.sqrt(3.0/7.0)
        Z[1] = 0.0
        Z[2] = -Z[0]

    elif(i==4):
        Z[0] = -0.76505532392946
        Z[1] = -0.28523151648064
        Z[2] = -Z[1]
        Z[3] = -Z[0]

    elif(i==5):
        Z[0] = -0.83022389627857
        Z[1] = -0.46884879347071
        Z[2] = 0.0
        Z[3] = -Z[1]
        Z[4] = -Z[0]

    elif(i==6):
        Z[0] = -0.87174014850961
        Z[1] = -0.59170018143314
        Z[2] = -0.20929921790248
        Z[3] = -Z[2]
        Z[4] = -Z[1]
        Z[5] = -Z[0]

    return Z

def qlobatto(i=6,r='b'):

#=================================================
# Zeros of the completed ith-degree Lobatto
# polynomial, and corresponding weights for the
# Lobatto integration quadrature
#
# This table contains values for i = 1, 2, ..., 6
# The default value is i=6
#=================================================

    if(i>6):
        print('Data is not available; Will take i=6')
        i=6

    Z = np.zeros(i+2)
    W = np.zeros(i+2)

    Z[0] = -1.0
    Z[-1] = -Z[0]
    W[0] = 2.0/((i+1)*(i+2))
    W[-1] = W[0]

    if(i==1):
	Z[1] = 0.0
	W[1] = 4.0/3.0

    elif(i==2):
	Z[1] = -1.0/np.sqrt(5.0)
	Z[2] = -Z[1]
	W[1] = 5.0/6.0
	W[2] = W[1]

    elif(i==3):
	Z[1] = -np.sqrt(3.0/7.0)
	Z[2] = 0.0
	Z[3] = -Z[1]
	W[1] = 49.0/90.0
	W[2] = 32.0/45.0
	W[3] = W[1]

    elif(i==4):
	Z[1] = -0.76505532392946
	Z[2] = -0.28523151648064
	Z[3] = -Z[2]
	Z[4] = -Z[1]
	W[1] = 0.37847495629785
	W[2] = 0.55485837703549
	W[3] = W[2]
	W[4] = W[1]

    elif(i==5):
	Z[1] = -0.83022389627857
	Z[2] = -0.46884879347071
	Z[3] = 0.0
	Z[4] = -Z[2]
	Z[5] = -Z[1]
	W[1] = 0.27682604736157
	W[2] = 0.43174538120986
	W[3] = 0.48761904761905
	W[4] = W[2]
	W[5] = W[1]

    elif(i==6):

	Z[1] = - 0.87174014850961
	Z[2] = - 0.59170018143314 
	Z[3] = - 0.20929921790248
	Z[4] = -Z[3]
	Z[5] = -Z[2]
	Z[6] = -Z[1]
	W[1] = 0.21070422714350
	W[2] = 0.34112269248350
	W[3] = 0.41245879465870
	W[4] = W[3]
	W[5] = W[2]
	W[6] = W[1]

    if(r=='z'):
	return Z
    elif(r=='w'):
	return W
    else:
	return Z, W

def gaussTri(m):
    
#================================================
# Abscissas (xi, eta) and weights (w)
# for Gaussian integration over a flat triangle
# in the xi-eta plane
#
# Integration is performed with respect
# to the triangle barycentric coordinates
#
# SYMBOLS:
# -------
#
# m: order of the quadrature
#    choose from 1,3,4,6,7,9,12,13
#    Default value is 7
#================================================

    #------------------------
    # Checks for valid values
    #------------------------
    if((m != 1) and (m != 3) and (m != 4) and (m != 6) and (m != 7)
       and (m != 9) and (m != 12) and (m != 13)):
        m = 7

    if(m==1):
        xi = np.array([1.0/3.0])
        eta = np.array([1.0/3.0])
        w = np.array([1.0])


    elif(m==3):
        xi = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0])
        eta = np.array([1.0/6.0, 1.0/6.0, 2.0/3.0])
        w = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
    
    elif(m==4):
        xi = np.array([1.0/3.0, 1.0/5.0, 3.0/5.0, 1.0/5.0])
        eta = np.array([1.0/3.0, 1.0/5.0, 1.0/5.0, 3.0/5.0])
        w = np.array([-27.0/48.0, 25.0/48.0, 25.0/48.0, 25.0/48.0])

    elif(m==6):
        al = 0.816847572980459
        be = 0.445948490915965
        ga = 0.108103018168070
        de = 0.091576213509771
        o1 = 0.109951743655322
        o2 = 0.223381589678011
        xi = np.array([de, al, de, be, ga, be])
        eta = np.array([de, de, al, be, be, ga])
        w = np.array([o1, o1, o1, o2, o2, o2])

    elif(m==7):
        al = 0.797426958353087
        be = 0.470142064105115
        ga = 0.059715871789770
        de = 0.101286507323456
        o1 = 0.125939180544827
        o2 = 0.132394152788506
        xi = np.array([de, al, de, be, ga, be, 1.0/3.0])
        eta = np.array([de, de, al, be, be, ga, 1.0/3.0])
        w = np.array([o1, o1, o1, o2, o2, o2, 0.225])

    elif(m==9):
        al = 0.124949503233232
        qa = 0.165409927389841
        rh = 0.797112651860071
        de = 0.437525248383384
        ru = 0.037477420750088
        o1 = 0.205950504760887
        o2 = 0.063691414286223
        xi = np.array([de, al, de, qa, ru, rh, qa, ru, rh])
        eta = np.array([de, de, al, ru, qa, qa, rh, rh, ru])
        w = np.array([o1, o1, o1, o2, o2, o2, o2, o2, o2])

    elif(m==12):
        al = 0.873821971016996
        be = 0.249286745170910
        ga = 0.501426509658179
        de = 0.063089014491502
        rh = 0.636502499121399
        qa = 0.310352451033785
        ru = 0.053145049844816
        o1 = 0.050844906370207
        o2 = 0.116786275726379
        o3 = 0.082851075618374
        xi = np.array([de, al, de, be, ga, be, qa, ru, rh, qa, ru, rh])
        eta = np.array([de, de, al, be, be, ga, ru, qa, qa, rh, rh, ru])
        w = np.array([o1, o1, o1, o2, o2, o2, o3, o3, o3, o3, o3, o3])

    elif(m==13):
        al = 0.479308067841923
        be = 0.065130102902216
        ga = 0.869739794195568
        de = 0.260345966079038
        rh = 0.638444188569809
        qa = 0.312865496004875
        ru = 0.048690315425316
        o1 = 0.175615257433204
        o2 = 0.053347235608839
        o3 = 0.077113760890257
        o4 =-0.149570044467670
        xi = np.array([de, al, de, be, ga, be, qa, ru, rh, qa, ru, rh, 1.0/3.0])
        eta = np.array([de, de, al, be, be, ga, ru, qa, qa, rh, rh, ru, 1.0/3.0])
        w = np.array([o1, o1, o1, o2, o2, o2, o3, o3, o3, o3, o3, o3, o4])

    return xi, eta, w

#=================================================
# Funciones nuevas implementadas
#=================================================

def proriol(k,l,xi,eta):
    #=================================================
    # Evalua los polinomios de Proriol usando las
    # formas precalculadas
    # 
    # Admite polinomios de orden hasta 3: k+l < 4
    #=================================================
    
    if(k==0 and l==0):
        val = 1
    elif(k==1 and l==0):
        val = 2*xi+eta -1
    elif(k==0 and l==1):
        val = 3*eta -1
    elif(k==2 and l==0):
        val = 6*xi**2+6*xi*eta+eta**2 -6*xi-2*eta +1
    elif(k==1 and l==1):
        val = (2*xi+eta -1)*(5*eta -1)
    elif(k==0 and l==2):
        val = 10*eta**2 -8*eta +1
    elif(k==3 and l==0):
        val = (2*xi+eta -1)*(10*xi**2+10*xi*eta+eta**2 -10*xi-2*eta +1)
    elif(k==2 and l==1):
        val = (6*xi**2+6*xi*eta+eta**2 -6*xi-2*eta +1)*(7*eta-1)
    elif(k==1 and l==2):
        val = (2*xi+eta -1)*(21*eta**2 -12*eta +1)
    elif(k==0 and l==3):
        val = 35*eta**3 -45*eta**2 +15*eta -1
    else:
        raise NameError('La cantidad de elementos no coincide con el orden')

    return val

def dproriol(k,l,xi,eta,var='xi'):
    #=================================================
    # Evalua las derivadas de los polinomios de
    # Proriol usando las formas precalculadas
    # 
    # Admite polinomios de orden hasta 3: k+l < 4
    #=================================================
    
    if(var=='xi'):
        if(k==0 and l==0):
            val = 0
        elif(k==1 and l==0):
            val = 2
        elif(k==0 and l==1):
            val = 0
        elif(k==2 and l==0):
            val = 12*xi+6*eta-6
        elif(k==1 and l==1):
            val = 2*(5*eta -1)
        elif(k==0 and l==2):
            val = 0
        elif(k==3 and l==0):
            val = 2*(10*xi**2+10*xi*eta+eta**2 -10*xi-2*eta +1)+(2*xi+eta -1)*(20*xi+10*eta-10)
        elif(k==2 and l==1):
            val = (12*xi+6*eta-6)*(7*eta-1)
        elif(k==1 and l==2):
            val = 2*(21*eta**2 -12*eta +1)
        elif(k==0 and l==3):
            val = 0
        else:
            raise NameError('La cantidad de elementos no coincide con el orden')
    elif(var=='eta'):
        if(k==0 and l==0):
            val = 0
        elif(k==1 and l==0):
            val = 1
        elif(k==0 and l==1):
            val = 3
        elif(k==2 and l==0):
            val = 6*xi+2*eta-2
        elif(k==1 and l==1):
            val = (5*eta -1)+(2*xi+eta -1)*5
        elif(k==0 and l==2):
            val = 20*eta-8
        elif(k==3 and l==0):
            val = (10*xi**2+10*xi*eta+eta**2 -10*xi-2*eta +1)+(2*xi+eta -1)*(10*xi+2*eta-2)
        elif(k==2 and l==1):
            val = (6*xi+2*eta-2)*(7*eta-1)+(6*xi**2+6*xi*eta+eta**2 -6*xi-2*eta +1)*7
        elif(k==1 and l==2):
            val = (21*eta**2 -12*eta +1)+(2*xi+eta -1)*(42*eta-12)
        elif(k==0 and l==3):
            val = 105*eta**2-90*eta +15
        else:
            raise NameError('La cantidad de elementos no coincide con el orden')
    else:
        raise NameError('Variable de derivacion invalida')

    return val

def genVDM(m):
    #=================================================
    # Evalua la matriz de Vandermonde generalizada, su
    # determinante y la matriz de cofactores
    # 
    # Admite polinomios de orden hasta 3: k+l < 4
    #=================================================
    
    # Recupera las posiciones estandar de los nodos
    npe = (m+1)*(m+2)/2
    distr = 0.5+0.5*lobatto(m-1)
    xist = np.zeros(npe)
    etast = np.zeros(npe)

    xist[0:3*m] = np.concatenate([[0],distr,[1],distr[::-1],np.zeros(m)])
    etast[0:3*m] = np.concatenate([np.zeros(m+1),distr,[1],distr[::-1]])
    if(m==3):
            xist[9] = 1./3.
            etast[9] = 1./3.
    
    # Inicializa las matrices requeridas para el calculo
    Mat_VDM = np.zeros([npe,npe])
    Mat_Cof = np.zeros([npe,npe])
    Mat_Min = np.zeros([npe-1,npe-1])
    prind = np.array([[0,1,0,2,1,0,3,2,1,0],
                      [0,0,1,0,1,2,0,1,2,3]])

    # Construye la matriz de Vandermonde generalizada
    for i in range(npe):
        for j in range(npe):
            k = prind[0,i]
            l = prind[1,i]
            Mat_VDM[i,j] = proriol(k,l,xist[j],etast[j])

    Det_VDM = lin.det(Mat_VDM)

    for i in range(npe):
        for j in range(npe):
            Temp = np.delete(Mat_VDM,i,0)
            Mat_Min = np.delete(Temp,j,1)
            Mat_Cof[i,j] = (-1)**(i+j)*np.linalg.det(Mat_Min)        
    return Mat_VDM, Det_VDM, Mat_Cof

def genNodes(Ne,Np,px,py,ordn='xy'):

#=================================================
# Genera los nodos de interpolacion segun el
# numero de elementos y sus ordenes. Esta es la
# numeracion local de los nodos: (l,i)
# 
# Ne = Np.size
# max(Np) < 6
#=================================================

    ne = Ne[0]*Ne[1]   # Recupera el numero de elementos
    npe = (Np[0]+1)*(Np[1]+1)   # Recupera el numero de nodos por elemento
    x = np.zeros([ne,npe])   # Coordenadas x por elemento
    y = np.zeros([ne,npe])   # Coordenadas y por elemento

    # Genera los nodos de interpolacion de Lobatto
    dx = qlobatto(Np[0]-1,'z')
    dy = qlobatto(Np[1]-1,'z')

    for l in range(ne):
        cx = l % Ne[0]
        cy = l / Ne[0]
        midx = 0.5 * (px[cx+1]+px[cx])
        disx = 0.5 * (px[cx+1]-px[cx])
        midy = 0.5 * (py[cy+1]+py[cy])
        disy = 0.5 * (py[cy+1]-py[cy])
    
        pxe = midx+disx*dx
        pye = midy+disy*dy
        (xv,yv) = np.meshgrid(pxe,pye,indexing=ordn)
        x[l,:] = xv.reshape(npe)
        y[l,:] = yv.reshape(npe)
    return x, y, ne, npe

def genNodesTri(Ne,m,px,py):

#=================================================
# Genera los nodos de interpolacion segun el
# numero de elementos y sus ordenes para una malla
# triangular. Esta es la numeracion local de los
# nodos: (l,i)
# 
# m < 4
#=================================================

    ne = 2*Ne[0]*Ne[1]   # Recupera el numero de elementos
    npe = (m+1)*(m+2)/2   # Este es el numero de nodos por elemento
    x = np.zeros([ne,npe])   # Coordenadas x por elemento
    y = np.zeros([ne,npe])   # Coordenadas y por elemento

    # Genera los nodos de interpolacion de Lobatto
    distr = lobatto(m-1)

    for l in range(ne/2):   # Desarrolla dos elementos por iteracion
        cx = l % Ne[0]
        cy = l / Ne[0]

        pxe = 0.5 * ((px[cx+1]+px[cx])+(px[cx+1]-px[cx])*distr)
        pye = 0.5 * ((py[cy+1]+py[cy])+(py[cy+1]-py[cy])*distr)

        # Elemento inferior

        x[2*l,0:3*m] = np.concatenate([[px[cx]],pxe,[px[cx+1]],pxe[::-1],
                                       px[cx]*np.ones(m)])
        y[2*l,0:3*m] = np.concatenate([py[cy]*np.ones(m+1),
                                       pye,[py[cy+1]],pye[::-1]])
        if(m==3):
            x[2*l,9] = (2.*px[cx]+px[cx+1])/3.
            y[2*l,9] = (2.*py[cy]+py[cy+1])/3.

        # Elemento superior

        x[2*l+1,0:3*m] = np.concatenate([[px[cx+1]],pxe[::-1],[px[cx]],pxe,
                                         px[cx+1]*np.ones(m)])
        y[2*l+1,0:3*m] = np.concatenate([[py[cy]],pye,py[cy+1]*np.ones(m+1),
                                         pye[::-1]])
        if(m==3):
            x[2*l+1,9] = (2.*px[cx+1]+px[cx])/3.
            y[2*l+1,9] = (2.*py[cy+1]+py[cy])/3.
            
    return x, y, ne, npe

def genConn(Ne,Npm,ne,npe,x,y,L,W,U0,h,tipo='quad'):

#=================================================
# Genera los vectores de numeracion global, la
# matriz de conectividad y el vector de frontera
#
# max(Np) < 6 (quad) o m < 4 (tri)
#=================================================

    # Calcula el numero de nodos globales para definir el tamano de las matrices contenedoras
    if(tipo == 'quad'):
        ng = ne*npe - ((Ne[0]-1)*(Npm[1]+1)+(Ne[1]-1)*(Npm[0]+1)
                       +(Ne[0]-1)*(Ne[1]-1)*(sum(Npm)+1))
    elif(tipo == 'tri'):
        ng = ne*npe - ((ne/2+Ne[0]+Ne[1]-2)*(Npm+1)+(Ne[0]-1)*(Ne[1]-1)*(2*Npm+1))    

    coords = np.zeros([ng,2])   # Almacena las coordenadas de todos los nodos sin redundancia
    C = np.zeros((ne,npe),int)   # Matriz de conectividad
    gfl = np.zeros([ng,3])   # Indicador de nodos de frontera [indicador, vel. x, vel. y]

    # Se introduce manualmente el primer nodo
    coords[0,0] = x[0,0]
    coords[0,1] = y[0,0]
    # C[0,0] = 0 Esta condicion ya se tiene de la inicializacion de C
    gfl[0,0] = 1   # Indica un nodo de frontera (el valor de la velocidad es de [0,0])

    cont = 1   # Contador global
    tol = h*1e-6   # Tolerancia para identificacion de nodos iguales
    for l in range(ne):
        for k in range(npe):
            existe = False
            for n in range(cont):
                if(np.abs(coords[n,0]-x[l,k])+np.abs(coords[n,1]-y[l,k]) < tol):
                    existe = True
                    C[l,k] = n   # El nodo (l,m) corresponde al nodo global n
            if(not existe):   # Registra el nodo si no existe todavia
                coords[cont,0] = x[l,k]
                coords[cont,1] = y[l,k]
                C[l,k] = cont
                #   Identifica los nodos de frontera y actualiza el indicador
                if (L/2.-np.abs(x[l,k]) < tol or W/2.-np.abs(y[l,k]) < tol):
                    gfl[cont,0] = 1
                    # Si se encuentra en la frontera superior actualiza la velocidad x
                    if(W/2.-y[l,k] < tol):
                        gfl[cont,1] = U0
                cont += 1
    return coords, C, gfl, ng

def matDiff(Ne,Np,ne,npe,ng,x,y,C,gfl,mu=1):

#=================================================
# Genera la matriz de difusion/rigidez y el vector
# del lado derecho del sistema sujeto a condicio-
# nes de Dirichlet
#
# max(Np) < 6
#=================================================

    # Recupera los nodos de interpolacion de Lobatto
    dx = qlobatto(Np[0]-1,'z')
    dy = qlobatto(Np[1]-1,'z')

    (xi,wx) = qlobatto(Np[0],'b')   # Utiliza un orden mayor para tener una integral exacta
    (eta,wy) = qlobatto(Np[1],'b')   # Utiliza un orden mayor para tener una integral exacta

    NQ = (Np[0]+2)*(Np[1]+2)   # Numero de nodos a usar en la cuadratura

    (ix,iy) = np.meshgrid(range(Np[0]+2),range(Np[1]+2),indexing='xy')
    M_ind = np.array([ix.reshape(NQ),iy.reshape(NQ)]).T   # Indices para la cuadratura

    Mat_dif = np.zeros([2*ng+ne,2*ng+ne])   # Inicializa la matriz de difusion global

    for l in range(ne):
        elm_dm = np.zeros([npe,npe])   # Inicializa la matriz de difusion elemental
        vecD = np.zeros([npe,2])
        # Realiza la cuadratura de Gauss
        for nq in M_ind:
            psi = np.ones(npe)   # Almacena las funciones de interpolacion
            dpsixi = np.zeros(npe)   # Almacena las derivadas con respecto a xi
            dpsieta = np.zeros(npe)   # Almacena las derivadas con respecto a eta
            grpsi = np.zeros([npe,2])   # Almacena los gradientes de las funciones
            dxdxi = 0.
            dxdeta = 0.
            dydxi = 0.
            dydeta = 0.

            for i in range(npe):   #Realiza los calculos sobre cada elemento
                divx = i % (Np[0]+1)
                divy = i / (Np[0]+1)

                # Calcula las funciones de interpolacion
                fx = 1
                for ind in range(Np[0]+1):
                    if(ind != divx):
                        fx *= (xi[nq[0]]-dx[ind])/(dx[divx]-dx[ind])
                # print fx,
                fy = 1
                for ind in range(Np[1]+1):
                    if(ind != divy):
                        fy *= (eta[nq[1]]-dy[ind])/(dy[divy]-dy[ind])
                psi[i] = fx*fy
                # print fy, psi[i]
            
                # Calcula las derivadas con respecto a las dos variables
                for ind in range(Np[0]+1):
                    if(ind != divx):
                        prod = 1
                        for ind2 in range(Np[0]+1):
                            if(ind2 != divx and ind2 != ind):
                                prod *= (xi[nq[0]]-dx[ind2])/(dx[divx]-dx[ind2])
                        dpsixi[i] += 1./(dx[divx]-dx[ind])*prod
                dpsixi[i] *= fy
                for ind in range(Np[1]+1):
                    if(ind != divy):
                        prod = 1
                        for ind2 in range(Np[1]+1):
                            if(ind2 != divy and ind2 != ind):
                                prod *= (eta[nq[1]]-dy[ind2])/(dy[divy]-dy[ind2])
                        dpsieta[i] += 1./(dy[divy]-dy[ind])*prod
                dpsieta[i] *= fx
                # print dpsixi[i], dpsieta[i]

                # Actualiza los valores de las derivadas de la funcion de transformacion
                dxdxi += x[l,i]*dpsixi[i]
                dxdeta += x[l,i]*dpsieta[i]
                dydxi += y[l,i]*dpsixi[i]
                dydeta += y[l,i]*dpsieta[i]
                # print dxdxi,dxdeta,dydxi,dydeta
            
            for i in range(npe):
                # Calcula el gradiente para cada funcion de interpolacion
                grpsi[i,:] = lin.solve(np.array([[dxdxi,dydxi],[dxdeta,dydeta]]),
                                       np.array([dpsixi[i],dpsieta[i]]))
        
            # Calcula el factor de correccion
            hs = np.abs(dxdxi*dydeta-dxdeta*dydxi)
        
            # Construye la matriz de difusion elemental
            for i in range(npe):
                for j in range(npe):
                    elm_dm[i,j] += np.dot(grpsi[i,:],grpsi[j,:])*(hs*wx[nq[0]]*wy[nq[1]])
            # Construye los vectores D
                vecD[i,:] += grpsi[i,:]*(hs*wx[nq[0]]*wy[nq[1]])
        
        # Actualiza la matriz de difusion global
        Mat_dif[np.meshgrid(C[l,:],C[l,:],indexing='ij')] += mu*elm_dm
        Mat_dif[np.meshgrid(ng+C[l,:],ng+C[l,:],indexing='ij')] += mu*elm_dm
    
        # Extensiones 
        Mat_dif[C[l,:],2*ng+l] -= vecD[:,0]   #incorpora Dx en el bloque B
        Mat_dif[ng+C[l,:],2*ng+l] -= vecD[:,1]   #incorpora Dy en el bloque B
        Mat_dif[2*ng+l,C[l,:]] -= vecD[:,0]   #incorpora Dx en el bloque C
        Mat_dif[2*ng+l,ng+C[l,:]] -= vecD[:,1]   #incorpora Dy en el bloque C

    # Construye el vector del lado derecho
    vec_b = np.zeros(2*ng+ne)

    # Implementa las condiciones de frontera
    for ind in range(ng):
        if(gfl[ind,0]==1):
            # Ajusta las entradas al lado derecho
            vec_b -= Mat_dif[:,ind]*gfl[ind,1] + Mat_dif[:,ng+ind]*gfl[ind,2]
            # Ajusta el valor de la entrada en b
            vec_b[ind] = gfl[ind,1]   # Coor. x
            vec_b[ng+ind] = gfl[ind,2]   # Coor. y
            # Ajusta la matriz del problema
            Mat_dif[:,ind] = np.zeros(2*ng+ne)
            Mat_dif[:,ng+ind] = np.zeros(2*ng+ne)
            Mat_dif[ind,:] = np.zeros(2*ng+ne)
            Mat_dif[ng+ind,:] = np.zeros(2*ng+ne)
            Mat_dif[ind,ind] = 1.
            Mat_dif[ng+ind,ng+ind] = 1.

    return Mat_dif, vec_b

def matDiffTri(Ne,m,ne,npe,ng,x,y,C,gfl,mu=1):

#=================================================
# Genera la matriz de difusion/rigidez y el vector
# del lado derecho del sistema sujeto a condicio-
# nes de Dirichlet para el caso triangular
#
# m < 4
#=================================================

    # Define los nodos a usar en la cuadratura

    NQ = 7   # Numero de nodos a usar en la cuadratura (1,3,4,6,7,9,12,13)
    (xiq, etaq, wq) = gaussTri(NQ)
    (Mat_VDM, Det_VDM, Mat_Cof) = genVDM(m)
    prind = np.array([[0,1,0,2,1,0,3,2,1,0],
                          [0,0,1,0,1,2,0,1,2,3]])

    Mat_dif = np.zeros([2*ng+ne,2*ng+ne])   # Inicializa la matriz de difusion global

    for ell in range(ne):
        elm_dm = np.zeros([npe,npe])   # Inicializa la matriz de difusion elemental
        vecD = np.zeros([npe,2])
        # Realiza la cuadratura de Gauss
        for nq in range(NQ):
            psi = np.ones(npe)   # Almacena las funciones de interpolacion
            dpsixi = np.zeros(npe)   # Almacena las derivadas con respecto a xi
            dpsieta = np.zeros(npe)   # Almacena las derivadas con respecto a eta
            grpsi = np.zeros([npe,2])   # Almacena los gradientes de las funciones
            dxdxi = 0.
            dxdeta = 0.
            dydxi = 0.
            dydeta = 0.

            for i in range(npe):   #Realiza los calculos sobre cada elemento
                # Calcula las funciones de interpolacion
                vect = np.zeros(npe)
                for j in range(npe):
                    k = prind[0,j]
                    l = prind[1,j]
                    vect[j] = proriol(k,l,xiq[nq],etaq[nq])
                psi[i] = np.dot(vect,Mat_Cof[:,i])/Det_VDM

                # Calcula las derivadas con respecto a las dos variables
                vect = np.zeros(npe)
                for j in range(npe):
                    k = prind[0,j]
                    l = prind[1,j]
                    vect[j] = dproriol(k,l,xiq[nq],etaq[nq],'xi')
                dpsixi[i] = np.dot(vect,Mat_Cof[:,i])/Det_VDM
                vect = np.zeros(npe)
                for j in range(npe):
                    k = prind[0,j]
                    l = prind[1,j]
                    vect[j] = dproriol(k,l,xiq[nq],etaq[nq],'eta')
                dpsieta[i] = np.dot(vect,Mat_Cof[:,i])/Det_VDM
                # print dpsixi[i], dpsieta[i]

                # Actualiza los valores de las derivadas de la funcion de transformacion
                dxdxi += x[ell,i]*dpsixi[i]
                dxdeta += x[ell,i]*dpsieta[i]
                dydxi += y[ell,i]*dpsixi[i]
                dydeta += y[ell,i]*dpsieta[i]
                # print dxdxi,dxdeta,dydxi,dydeta

            for i in range(npe):
                # Calcula el gradiente para cada funcion de interpolacion
                grpsi[i,:] = lin.solve(np.array([[dxdxi,dydxi],[dxdeta,dydeta]]),
                                        np.array([dpsixi[i],dpsieta[i]]))

            # Calcula el factor de correccion
            hs = np.abs(dxdxi*dydeta-dxdeta*dydxi)

            # Construye la matriz de difusion elemental
            for i in range(npe):
                for j in range(npe):
                    elm_dm[i,j] += np.dot(grpsi[i,:],grpsi[j,:])*(0.5*hs*wq[nq])
            # Construye los vectores D
                vecD[i,:] += grpsi[i,:]*(0.5*hs*wq[nq])

        # Actualiza la matriz de difusion global
        Mat_dif[np.meshgrid(C[ell,:],C[ell,:],indexing='ij')] += mu*elm_dm
        Mat_dif[np.meshgrid(ng+C[ell,:],ng+C[ell,:],indexing='ij')] += mu*elm_dm

        # Extensiones 
        Mat_dif[C[ell,:],2*ng+ell] -= vecD[:,0]   #incorpora Dx en el bloque B
        Mat_dif[ng+C[ell,:],2*ng+ell] -= vecD[:,1]   #incorpora Dy en el bloque B
        Mat_dif[2*ng+ell,C[ell,:]] -= vecD[:,0]   #incorpora Dx en el bloque C
        Mat_dif[2*ng+ell,ng+C[ell,:]] -= vecD[:,1]   #incorpora Dy en el bloque C

    # Construye el vector del lado derecho
    vec_b = np.zeros(2*ng+ne)

    # Implementa las condiciones de frontera
    for ind in range(ng):
        if(gfl[ind,0]==1):
            # Ajusta las entradas al lado derecho
            vec_b -= Mat_dif[:,ind]*gfl[ind,1] + Mat_dif[:,ng+ind]*gfl[ind,2]
            # Ajusta el valor de la entrada en b
            vec_b[ind] = gfl[ind,1]   # Coor. x
            vec_b[ng+ind] = gfl[ind,2]   # Coor. y
            # Ajusta la matriz del problema
            Mat_dif[:,ind] = np.zeros(2*ng+ne)
            Mat_dif[:,ng+ind] = np.zeros(2*ng+ne)
            Mat_dif[ind,:] = np.zeros(2*ng+ne)
            Mat_dif[ng+ind,:] = np.zeros(2*ng+ne)
            Mat_dif[ind,ind] = 1.
            Mat_dif[ng+ind,ng+ind] = 1.
    
    return Mat_dif, vec_b