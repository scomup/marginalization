#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

    
def makeJ(p, edge):
    x1, y1 = p[edge[:,0],0], p[edge[:,0],1]
    x2, y2 = p[edge[:,1],0], p[edge[:,1],1]
    i = edge[:,0]
    j= edge[:,1]
    idx = np.arange(edge.shape[0])
    J = np.zeros([edge.shape[0], p.shape[0]*2])
    J[idx,2*i] = 2*(x1-x2)
    J[idx,2*i+1] = 2*(y1-y2)
    J[idx,2*j] = -2*(x1-x2)
    J[idx,2*j+1] = -2*(y1-y2)

    return J

def makeR(p, edge, z):
    x1, y1 = p[edge[:,0],0], p[edge[:,0],1]
    x2, y2 = p[edge[:,1],0], p[edge[:,1],1]
    r = np.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
    return r - z

def T1(pp, edge, z):
    #Gauss-nowton method
    last_e = np.inf
    J = makeJ(pp, edge)
    hessian = np.dot(J.T,J)
    """
    [huu  huv
    huv   hvv]
    """
    hvv = hessian[2:,2:]
    huv = hessian[0:2,2:]
    huu = hessian[0:2,0:2]
    h_star = hvv - huv.T@np.linalg.inv(huu)@huv
    h_star_inv = np.linalg.pinv(h_star)
    hessian_inv = np.linalg.pinv(hessian)
    while(True):
        r = makeR(pp, edge, z)
        e = np.dot(r[1:].T,r[1:])
        print(e)
        if last_e - e < 0.00001:
            break
        last_e = e
        #plt.imshow(hessian)
        #plt.show()
        b = -np.dot(J.T, r)
        bu = b[0:2]
        bv = b[2:]
        bv_star = bv  -  huv.T@np.linalg.inv(huu) @ bu
        dx = np.dot(h_star_inv, bv_star)
        dx = np.dot(hessian_inv, b)
        #pp[1:] += dx.reshape(pp[1:].shape)
        pp += dx.reshape(pp.shape)
    return pp

def T2(pp, edge, z):
    #Gauss-nowton method
    last_e = np.inf
    J = makeJ(pp, edge)
    hessian = np.dot(J.T,J)
    hessian = hessian
    hessian_inv = np.linalg.pinv(hessian)
    while(True):
        r = makeR(pp, edge, z)
        e = np.dot(r.T,r)
        #print(e)
        if last_e - e < 0.00001:
            break
        last_e = e
        #plt.imshow(hessian)
        #plt.show()
        b = -np.dot(J.T, r)
        dx = np.dot(hessian_inv, b)
        pp += dx.reshape(pp.shape)
    return pp


if __name__ == "__main__":



    p = np.array([[0,0],[2,0], [2,2], [0, 2], [1, 2+np.sqrt(3)]]) + np.random.normal(0, 0.3, (5, 2))
    p1 = p.copy()
    p2 = p.copy()
    edge = np.array([[0, 1], [0, 2], [0, 3],  [1, 2], [2, 3], [2, 4], [3, 4]])
    z = np.array([2, 2*np.sqrt(2), 2, 2, 2, 2, 2])

    p1 = T1(p1, edge, z)

    p2 = p2[1:]
    edge2 = np.array([[1, 2], [2, 3], [2, 4], [3, 4]]) -1
    z2 = np.array([2, 2, 2, 2])
    p2 = T2(p2, edge2, z2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(-1,4)
    plt.ylim(-1,4)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(p[:,0], p[:,1],color='r')
    plt.scatter(p1[:,0], p1[:,1],color='g')
    plt.scatter(p2[:,0], p2[:,1],color='b')
    x1, y1 = p[edge[:,0],0], p[edge[:,0],1]
    x2, y2 = p[edge[:,1],0], p[edge[:,1],1]
    plt.plot([x1, x2], [y1,y2], color='r')
    x1, y1 = p1[edge[:,0],0], p1[edge[:,0],1]
    x2, y2 = p1[edge[:,1],0], p1[edge[:,1],1]
    plt.plot([x1, x2], [y1,y2], color='g')
    x1, y1 = p2[edge2[:,0],0], p2[edge2[:,0],1]
    x2, y2 = p2[edge2[:,1],0], p2[edge2[:,1],1]
    plt.plot([x1, x2], [y1,y2], color='b')


    plt.show()