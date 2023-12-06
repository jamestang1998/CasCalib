import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import plotting
import util
import matplotlib.image as mpimg

def closestDistanceBetweenLines(a0,a1,b0,b1):
    
    #Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    #Return the closest points on each segment and their distance
    
    # Calculate denomitator
    #print(a0.shape,a1.shape,b0.shape,b1.shape, "SDAPJASD")
    A = a1 - a0
    B = b1 - b0
    magA = torch.norm(A, dim = 1)
    magB = torch.norm(B, dim = 1)
    
    _A = torch.div(A, torch.transpose(torch.stack((magA, magA, magA)), 0, 1))
    _B = torch.div(B, torch.transpose(torch.stack((magB, magB, magB)), 0, 1))
    
    cross = torch.cross(_A, _B, dim = 1)
    denom = torch.norm(cross, dim = 1)**2
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)

    detA = torch.det(torch.stack((t, _B, cross), dim = 2))
    detB = torch.det(torch.stack((t, _A, cross), dim = 2))

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (torch.mul(torch.stack((t0, t0, t0), dim = 1), _A)) # Projected closest point on segment A
    pB = b0 + (torch.mul(torch.stack((t1, t1, t1), dim = 1), _B)) # Projected closest point on segment B

    
    return pA,pB, torch.norm(pA-pB, dim = 1)