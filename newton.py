import numpy as np

def dv(x0, func, ep=1e-6):
    """Calculation the first derivative value of funcion(func) at x0."""
    f_prime = (func(x0 + ep/2) - func(x0 - ep/2))/ep
    
    return f_prime

def ddv(x0, func, ep=1e-6):
    """Calculation the second derivative value of funcion(func) at x0."""
    f_prime_plus = dv(x0 + ep/2, func)
    f_prime_minus = dv(x0 - ep/2, func)
    
    return (f_prime_plus - f_prime_minus) / ep


def newton_mine(x0, func, tol=1e-6, max_iter=1000):
    """Apply the Newton method to find a point where the function equals zero, starting from \( x_0 \)."""
    x=x0+10
    count_iter =0
    
    
    while(count_iter < max_iter):
        f_dv = dv(x0, func)
        f_ddv = ddv(x0, func)

        x=x0-(f_dv/f_ddv)

        if(x-x0 < tol):
            break
        
        x0=x
        count_iter+=1
    return x, func(x)

"""For example"""
if __name__ == "__main__":
    func = lambda x: x**2 + 4*x + 5
    x_min, f_min = newton_mine(0, func)
    
    print(x_min, f_min)
    print("version 2")

