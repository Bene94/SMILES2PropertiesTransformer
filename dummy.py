from torch import nn
import torch

## function that test if a number is prime
def is_prime(n):
        return n > 1 and all(n%i for i in range(2, n))

## plot a mandelbrot fractall
def mandelbrot(x, y, max_iter):
        c = x + y*1j
        z = 0
        for n in range(max_iter):
                z = z**2 + c
                if abs(z) > 2:
                        return n
        return max_iter

# a function that compute the mandelbrot set
def mandelbrot_set(x_from, x_to, y_from, y_to, size, max_iter):
        height = size   
        width = size
        x_step = (x_to - x_from)/width
        y_step = (y_to - y_from)/height
        x = torch.linspace(x_from, x_to, width)
        y = torch.linspace(y_from, y_to, height)
        x = x.view(1, -1).expand(height, width)
        y = y.view(-1, 1).expand(height, width)
        img = torch.zeros(height, width)
        for i in range(height):
                for j in range(width):
                        img[i, j] = mandelbrot(x[i, j], y[i, j], max_iter)
        return img  

# plot image
def plot_image(img):
        import matplotlib.pyplot as plt
        plt.imshow(img.numpy(), cmap='gray')
        plt.show()

# draw the 2d fibuacci tree
def draw_fib_tree(size, depth):
        if depth == 0:
                return
        else:
                for i in range(size):
                        print(i)
                        draw_fib_tree(size, depth-1)
                        print('\n')

draw_fib_tree(10, 3)


img = mandelbrot_set(-2.0, 0.5, -1.25, 1.25, 1024, 80)
plot_image(img)

