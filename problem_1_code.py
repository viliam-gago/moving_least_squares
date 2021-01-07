import matplotlib.pyplot as plt
import numpy as np


#original fucntion
x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x) + 0.5 * np.cos(6 * np.pi * x + (np.pi / 4))

noise = np.random.normal(0,0.2,100)
noisy_y = y + noise

#plot function vs function with noise
plt.plot(x, y, color='k', linestyle='--', linewidth='1', label='original')
plt.scatter(x, noisy_y, marker='.', label='added_noise')
plt.title('Original function vs added noise function')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.show()


#OLS
#fit using polynomial basis -> here 8th order used
ols_fit = np.polyfit(x.reshape(1, -1)[0], noisy_y.reshape(1, -1)[0], 8)


plt.plot(x, y, color='k', linestyle='--', linewidth='0.5', label='original')
# plt.scatter(x1,noisy_f1,marker='.',label='noised_function')
plt.plot(x, np.polyval(ols_fit, x), 'r', label='LS approximation')
plt.legend()
plt.title('Original function vs LS approximation')
plt.xlabel('x')
plt.ylabel('f1(x)')
plt.legend()
plt.show()

################################################### SET THE BASIS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def create_design_matrix(x):
    ones_col = np.ones(len(x))
    ones_col = ones_col.reshape(-1, 1)
    basis_to_P = (ones_col, x.reshape(-1, 1), x.reshape(-1, 1) ** 2)
    P = np.hstack(basis_to_P)
    return P


unknowns = np.linspace(0, 1, 100)
h = 0.01
dmi = 20 * h
gauss_coef = 3
P = create_design_matrix(x)


###3
mls_fit = np.array([])
for unk in unknowns:
    # a - weight function selected - viz above

    r = np.abs(x - unk) / dmi
    w = (np.where(r <= 1, (np.exp((-r ** 2) * (gauss_coef ** 2)) - np.exp(-gauss_coef ** 2)) / (1 - np.exp(-gauss_coef ** 2)), 0))
    W = np.diag(w)

    A = np.dot(np.dot(P.T, W), P)
    A_inv = np.linalg.inv(A)
    B = np.dot(P.T, W)
    p = np.hstack([1, unk, unk ** 2])
    shape_func = np.dot(np.dot(p, A_inv), B)

    mls_fit = np.append(mls_fit, shape_func.dot(y))


plt.scatter(x, noisy_y, marker='.', s=10, color='k', alpha=0.3)
plt.plot(x, y, 'k--', label='Original', linewidth=0.5)
plt.plot(unknowns, mls_fit, 'g', linewidth=2, label='MLS')
plt.plot(x, np.polyval(ols_fit, x), 'r', label='OLS', linewidth=0.5)

fig = plt.gcf()
fig.set_size_inches(10, 8)


plt.xlim(0,1)
plt.ylim(-1.5,1.5)
plt.title('MLS vs OLS fit')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()