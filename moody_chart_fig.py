import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import brentq
import pickle
# def solve_colebrook(rel_roughness, reynolds_num):
#     """Solve the Colebrook equation for the friction factor.

#     Positional arguments:
#     rel_roughness -- epsilon/D (right axis on graph)
#     reynolds_num -- Re (bottom axis on graph)

#     """
#     def colebrook(f):
#         return 1/np.sqrt(f) + 2 * np.log10(rel_roughness/3.7 + 2.51/(reynolds_num*np.sqrt(f)))
#     return brentq(colebrook, 0.005, 0.1)

# solve_colebrook_v = np.vectorize(solve_colebrook)

xmin, xmax = 5E2, 1E8
ymin, ymax = 0.008, 0.1

# n_pts = 100

# laminar_Re = np.linspace(1, 3500, n_pts)
# turbulent_Re = np.logspace(np.log10(2000), np.log10(xmax), n_pts)

# relative_roughness_values = [5E-2, 4E-2, 3E-2, 2E-2, 1.5E-2, 1E-2, 8E-3, 6E-3, 4E-3,
#                             2E-3, 1E-3, 5E-4, 2E-4, 1E-4, 5E-5, 2E-5, 1E-5, 0]

# f_lam = 64 / laminar_Re

# f_turb = np.zeros((n_pts, len(relative_roughness_values)))

# for rough_ind, rough_val in enumerate(relative_roughness_values):
#     f_turb[:, rough_ind] = solve_colebrook_v(rough_val, turbulent_Re)
# fig, ax = plt.subplots() 
fig = pickle.load(open('/Users/slauer/Documents/GitHub/streamlits/moody.pkl','rb'))
# plt.ylabel(r"Friction Factor, $f$")          
# fig = plt.figure() 
# ax2 = ax.twinx()
# plt.plot(laminar_Re, f_lam)
# ax2.plot(turbulent_Re, f_turb)
# Reynolds number labels (horizontal axis)
plt.xlim(xmin, xmax)
# plt.xscale('log')
# plt.xlabel(r"Reynolds Number, $\mathit{Re}=\rho{}VD/\mu$")

# Friction factor labels (left vertical axis)
plt.ylim(ymin, ymax)
# plt.yscale('log')
# plt.gca().tick_params(which='both', right='off', top='off')
# ax2.set_ylabel(r"Relative roughness factor")
plt.plot([100000], [.03],'o', color='black')
# pickle.dump(fig,file('moody.pickle','wb'))
# plt.savefig("moody-diagram.png", bbox_inches='tight')
plt.show()