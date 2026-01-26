import math
import numpy as np
import matplotlib.pyplot as plt

#-- Simulation parameters
dt = 60.
t_max = 7200.
precip_time_centre = 1200.
precip_time_scale = 600.
precip_max_rate = 2e-5
#== Simulation parameters

#-- Model parameters
i_a = 1e-3
s = 0.01
h_a_init = 0.
h_s_init = 0.
#== Model parameters

#-- Initialize simulation
precip_cumulated = 0.0
nt = int(t_max / dt)
t_vector = [i * dt for i in range(0, nt + 1)]
precip_cumulated_vector = [0.0 for i in range (0, nt + 1)]
precip_rate_vector = [0.0 for i in range (0, nt + 1)]
h_a_vector = [0.0 for i in range (0, nt + 1)]
h_a_vector[0] = h_a_init
h_s_vector = [0.0 for i in range (0, nt + 1)]
h_s_vector[0] = h_s_init
h_r_vector = [0.0 for i in range (0, nt + 1)]
q_vector = [0.0 for i in range (0, nt + 1)]
infiltration_rate_vector = [0.0 for i in range (0, nt + 1)]
#== Initialize simulation

#-- Time loop
for n in range(0, nt):
    # Precip
    t = dt * n
    t_prime = (t - precip_time_centre) / precip_time_scale
    p_rate = precip_max_rate * math.exp(-t_prime * t_prime)
    precip_rate_vector[n] = p_rate
    precip_cumulated_vector[n+1] = precip_cumulated_vector[n] + p_rate * dt
    # Initial accumulation
    h_a_0 = h_a_vector[n]
    h_a = h_a_0 + dt * p_rate
    if (h_a < i_a):
        q = 0.0
        h_a_vector[n + 1] = h_a
    else:
        q = (h_a - i_a) / dt
        h_a_vector[n+1] = i_a
    q_vector[n] = q
    # split Soil reservoir / runoff
    X_begin = 1.0 - h_s_vector[n] / s
    X_end = 1.0 / (1.0 / X_begin + q * dt / s)
    h_s_vector[n + 1] = (1.0 - X_end) * s
    infiltration_rate = (h_s_vector[n + 1] - h_s_vector[n]) / dt
    h_r_vector[n+1] = h_r_vector[n] + (q - infiltration_rate) * dt
    infiltration_rate_vector[n] = infiltration_rate
#== Time loop

#-- Plot
#plt.plot (precip_cumulated_vector, h_a_vector)
#plt.plot (precip_cumulated_vector, h_s_vector)
plt.axes(xlabel = 't (s)', ylabel = 'h (m)')
plt.plot (t_vector, precip_cumulated_vector, color = 'blue', label = 'p (cumulated)')
plt.plot (t_vector, h_a_vector, color = 'grey', label = 'h in i_a')
plt.plot (t_vector, h_s_vector, color = 'green', label = 'h in soil')
plt.plot (t_vector, h_r_vector, color = 'red', label = 'h runoff')
plt.legend()
plt.title('Original SCS model')
plt.show()
#== Plot

