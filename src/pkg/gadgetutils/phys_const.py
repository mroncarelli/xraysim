pi = 3.141592653589793238462643383279502884197
c_light = 2.99792e10  # light speed [cm/s]
G_grav = 6.674e-8  # gravitational constant [dyn cm^2 g^-2]
h_Planck = 6.626068e-27  # Planck constant [cm^2 g s^-1]=[erg s]
k_B = 1.380658e-16  # Boltzmann constant [erg/K]
m_p = 1.6726231e-24  # proton mass [g]
m_e = 9.10938215e-28  # electron mass [g]
e_c = 1.6021765487e-19  # electron charge [C]
sigma_T = 6.6524587e-25  # Thomson cross section [cm^2]

# Conversion factors
pc2cm = 3.085678e18  # 1pc / 1cm [---]
Msun2g = 1.989e33  # 1M_Sun / 1g [---]
kpc2cm = pc2cm * 1.e3  # 1kpc / 1cm [---]
Mpc2cm = pc2cm * 1.e6  # 1Mpc / 1cm [---]
eV2J = e_c  # 1eV / 1J [---]
eV2erg = eV2J * 1.e7  # 1eV / 1erg [---]
keV2erg = eV2erg * 1.e3  # 1keV / 1erg [---]
ev2K = eV2erg / k_B  # 1eV / 1K [---]
keV2K = ev2K * 1.e3  # 1keV / 1K [---]

# Cosmological
H_0 = 100.  # Hubble constant [h km/(s*Mpc)]
H_0_cgs = H_0 * 1.e5 / Mpc2cm  # Hubble constant [h s^-1]
rho_crit = 3. / (8. * pi) * H_0 ** 2 / G_grav * (1.e5 / Mpc2cm) ** 2  # critical density [h^2 g/cm^3]
rho_crit_gadg = rho_crit * kpc2cm ** 3 * (1.e10 * Msun2g)  # [h^2 10^10 M_Sun kpc^-3] # critical density in Gadget units
Mpc2adim = H_0 / (c_light * 1.e-5)  # 1 h^-1 Mpc --> dimensionless
kpc2adim = Mpc2adim * 1.e-3  # 1 h^-1 kpc --> dimensionless

# Chemical
Xp = 0.76  # hydrogen mass fraction [---]
Yp = 1. - Xp  # helium mass fraction [---]
x_e0 = 1. + Yp / (2. * Xp)  # n_e/n_H [---]  (full ion, no metals)
mu0 = 1. / (2. * Xp + 3. / 4. * Yp)  # mean molecular weight [m_p] (full ion, no metals)

