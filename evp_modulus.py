# parameters

TMP0 = 296.0 # initial temperature for slurry [K]
WC0 = 651.830 # initial water contents in slurry [kg/m3]
TH0 = 202.412 # inital thickness of coating layer [um]
THcell = 2.0 
C1 = 0.65248278 # volume fraction of solvent
C2 = 0.00200 # (particle dia./ini. coating thickness)^2
C4 = 400

n = 0.5
Ce_e = 0.1
Ce_a = 0.9
T_jet = 130
T_boiling = 100
CT = 1

#Set_User_Memory_Name(0,"Water_Contents[kg]")
#Set_User_Memory_Name(1, "Water_Contents[kg/m3]")
#Set_User_Memory_Name(2, "Evaporation_Rate[kg/sec]")
#Set_User_Memory_Name(3, "Evaporation_Rate[kg/m3/sec]")
#Set_User_Memory_Name(4, "Wall_Heat_Flux")
#Set_User_Memory_Name(5, "h_monitoring")

wc0 = WC0*TH0*1e-3/THcell
#wc = WC0 #C_UDMI(c0,t0,1)
u_temperature_f = T*273.15 #T_1*273.15
u_temperature_w = T*273.15 #T_2*273.15 #u_temperature_f
h = 35 # T+273.15
u_fraction = 0.00725 #E
fluid_density = 1.1614 #kg/m3

cp_a=(28.11+1.967*(1e-3)*u_temperature_f+4.802*(1e-6)*(u_temperature_f**2)-1.966*(1e-9)*(u_temperature_f**3))/28.970*1000
# cp_a = 주변 공기의 비열 [J/KgK]
cp_v=(32.24+1.923*(1e-3)*u_temperature_w+10.55*(1e-6)*(u_temperature_w**2)-3.595*(1e-9)*(u_temperature_w**3))/18.015*1000
# cp_v = 계면 수증기의 비열 [J/KgK]
P_v = Min(exp(23.2-3816.4/(u_temperature_w-46.1)),100000)
# if ((u_temperature_w > 273)&&(u_temperature_w<473))
u_pressure = 101325
x_v = 0.62198*P_v/(u_pressure-P_v)
# 표면에서는 순수 용매가 포화상태로 있다고 가정시 계면 수증기의 절대습도 x_v[kg/kg]
x_a = u_fraction #주변 공기의 절대습도 [kg/kg]
C3 = ((wc/wc0)**n) # solvent remaining coefficient
porous_correction = C1*C2*C3*C4*CT
#m_dot = heat_flux/(2317.0*(10**3))*porous_correction #the heat of vaporization = 2317 [kj/kg] at 350K
m_dot = h/(cp_a+cp_v*x_a)*(x_v-x_a)*porous_correction/(1e-3*THcell) #kg/m3/sec
evp_water = m_dot/fluid_density #(1e-3*THcell) #공기의 무게 1m3당 1.2kg, plate area 0.46

# wc = wc-m_dot*(1e-3*THcell)

# source term for energy equation
hfg = 2257.0 #KJ/Kg
src_h = -m_dot*hfg#*1000.0
src_temp = src_h / (1e-3*cp_a) / fluid_density #/(le-3*THcell)
nd_src_temp = src_temp / 273.15

# set equations
# normal_gradient_water = (normal_z*E.diff(z))