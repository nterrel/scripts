from ani2x_nc1 import CustomEnsemble
from atomic_rhos import *

this_device = torch.device('cpu')
model = CustomEnsemble(periodic_table_index=True, return_option=3).to(this_device)


_, min_ch4_e_total, ae_ch4_min = model(min_ch4)
_, bad_ch4_e_total, ae_ch4_bad = model(bad_ch4)
_, min_c10h10_e_total, ae_c10h10_min = model(min_c10h10)
_, bad_c10h10_e_total, ae_c10h10_bad = model(bad_c10h10)

print('ae ch4', ae_ch4_min)
#print('ae bad ch4', ae_ch4_bad)
#print('ae c10h10', ae_c10h10_min)
#print('ae bad c10h10', ae_c10h10_bad)

#print('ae ch4 stdev', ae_ch4_min.std(0))
#print('ae bad ch4 stdev', ae_ch4_bad.std(0))

#print('ae c10h10 stdev', ae_c10h10_min.std(0))
#print('ae bad c10h10 stdev', ae_c10h10_bad.std(0))

#print('ch4 total e stdev', min_ch4_e_total.std(0))
ani2x = torchani.models.ANI2x(periodic_table_index=True).to(this_device)
_, e_2x = ani2x.members_energies(min_ch4)
#print('ch4 2x e stdev',e_2x.std(0))
e_qbc = ani2x.energies_qbcs(min_ch4)[2]*627.509
#print(e_qbc)
