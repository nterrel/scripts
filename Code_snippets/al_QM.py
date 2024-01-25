import anialservertools as ast
import anialtools as alt
import anitraintools as trn
from time import sleep
import os
import sys

#Overall process: Train --> Sampling --> QM --> bring back from server --> Store data as H5 file --> Use these data for next iterations
#FOR ANI-1x
# ----------------------- Moria ---------------------------------------------------------
root_dir = '/data/kdavis2/nms_net/'                         #Root directory
datdir = 'ANI-2x-0000.00'
h5stor = root_dir + 'h5files/'                               #h5store location
optlfile = root_dir + 'optimized_input_files.dat'

# ----------------------- Comet ---------------------------------------------------------
hostname = "comet.sdsc.xsede.org"                            #Use comet to run g09 for QM data generation
username = "kdavis2"                                        #User name for comet + set up RSA key
swkdir = '/oasis/scratch/comet/kdavis2/temp_project/b3lyp_al/'            #Server working directory in comet scratch

mae = 'module load gnu/4.9.2\n' +\
      'module load gaussian\n' +\
      'export PATH="/home/jsmith48/Gits/RCDBuilder/build/bin:$PATH"\n' +\
      'export LD_LIBRARY_PATH="/home/jsmith48/Gits/RCDBuilder/build/lib:$LD_LIBRARY_PATH"\n'

fpatoms = ['C', 'N', 'O', 'S', 'F', 'Cl']                                    #Atoms in system; except H

jtime = "0-4:00"                                             #Max job time					                                                                 

#---- Training Parameters ----
GPU = [0] 		                                     #GPU IDs on Moria; Use [2,3,4,5]; For testing [0]; Not [1,6]
print ("GPU = ", GPU)

M   = 0.01                                                   #Max error per atom in kcal/mol; QBC cutoff 
Nnets = 8                                                    #Number of networks in ensemble; 4 if need results quickly
Nblock = 16                                                  #Number of blocks in split; Nbtrain = (Nblock-(Nbval+Nbtest))
Nbvald = 2                                                   #Number of valid blocks
Nbtest = 2 						     #Number of test blocks
aevsize = 1008                                                #Should match inputsize under network_setup in /moria/auto_rxn_al/modelrxn/inputtrain.ipt; look into AEV formula for change

wkdir = '/data/kdavis2/nms_net/model/ANI-2x-0000/'                        #Network dir in /moria/auto_rxn_al/modelrxn/ to store new models
saefile = '/data/kdavis2/nms_net/model/sae_linfit.dat'
cstfile = '/data/kdavis2/nms_net/model/rHCNOSFCl-5.1R_16-3.5A_a8-4.params'         #5.2 radial cutoff; 16 radial values; 3.5 angular cutoff; 4-8 values

# ----------Trainer Input designer------------- #
ipt = trn.anitrainerinputdesigner()

ipt.set_parameter('atomEnergyFile','sae_linfit.dat')
ipt.set_parameter('sflparamsfile','rHCNO-5.2R_16-3.5A_a4-8.params')
ipt.set_parameter('eta','0.001')
ipt.set_parameter('energy','1')
ipt.set_parameter('force','0')
ipt.set_parameter('fmult','1.0')
ipt.set_parameter('feps','0.001')
ipt.set_parameter('dipole','0')
ipt.set_parameter('charge','0')
ipt.set_parameter('cdweight','2.0')
ipt.set_parameter('tolr', '60')
ipt.set_parameter('tbtchsz', '2560')                                  #maybe need change if I change aevsize; find info in 1ccx supplementary 
ipt.set_parameter('vbtchsz', '2560')                                  #maybe need change if I change aevsize
ipt.set_parameter('nkde', '2')

# Set network layers; Network Architecture in ANI-1ccx (couldnt find for 1x)
ipt.add_layer('H', {"nodes":160,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
ipt.add_layer('H', {"nodes":128,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-5})
ipt.add_layer('H', {"nodes":96,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

ipt.add_layer('C', {"nodes":144,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
ipt.add_layer('C', {"nodes":112,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-5})
ipt.add_layer('C', {"nodes":96 ,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

ipt.add_layer('N', {"nodes":128,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
ipt.add_layer('N', {"nodes":112,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-5})
ipt.add_layer('N', {"nodes":96 ,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

ipt.add_layer('O', {"nodes":128,"activation":9,"type":0,"l2norm":1,"l2valu":5.000e-3})
ipt.add_layer('O', {"nodes":112,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-5})
ipt.add_layer('O', {"nodes":96 ,"activation":9,"type":0,"l2norm":1,"l2valu":1.000e-6})

ipt.print_layer_parameters()

#-------------------------------------------------------------------------------------------------------------

# Training variables

### BEGIN CONFORMATIONAL REFINEMENT LOOP HERE ###
N = [0]

for i in N:
    netdir = wkdir+'ANI-2x-0000.00'+str(i).zfill(2)+'/'        #ANI-1x-B3LYP-0000.0000 --> ANI-1x-B3LYP-0000.000N

    if not os.path.exists(netdir):
        os.mkdir(netdir)

    netdict = {'cnstfile' : cstfile,
               'saefile': saefile,
               'iptsize': 1008,
               'nnfprefix': netdir+'train',
               'aevsize': aevsize,
               'num_nets': Nnets,
               'atomtyp' : ['H','C','N','O', 'S', 'F', 'Cl']                       #fpatoms + H
               }

    ## Train the ensemble ##
#    aet = trn.alaniensembletrainer(netdir, netdict, ipt, h5stor, Nnets)
#    aet.build_strided_training_cache(Nblock,Nbvald,Nbtest,False)
#    aet.train_ensemble(GPU)
#    exit(0)                                                       #If exit then active learning sampling will not be done!

    #### Sampling parameters ####
    nmsparams = {'T': 300.0, # Temperature
             'Ngen': 100, # Confs to generate
             'Nkep': 50, # Diverse confs to keep
             'maxd': 0.5, # Diverse confs to keep
             'sig' : M,
             }

    if not os.path.exists(root_dir + datdir + str(i+1).zfill(2)):
        os.mkdir(root_dir + datdir + str(i+1).zfill(2))

    ldtdir = root_dir  # local data directories

    ## Run active learning sampling ##    
    acs = alt.alconformationalsampler(ldtdir, datdir + str(i+1).zfill(2), optlfile, fpatoms+['H'], netdict)
    acs.run_sampling_nms(nmsparams, GPU)
    exit(0)                                                       #If exit then active learning sampling will not be done!

    ## Submit jobs, return and pack data
    ast.generateQMdata(hostname, username, swkdir, ldtdir, datdir + str(i+1).zfill(2), h5stor, mae, jtime,max_jobs=100)
