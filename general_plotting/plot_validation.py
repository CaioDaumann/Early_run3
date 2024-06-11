# Script to test the application of normalizing flows outisde of the main code enviroment

# python libraries import
import os 
import pandas
import numpy as np
import glob
import torch
import pandas as pd
import zuko

import matplotlib.pyplot as plt 
import mplhep, hist
plt.style.use([mplhep.style.CMS])

# importing other scripts
import standalone_plot        as plot_utils
#import zmmg_process_utils     as zmmg_utils
#from   apply_flow_zmmg        import zmmg_kinematics_reweighting

# This selection is actually made specially for electrons reconstructed as photons from base
def perform_zee_selection( data_df, mc_df ):
    # first we need to calculathe the invariant mass of the electron pair

    # variables names used to calculate the mass 
    data_mass_vars = ["lead_pt","lead_ScEta","lead_phi","sublead_pt","sublead_ScEta","sublead_phi","fixedGridRhoAll", "lead_mvaID"]
    mass_vars      = ["lead_pt","lead_ScEta","lead_phi","sublead_pt","sublead_ScEta","sublead_phi","fixedGridRhoAll", "lead_mvaID"]

    mass_inputs_data = np.array( data_df[data_mass_vars]) 
    mass_inputs_mc   = np.array( mc_df[mass_vars]  )

    # Calculating the invariant mass with the expression of a par of massless particles 
    mass_data = np.array( data_df["mass"])  
    mass_mc   = np.array( mc_df["mass"]  ) 

    # now, in order to perform the needed cuts two masks will be created
    mask_data = np.logical_and( mass_data > 80 , mass_data < 100  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,0] > 35  ) #tag pt cut
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,3] > 25  ) #probe pt cut
    mask_data = np.logical_and( mask_data , np.abs(mass_inputs_data[:,4]) < 2.5  )  # eta cut
    mask_data = np.logical_and( mask_data , np.abs(mass_inputs_data[:,5]) < 3.1415  ) # phi cut

    mask_mc   = np.logical_and( mass_mc > 80 , mass_mc < 100  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,0] > 35  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,3] > 25  )
    mask_mc   = np.logical_and( mask_mc , np.abs(mass_inputs_mc[:,4]) < 2.5  )
    mask_mc   = np.logical_and( mask_mc , np.abs(mass_inputs_mc[:,5]) < 3.1415  )

    # return the masks for further operations
    return mask_data, mask_mc


def calculate_bins_position(array, num_bins=12):

    array_sorted = np.sort(array)  # Ensure the array is sorted
    n = len(array)
    
    # Calculate the exact number of elements per bin
    elements_per_bin = n // num_bins
    
    # Adjust bin_indices to accommodate for numpy's 0-indexing and avoid out-of-bounds access
    bin_indices = [i*elements_per_bin for i in range(1, num_bins)]
    
    # Find the array values at these adjusted indices
    bin_edges = array_sorted[bin_indices]

    bin_edges = np.insert(bin_edges, 0, np.min(array))
    bin_edges = np.append(bin_edges, np.max(array))
    
    return bin_edges

# Due to the diferences in the kinematic distirbutions of the data and MC a reweithing must be performed to account for this
def perform_reweighting(simulation_df, data_df):
    
    # Reading and normalizing the weights
    mc_weights = np.array(simulation_df["weight"])
    mc_weights = mc_weights/np.sum( mc_weights )

    data_weights = np.ones(len(data_df["probe_pt"]))
    data_weights = data_weights/np.sum( data_weights )

    # Defining the reweigthing binning! - Bins were chossen such as each bin has ~ the same number of events
    pt_bins  = calculate_bins_position(np.array(simulation_df["probe_pt"]), 40)
    eta_bins = calculate_bins_position(np.array(simulation_df["probe_ScEta"]), 40)
    rho_bins = calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 40) #np.linspace( 5,65, 30) #calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 70)

    bins = [ pt_bins , eta_bins, rho_bins ]

    # Calculate 3D histograms
    data1 = [ np.array(simulation_df["probe_pt"]) , np.array(simulation_df["probe_ScEta"]) , np.array(simulation_df["fixedGridRhoAll"])]
    data2 = [ np.array(data_df["probe_pt"])       , np.array(data_df["probe_ScEta"])       , np.array(data_df["fixedGridRhoAll"])]

    hist1, edges = np.histogramdd(data1, bins=bins  , weights=mc_weights   , density=True)
    hist2, _     = np.histogramdd(data2, bins=edges , weights=data_weights , density=True)

    # Compute reweighing factors
    reweight_factors = np.divide(hist2, hist1, out=np.zeros_like(hist1), where=hist1!=0)

    # Find bin indices for each point in data1
    bin_indices = np.vstack([np.digitize(data1[i], bins=edges[i]) - 1 for i in range(3)]).T

    # Ensure bin indices are within valid range
    for i in range(3):
        bin_indices[:,i] = np.clip(bin_indices[:,i], 0, len(edges[i]) - 2  )
        
    # Apply reweighing factors
    simulation_weights = mc_weights * reweight_factors[bin_indices[:,0], bin_indices[:,1], bin_indices[:,2]]

    # normalizing both to one!
    data_weights       = data_weights/np.sum( data_weights )
    simulation_weights = simulation_weights/np.sum( simulation_weights )

    return data_weights, simulation_weights



var_list_corr = [
                "lead_pt", 
                "fixedGridRhoAll",
                "lead_ScEta",
                "lead_phi",
                "sublead_pt", 
                "sublead_ScEta",
                "sublead_phi",
                "pt",
                "eta",
                "mass"
                ]



data_var_list    = [
                    "lead_pt", 
                    "fixedGridRhoAll",
                    "lead_ScEta",
                    "lead_phi", 
                    "sublead_pt", 
                    "sublead_ScEta",
                    "sublead_phi", 
                    "pt",
                    "eta",
                    "mass"
                    ]


var_list    = [ 
                "lead_pt", 
                "fixedGridRhoAll",
                "lead_ScEta",
                "lead_phi",
                "sublead_pt", 
                "sublead_ScEta",
                "sublead_phi", 
                "pt",
                "eta",
                "mass"
                ]


data_conditions_list = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]
conditions_list      = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]


def read_data():

    # We read the whole data and MC and weight MC by the absolute luminosity of pre and postEE
    
    # Reading data!
    files = glob.glob(    "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Hgg_files/DY_with_base_checks/DataF_2022/nominal/*.parquet")
    files_2 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Hgg_files/DY_with_base_checks/DataG_2022/nominal/*.parquet")
    files_3 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Hgg_files/DY_with_base_checks/DataE_2022/nominal/*.parquet")
    files_4 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Hgg_files/DY_with_base_checks/DataC_2022/nominal/*.parquet")
    files_5 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Hgg_files/DY_with_base_checks/DataD_2022/nominal/*.parquet") 

    files = [files, files_2, files_3, files_4, files_5]

    data = [pd.read_parquet(f) for f in files]
    data_df = pd.concat(data,ignore_index=True)

    #data_df["probe_energyErr"] = data_df["probe_energyErr"]/ data_df["probe_pt"]*np.cosh( data_df["probe_eta"] ) 

    postEE_files = glob.glob("/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Hgg_files/DY_with_base_checks_with_SS_3/DYto2L_postEE/nominal/*.parquet") 
    preEE_files  = glob.glob("/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Hgg_files/DY_with_base_checks_with_SS_3/DYto2L_preEE/nominal/*.parquet")

    postEE = [pd.read_parquet(f) for f in postEE_files]
    postEE = pd.concat(postEE,ignore_index=True)  

    preEE = [pd.read_parquet(f) for f in preEE_files]
    preEE = pd.concat(preEE,ignore_index=True)  
    
    # Now lets scale the weights by the pre and post EE luminosities
    preEE["weight"]  = preEE["weight"]*8.1
    postEE["weight"] = postEE["weight"]*27.0
    
    mc_df = pd.concat([preEE, postEE], ignore_index=True)
    
    #mc_df["probe_energyErr"]     = mc_df["probe_energyErr"]/ mc_df["probe_pt"]*np.cosh( mc_df["probe_eta"] )
    #mc_df["probe_raw_energyErr"] = mc_df["probe_raw_energyErr"]/ mc_df["probe_pt"]*np.cosh( mc_df["probe_eta"] )
    
    return data_df, mc_df

def main():
   
    PostEE_plots = True

    data_df, mc_df = read_data()

    # performing the data selection!
    mask_data, mask_mc = perform_zee_selection( data_df, mc_df )

    # we still need the selection, and the rw!
    data_test_weights    = np.ones( len(data_df["lead_r9"]))
    mc_test_weights      = np.array( mc_df["weight"] )
    #data_test_weights, mc_test_weights = perform_reweighting(mc_df[mask_mc], data_df[mask_data])

    # This carries the corrected inputs
    mc_df = mc_df[mask_mc]
    mc_test_weights = mc_test_weights[mask_mc]
    
    data_df = data_df[mask_data]
    data_vector = np.array(data_df[data_var_list])
    data_test_weights    = np.ones( len(data_df["lead_r9"]))

    samples              = np.array( mc_df[var_list_corr]  )

    #mc_test_weights      = np.array( mc_df["weight"])
    mc_vector = np.array(mc_df[var_list] )
    
    for i in range( np.shape(samples)[1] ):

        mean = np.mean( np.nan_to_num(np.array( data_vector[:,i] ))  )
        std  = np.std(  np.nan_to_num(np.array( data_vector[:,i] ))  )

        if( str(var_list[i]) == 'pt'  ):
            data_hist     = hist.new.Reg(50, 0.0, mean + 4*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, 0.0, mean + 4*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, 0.0, mean + 4*std, overflow=True).Weight() 
        elif( 'pt' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(50, 25, mean + 4*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, 25, mean + 4*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, 25, mean + 4*std, overflow=True).Weight() 
        elif( 'Rho' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
        elif( 'Eta' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight() 
        elif( 'eta' in str(var_list[i])  ):   
            data_hist     = hist.new.Reg(50, -6.5, 6.5, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, -6.5, 6.5, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, -6.5, 6.5, overflow=True).Weight() 
        elif( 'phi' in str(var_list[i])  ):   
            data_hist     = hist.new.Reg(50, -3.1415, 3.1415, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, -3.1415, 3.1415, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, -3.1415, 3.1415, overflow=True).Weight() 
        elif( 'mass' in str(var_list[i])  ):   
            data_hist     = hist.new.Reg(50, 80, 100, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, 80, 100, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, 80, 100, overflow=True).Weight() 
        else:
            data_hist     = hist.new.Reg(30, mean - 1.5*std, mean + 1.5*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(30, mean - 1.5*std, mean + 1.5*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(30, mean - 1.5*std, mean + 1.5*std, overflow=True).Weight() 

        data_hist.fill(     np.array( np.array(data_vector[:,i] )) )
        mc_hist.fill(       np.array( np.array(mc_vector[:,i] )) , weight = len( np.array(data_vector[:,i] ))*mc_test_weights)
        mc_corr_hist.fill(  np.array( np.array(samples[:,i]))    , weight = len( np.array(data_vector[:,i] ))*mc_test_weights)

        if( str(var_list[i]) == 'pt'  ):
            var_list[i] = 'Dielectron_pt'
        if( str(var_list[i]) == 'eta'  ):
            var_list[i] = 'Dielectron_eta'

        if( PostEE_plots ):
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots/" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i].replace("raw_","")), postEE = True, endcap=False )
        else:
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots/" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i].replace("raw_","")), postEE = False, endcap=False )


if __name__ == "__main__":
    main()