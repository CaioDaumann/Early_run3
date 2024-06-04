import awkward as ak
import numpy
import os
import warnings
import json

from coffea import processor
from coffea.analysis_tools import Weights
from copy import deepcopy

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# importing other scripts
import utils 

fiducialCuts = 'geometric'
# diphoton preselection cuts
min_pt_photon = 25.0
min_pt_lead_photon = 35.0

# Lets read the .json
samplejson = 'keep_v13_sample.json'
# load dataset
xrootd_pfx = "root://"
limit = 25 # limit the number of read files
xrd_pfx_len = len(xrootd_pfx)
with open(samplejson) as f:
    sample_dict = json.load(f)

for key in sample_dict.keys():
    sample_dict[key] = sample_dict[key][:int(limit)]

# Looping over the files
total_events = 0
total_pass_EE_leak = 0
total_goodVertices = 0
total_globalSuperTightHalo2016Filter = 0
total_HBHENoiseFilter = 0
total_HBHENoiseIsoFilter = 0
total_EcalDeadCellTriggerPrimitiveFilter = 0
total_BadPFMuonFilter = 0
total_eeBadScFilter = 0
total_all_filters = 0
total_photons = 0

for key in sample_dict.keys():
    for file in sample_dict[key]:

        print('Processing file: ', file)

        fname = file
        # Pass the file name directly as a string
        events = NanoEventsFactory.from_root(
            fname,
            schemaclass=NanoAODSchema,
            treepath="Events",  # ensure the treepath is correctly specified
            metadata={"dataset": "DYJets"},
        ).events()

        # defining the pre-selection
        photons = utils.photon_preselection(photons = events.Photon, events=events,apply_electron_veto=True, year="2022")
        #photons = ak.where(ak.num(photons) == 0, None, photons)
        #photons = events.Photon
 
        # Now diphoton selection
        # sort photons in each event descending in pt
        # make descending-pt combinations of photons
        photons = photons[ak.argsort(photons.pt, ascending=False)]
        photons["charge"] = ak.zeros_like(photons.pt)  
        diphotons = ak.combinations(photons, 2, fields=["pho_lead", "pho_sublead"])

        # the remaining cut is to select the leading photons
        # the previous sort assures the order
        diphotons = diphotons[
            diphotons["pho_lead"].pt > min_pt_lead_photon
        ]

        # now turn the diphotons into candidates with four momenta and such
        diphoton_4mom = diphotons["pho_lead"] + diphotons["pho_sublead"]
        diphotons["pt"] = diphoton_4mom.pt
        diphotons["eta"] = diphoton_4mom.eta
        diphotons["phi"] = diphoton_4mom.phi
        diphotons["mass"] = diphoton_4mom.mass
        diphotons["charge"] = diphoton_4mom.charge

        diphoton_pz = diphoton_4mom.z
        diphoton_e = diphoton_4mom.energy

        diphotons["rapidity"] = 0.5 * numpy.log((diphoton_e + diphoton_pz) / (diphoton_e - diphoton_pz))

        diphotons = ak.with_name(diphotons, "PtEtaPhiMCandidate")

        # sort diphotons by pT
        diphotons = diphotons[
            ak.argsort(diphotons.pt, ascending=False)
        ]

        # Mass selection
        mass_mask = (diphotons.mass > 100) & (diphotons.mass < 180)
        diphotons = diphotons[mass_mask]

        # Determine if event passes fiducial Hgg cuts at detector-level
        if fiducialCuts == 'classical':
            fid_det_passed = (diphotons.pho_lead.pt / diphotons.mass > 1 / 3) & (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4) & (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10) & ((diphotons.pho_sublead.pfRelIso03_all_quadratic * diphotons.pho_sublead.pt) < 10) & (numpy.abs(diphotons.pho_lead.eta) < 2.5) & (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
        elif fiducialCuts == 'geometric':
            fid_det_passed = (numpy.sqrt(diphotons.pho_lead.pt * diphotons.pho_sublead.pt) / diphotons.mass > 1 / 3) & (diphotons.pho_sublead.pt / diphotons.mass > 1 / 4) & (diphotons.pho_lead.pfRelIso03_all_quadratic * diphotons.pho_lead.pt < 10) & (diphotons.pho_sublead.pfRelIso03_all_quadratic * diphotons.pho_sublead.pt < 10) & (numpy.abs(diphotons.pho_lead.eta) < 2.5) & (numpy.abs(diphotons.pho_sublead.eta) < 2.5)
        elif fiducialCuts == 'none':
            fid_det_passed = diphotons.pho_lead.pt > -10  # This is a very dummy way but I do not know how to make a true array of outer shape of diphotons
        else:
            warnings.warn("You chose %s the fiducialCuts mode, but this is currently not supported. You should check your settings. For this run, no fiducial selection at detector level is applied." % fiducialCuts)
            fid_det_passed = diphotons.pho_lead.pt > -10

        diphotons = diphotons[fid_det_passed]

        # Some selection!
        selection_mask = ak.num(diphotons) >= 1  #~ak.is_none(diphotons)
        diphotons      = diphotons[selection_mask]
        events         = events[selection_mask ]
        
        # Lets count the total number of events before the filters!
        total_events = total_events + len( events )
        total_photons = total_photons + len( events)

        flags = ["goodVertices", "globalSuperTightHalo2016Filter", "HBHENoiseFilter", "HBHENoiseIsoFilter", "EcalDeadCellTriggerPrimitiveFilter", "BadPFMuonFilter", "eeBadScFilter"]
        
        print('We are on the flags part!!')
        for flag in flags:
            if( "goodVertices" in flag ):
                total_goodVertices = total_goodVertices + len( events[events.Flag[flag] == True] )
            elif( "globalSuperTightHalo2016Filter" in flag ):
                total_globalSuperTightHalo2016Filter = total_globalSuperTightHalo2016Filter + len( events[events.Flag[flag] == True] )
            elif( "HBHENoiseFilter" in flag ):
                total_HBHENoiseFilter = total_HBHENoiseFilter + len( events[events.Flag[flag] == True] )
            elif( "HBHENoiseIsoFilter" in flag ):
                total_HBHENoiseIsoFilter = total_HBHENoiseIsoFilter + len( events[events.Flag[flag] == True] )
            elif( "EcalDeadCellTriggerPrimitiveFilter" in flag ):
                total_EcalDeadCellTriggerPrimitiveFilter = total_EcalDeadCellTriggerPrimitiveFilter + len( events[events.Flag[flag] == True] )
            elif( "BadPFMuonFilter" in flag ):
                total_BadPFMuonFilter = total_BadPFMuonFilter + len( events[events.Flag[flag] == True] )
            elif( "eeBadScFilter" in flag ):
                total_eeBadScFilter = total_eeBadScFilter + len( events[events.Flag[flag] == True] )
                
        # All filters - logical and
        logi1 = numpy.logical_and(events.Flag["goodVertices"] == True ,   events.Flag["globalSuperTightHalo2016Filter"] == True )
        logi2 = numpy.logical_and(events.Flag["HBHENoiseFilter"] == True ,  events.Flag["HBHENoiseIsoFilter"] == True )
        logi3 = numpy.logical_and(events.Flag["EcalDeadCellTriggerPrimitiveFilter"] == True ,  events.Flag["BadPFMuonFilter"] == True )
        logi4 = numpy.logical_and(events.Flag["eeBadScFilter"] == True ,  logi3 )
        logic_1_2 = numpy.logical_and(logi1, logi2)
        logic_3_4 = numpy.logical_and(logi4, logi3)
        ultimate_logic = numpy.logical_and(logic_1_2, logic_3_4)
        total_all_filters = total_all_filters + len( events[ ultimate_logic == True ] )

        # Adding the SC eta
        events.Photon = utils.add_photon_SC_eta(events.Photon, events.PV)

        # EE veto leak: - also need to check this efficiency
        events.Photon = utils.veto_EEleak_flag(events.Photon)
        photons = events.Photon
        
        # We need to do it again after the EEvetor
        photons = photons[ak.argsort(photons.pt, ascending=False)]
        photons["charge"] = ak.zeros_like(photons.pt)  
        diphotons = ak.combinations(photons, 2, fields=["pho_lead", "pho_sublead"])

        # the remaining cut is to select the leading photons
        # the previous sort assures the order
        diphotons = diphotons[
            diphotons["pho_lead"].pt > min_pt_lead_photon
        ]

        selection_mask = ak.num(diphotons) >= 1 #~ak.is_none(diphotons)
        diphotons      = diphotons[selection_mask]
        events         = events[selection_mask ]
        
        total_pass_EE_leak = total_pass_EE_leak + len( events )
        
print( 'Total number of events: ',  total_events )
print( 'EE leak efficiency: ',  total_pass_EE_leak/total_photons )
print('goodVertices efficiency: ', total_goodVertices/total_events)
print('globalSuperTightHalo2016Filter efficiency: ', total_globalSuperTightHalo2016Filter/total_events)
print('HBHENoiseFilter efficiency: ', total_HBHENoiseFilter/total_events)
print('HBHENoiseIsoFilter efficiency: ', total_HBHENoiseIsoFilter/total_events)
print('EcalDeadCellTriggerPrimitiveFilter efficiency: ', total_EcalDeadCellTriggerPrimitiveFilter/total_events)
print('BadPFMuonFilter efficiency: ', total_BadPFMuonFilter/total_events)
print('eeBadScFilter efficiency: ', total_eeBadScFilter/total_events)
print('All filters efficiency: ', total_all_filters/total_events)