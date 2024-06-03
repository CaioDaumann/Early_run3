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
limit = 2 # limit the number of read files
xrd_pfx_len = len(xrootd_pfx)
with open(samplejson) as f:
    sample_dict = json.load(f)

for key in sample_dict.keys():
    sample_dict[key] = sample_dict[key][:int(limit)]

# Looping over the files
total_events = 0
total_pass_events = 0

for key in sample_dict.keys():
    for file in sample_dict[key]:

        fname = file
        # Pass the file name directly as a string
        events = NanoEventsFactory.from_root(
            fname,
            schemaclass=NanoAODSchema,
            treepath="Events",  # ensure the treepath is correctly specified
            metadata={"dataset": "DYJets"},
        ).events()

        # defining the pre-selection
        photons, mask = utils.photon_preselection(photons = events.Photon, events=events,apply_electron_veto=True, year="2022")

        # Now diphoton selection
        # sort photons in each event descending in pt
        # make descending-pt combinations of photons
        photons = photons[ak.argsort(photons.pt, ascending=False)]
        photons["charge"] = ak.zeros_like(
            photons.pt
        )  # added this because charge is not a property of photons in nanoAOD v11. We just assume every photon has charge zero...
        diphotons = ak.combinations(
            photons, 2, fields=["pho_lead", "pho_sublead"]
        )

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

        # Some selection!
        selection_mask = ~ak.is_none(diphotons)
        diphotons      = diphotons[selection_mask]
        events         = events[selection_mask ]

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

        # Now lets count the number of events that passes the filters and the total number of events
        total_events = total_events + len(events)
        total_pass_events = total_pass_events +  len( events[events.Flag.BadPFMuonFilter == True] )

print( total_pass_events, total_events )
print( 'BadPFMuonFilter Efficiency: ', total_pass_events/total_events )