import awkward
import numpy


# photon preselection for Run3 -> take as input nAOD Photon collection and return the Photons that pass
# cuts (pt, eta, sieie, mvaID, iso... etc)
#
def photon_preselection(
    photons: awkward.Array,
    events: awkward.Array,
    apply_electron_veto=True,
    year="2023",
) -> awkward.Array:
    """
    Apply preselection cuts to photons.
    Note that these selections are applied on each photon, it is not based on the diphoton pair.
    """
    
    # muon selection cuts
    muon_pt_threshold = 10
    muon_max_eta = 2.4
    mu_iso_wp = "medium"
    global_muon = False

    # electron selection cuts
    electron_pt_threshold = 15
    electron_max_eta = 2.5
    el_iso_wp = "WP80"

    # jet selection cuts
    jet_jetId = "tightLepVeto"  # can be "tightLepVeto" or "tight": https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV#nanoAOD_Flags
    jet_dipho_min_dr = 0.4
    jet_pho_min_dr = 0.4
    jet_ele_min_dr = 0.4
    jet_muo_min_dr = 0.4
    jet_pt_threshold = 20
    jet_max_eta = 4.7

    clean_jet_dipho = False
    clean_jet_pho = True
    clean_jet_ele = False
    clean_jet_muo = False

    # diphoton preselection cuts
    min_pt_photon = 25.0
    min_pt_lead_photon = 35.0
    min_mvaid = -0.9
    max_sc_eta = 2.5
    gap_barrel_eta = 1.4442
    gap_endcap_eta = 1.566
    max_hovere = 0.08
    min_full5x5_r9 = 0.8
    max_chad_iso = 20.0
    max_chad_rel_iso = 0.3

    min_full5x5_r9_EB_high_r9 = 0.85
    min_full5x5_r9_EE_high_r9 = 0.9
    min_full5x5_r9_EB_low_r9 = 0.5
    min_full5x5_r9_EE_low_r9 = 0.8
    max_trkSumPtHollowConeDR03_EB_low_r9 = (
            6.0  # for v11, we cut on Photon_pfChargedIsoPFPV
        )
    max_trkSumPtHollowConeDR03_EE_low_r9 = 6.0  # Leaving the names of the preselection cut variables the same to change as little as possible
    max_sieie_EB_low_r9 = 0.015
    max_sieie_EE_low_r9 = 0.035
    max_pho_iso_EB_low_r9 = 4.0
    max_pho_iso_EE_low_r9 = 4.0

    eta_rho_corr = 1.5
    low_eta_rho_corr = 0.16544
    high_eta_rho_corr = 0.13212
    
    # EA values for Run3 from Egamma
    EA1_EB1 = 0.102056
    EA2_EB1 = -0.000398112
    EA1_EB2 = 0.0820317
    EA2_EB2 = -0.000286224
    EA1_EE1 = 0.0564915
    EA2_EE1 = -0.000248591
    EA1_EE2 = 0.0428606
    EA2_EE2 = -0.000171541
    EA1_EE3 = 0.0395282
    EA2_EE3 = -0.000121398
    EA1_EE4 = 0.0369761
    EA2_EE4 = -8.10369e-05
    EA1_EE5 = 0.0369417
    EA2_EE5 = -2.76885e-05
    e_veto = 0.5
    
    # hlt-mimicking cuts
    rho = events.Rho.fixedGridRhoAll * awkward.ones_like(photons.pt)
    photon_abs_eta = numpy.abs(photons.eta)
    
    if year == "2022":
        # quadratic EA corrections in Run3 : https://indico.cern.ch/event/1204277/contributions/5064356/attachments/2538496/4369369/CutBasedPhotonID_20221031.pdf
        pass_phoIso_rho_corr_EB = (
            ((photon_abs_eta > 0.0) & (photon_abs_eta < 1.0))
            & (
                photons.pfPhoIso03 - (rho * EA1_EB1) - (rho * rho * EA2_EB1)
                < max_pho_iso_EB_low_r9
            )
        ) | (
            ((photon_abs_eta > 1.0) & (photon_abs_eta < 1.4442))
            & (
                photons.pfPhoIso03 - (rho * EA1_EB2) - (rho * rho * EA2_EB2)
                < max_pho_iso_EB_low_r9
            )
        )

        pass_phoIso_rho_corr_EE = (
            (
                ((photon_abs_eta > 1.566) & (photon_abs_eta < 2.0))
                & (
                    photons.pfPhoIso03
                    - (rho * EA1_EE1)
                    - (rho * rho * EA2_EE1)
                    < max_pho_iso_EB_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.0) & (photon_abs_eta < 2.2))
                & (
                    photons.pfPhoIso03
                    - (rho * EA1_EE2)
                    - (rho * rho * EA2_EE2)
                    < max_pho_iso_EB_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.2) & (photon_abs_eta < 2.3))
                & (
                    photons.pfPhoIso03
                    - (rho * EA1_EE3)
                    - (rho * rho * EA2_EE3)
                    < max_pho_iso_EB_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.3) & (photon_abs_eta < 2.4))
                & (
                    photons.pfPhoIso03
                    - (rho * EA1_EE4)
                    - (rho * rho * EA2_EE4)
                    < max_pho_iso_EB_low_r9
                )
            )
            | (
                ((photon_abs_eta > 2.4) & (photon_abs_eta < 2.5))
                & (
                    photons.pfPhoIso03
                    - (rho * EA1_EE5)
                    - (rho * rho * EA2_EE5)
                    < max_pho_iso_EB_low_r9
                )
            )
        )

    isEB_high_r9 = (photons.isScEtaEB) & (photons.r9 > min_full5x5_r9_EB_high_r9)
    isEE_high_r9 = (photons.isScEtaEE) & (photons.r9 > min_full5x5_r9_EE_high_r9)
    iso = photons.pfChargedIsoPFPV if hasattr(photons, "pfChargedIsoPFPV") else photons.trkSumPtHollowConeDR03  # photons.pfChargedIsoPFPV for v11, photons.trkSumPtHollowConeDR03 v12 and above
    rel_iso = photons.pfRelIso03_chg if hasattr(photons, "pfRelIso03_chg") else photons.pfRelIso03_chg_quadratic  # photons.pfRelIso03_chg for v1?, photons.pfRelIso03_chg_quadratic v12 and above
    isEB_low_r9 = (
        (photons.isScEtaEB)
        & (photons.r9 > min_full5x5_r9_EB_low_r9)
        & (photons.r9 < min_full5x5_r9_EB_high_r9)
        & (
            iso
            < max_trkSumPtHollowConeDR03_EB_low_r9
        )
        & (photons.sieie < max_sieie_EB_low_r9)
        & (pass_phoIso_rho_corr_EB)
    )
    isEE_low_r9 = (
        (photons.isScEtaEE)
        & (photons.r9 > min_full5x5_r9_EE_low_r9)
        & (photons.r9 < min_full5x5_r9_EE_high_r9)
        & (
            iso
            < max_trkSumPtHollowConeDR03_EE_low_r9
        )
        & (photons.sieie < max_sieie_EE_low_r9)
        & (pass_phoIso_rho_corr_EE)
    )
    # not apply electron veto for for TnP workflow
    e_veto = e_veto if apply_electron_veto else -1
    return photons[
        (photons.electronVeto > e_veto)
        & (photons.pt > min_pt_photon)
        & (photons.isScEtaEB | photons.isScEtaEE)
        & (photons.mvaID > min_mvaid)
        & (photons.hoe < max_hovere)
        & (
            (photons.r9 > min_full5x5_r9)
            | (
                rel_iso * photons.pt < max_chad_iso
            )
            | (rel_iso < max_chad_rel_iso)
        )
        & (isEB_high_r9 | isEB_low_r9 | isEE_high_r9 | isEE_low_r9)
    ], (
        (photons.electronVeto > e_veto)
        & (photons.pt > min_pt_photon)
        & (photons.isScEtaEB | photons.isScEtaEE)
        & (photons.mvaID > min_mvaid)
        & (photons.hoe < max_hovere)
        & (
            (photons.r9 > min_full5x5_r9)
            | (
                rel_iso * photons.pt < max_chad_iso
            )
            | (rel_iso < max_chad_rel_iso)
        )
        & (isEB_high_r9 | isEB_low_r9 | isEE_high_r9 | isEE_low_r9)
    )
