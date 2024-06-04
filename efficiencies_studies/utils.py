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
    ]
    
## EE leak veto !!
def veto_EEleak_flag(egammas: awkward.Array) -> awkward.Array:
    """
    Add branch to veto electrons/photons in the EE+ leak region for 2022. Ref to:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#Notes_on_addressing_EE_issue_in
    """
    if hasattr(egammas, "isScEtaEE"):
        # photons
        out_EEleak_region = (egammas.isScEtaEB) | (
            (egammas.isScEtaEE) & (
                # keepng all the EE- photons : nanoAOD >= V13 with ScEta stored; Else with "eta" instead
                (egammas.ScEta < -1.5) | (
                    # keepng the EE+ photons not in the leakage region
                    (egammas.ScEta > 1.5) & (
                        (egammas.seediEtaOriX >= 45)
                        | (egammas.seediPhiOriY <= 72)
                    )
                )
            )
        )
    else:
        # electrons
        electron_ScEta = egammas.eta + egammas.deltaEtaSC
        out_EEleak_region = (numpy.abs(electron_ScEta) < self.gap_barrel_eta) | (
            (numpy.abs(electron_ScEta) > self.gap_endcap_eta)
            & (numpy.abs(electron_ScEta) < self.max_sc_eta) & (
                # keepng all the EE- objects
                (electron_ScEta < -1.5) | (
                    # keepng the EE+ objects not in the leakage region
                    (electron_ScEta > 1.5) & (
                        (egammas.seediEtaOriX >= 45)
                        | (egammas.seediPhiOriY <= 72)
                    )
                )
            )
        )

    egammas["vetoEELeak"] = out_EEleak_region

    return egammas

# Adds the SC eta to the photon object
def add_photon_SC_eta(photons: awkward.Array, PV: awkward.Array) -> awkward.Array:
    """
    Add supercluster eta to photon object, following the implementation from https://github.com/bartokm/GbbMET/blob/026dac6fde5a1d449b2cfcaef037f704e34d2678/analyzer/Analyzer.h#L2487
    In the current NanoAODv11, there is only the photon eta which is the SC eta corrected by the PV position.
    The SC eta is needed to correctly apply a number of corrections and systematics.
    """

    PV_x = PV.x.to_numpy()
    PV_y = PV.y.to_numpy()
    PV_z = PV.z.to_numpy()

    mask_barrel = photons.isScEtaEB
    mask_endcap = photons.isScEtaEE

    tg_theta_over_2 = numpy.exp(-photons.eta)
    # avoid dividion by zero
    tg_theta_over_2 = numpy.where(tg_theta_over_2 == 1., 1 - 1e-10, tg_theta_over_2)
    tg_theta = 2 * tg_theta_over_2 / (1 - tg_theta_over_2 * tg_theta_over_2)  # tg(a+b) = tg(a)+tg(b) / (1-tg(a)*tg(b))

    # calculations for EB
    R = 130.
    angle_x0_y0 = numpy.zeros_like(PV_x)

    angle_x0_y0[PV_x > 0] = numpy.arctan(PV_y[PV_x > 0] / PV_x[PV_x > 0])
    angle_x0_y0[PV_x < 0] = numpy.pi + numpy.arctan(PV_y[PV_x < 0] / PV_x[PV_x < 0])
    angle_x0_y0[((PV_x == 0) & (PV_y >= 0))] = numpy.pi / 2
    angle_x0_y0[((PV_x == 0) & (PV_y < 0))] = -numpy.pi / 2

    alpha = angle_x0_y0 + (numpy.pi - photons.phi)
    sin_beta = numpy.sqrt(PV_x**2 + PV_y**2) / R * numpy.sin(alpha)
    beta = numpy.abs(numpy.arcsin(sin_beta))
    gamma = numpy.pi / 2 - alpha - beta
    length = numpy.sqrt(R**2 + PV_x**2 + PV_y**2 - 2 * R * numpy.sqrt(PV_x**2 + PV_y**2) * numpy.cos(gamma))
    z0_zSC = length / tg_theta

    tg_sctheta = numpy.copy(tg_theta)
    # correct values for EB
    tg_sctheta = awkward.where(mask_barrel, R / (PV_z + z0_zSC), tg_sctheta)

    # calculations for EE
    intersection_z = numpy.where(photons.eta > 0, 310., -310.)
    base = intersection_z - PV_z
    r = base * tg_theta
    crystalX = PV_x + r * numpy.cos(photons.phi)
    crystalY = PV_y + r * numpy.sin(photons.phi)
    # correct values for EE
    tg_sctheta = awkward.where(
        mask_endcap, numpy.sqrt(crystalX**2 + crystalY**2) / intersection_z, tg_sctheta
    )

    sctheta = numpy.arctan(tg_sctheta)
    sctheta = awkward.where(
        sctheta < 0, numpy.pi + sctheta, sctheta
    )
    ScEta = -numpy.log(
        numpy.tan(sctheta / 2)
    )

    photons["ScEta"] = ScEta

    return photons
