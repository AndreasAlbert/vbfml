import os
import uproot
from vbfml.models import sequential_dense_model
from vbfml.input.sequences import DatasetInfo, MultiDatasetSequence
import re
import glob


def get_n_events(filename, treename):
    try:
        return uproot.open(filename)[treename].num_entries
    except uproot.exceptions.KeyInFileError:
        return 0


labels = {
    "VBF_HToInvisible_M125_withDipoleRecoil_pow_pythia8_2017": "signal_17",
    "EWKWPlus2Jets_WToLNu_M-50_withDipoleRecoil-mg_2017" : "ewk_w_17",
    "EWKWMinus2Jets_WToLNu_M-50_withDipoleRecoil-mg_2017" : "ewk_w_17",
    "EWKZ2Jets_ZToLL_M-50_withDipoleRecoil-mg_2017": "ewk_zll_17",
    "EWKZ2Jets_ZToNuNu_M-50_withDipoleRecoil-mg_2017": "ewk_znn_17",
    "WJetsToLNu_HT-100To200-MLM_2017": "qcd_w_lo_17",
    "WJetsToLNu_HT-1200To2500-MLM_2017": "qcd_w_lo_17",
    "WJetsToLNu_HT-200To400-MLM_2017": "qcd_w_lo_17",
    "WJetsToLNu_HT-2500ToInf-MLM_2017": "qcd_w_lo_17",
    "WJetsToLNu_HT-400To600-MLM_2017": "qcd_w_lo_17",
    "WJetsToLNu_HT-600To800-MLM_2017": "qcd_w_lo_17",
    "WJetsToLNu_HT-800To1200-MLM_2017": "qcd_w_lo_17",
    "WJetsToLNu_Pt-100To250_MatchEWPDG20-amcatnloFXFX_2017": "qcd_w_nlo_17",
    "WJetsToLNu_Pt-250To400_MatchEWPDG20-amcatnloFXFX_2017": "qcd_w_nlo_17",
    "WJetsToLNu_Pt-400To600_MatchEWPDG20-amcatnloFXFX_2017": "qcd_w_nlo_17",
    "WJetsToLNu_Pt-600ToInf_MatchEWPDG20-amcatnloFXFX_2017": "qcd_w_nlo_17",
    "Z1JetsToNuNu_M-50_LHEFilterPtZ-150To250_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "Z1JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "Z1JetsToNuNu_M-50_LHEFilterPtZ-400ToInf_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "Z1JetsToNuNu_M-50_LHEFilterPtZ-50To150_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "Z2JetsToNuNu_M-50_LHEFilterPtZ-150To250_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "Z2JetsToNuNu_M-50_LHEFilterPtZ-250To400_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "Z2JetsToNuNu_M-50_LHEFilterPtZ-400ToInf_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "Z2JetsToNuNu_M-50_LHEFilterPtZ-50To150_MatchEWPDG20-amcatnloFXFX_2017": "qcd_znn_nlo_17",
    "ZJetsToNuNu_HT-100To200-MLM_2017": "qcd_znn_ht_lo_17",
    "ZJetsToNuNu_HT-1200To2500-MLM_2017": "qcd_znn_ht_lo_17",
    "ZJetsToNuNu_HT-200To400-MLM_2017": "qcd_znn_ht_lo_17",
    "ZJetsToNuNu_HT-2500ToInf-MLM_2017": "qcd_znn_ht_lo_17",
    "ZJetsToNuNu_HT-400To600-MLM_2017": "qcd_znn_ht_lo_17",
    "ZJetsToNuNu_HT-600To800-MLM_2017": "qcd_znn_ht_lo_17",
    "ZJetsToNuNu_HT-800To1200-MLM_2017": "qcd_znn_ht_lo_17",
}


def get_datasets():
    files = glob.glob("/media/nas/cms/vbfml/trees/*root")
    datasets = []
    for file in files:
        m = re.match("tree_(.*_\d{4}).root", os.path.basename(file))

        dataset_name = m.groups()[0]
        if '2018' in dataset_name:
            continue

        n_events = get_n_events(file, "sr_vbf")
        if not n_events:
            continue
        dataset = DatasetInfo(
            name=dataset_name,
            label=labels[dataset_name],
            files=[file],
            treename='sr_vbf',
            n_events=int(0.1*n_events),
        )
        datasets.append(dataset)
    return datasets


features = ["mjj","dphijj"]
mds = MultiDatasetSequence(batch_size=1000, branches=features, shuffle=True, batch_buffer_size=5000)
for dataset in get_datasets():
    if dataset.label in [
        "signal_17",
        "qcd_w_nlo_17",
        "qcd_znn_nlo_17",
        "ewk_znn_17",
        "ewk_w_18",
    ]:
        mds.add_dataset(dataset)

model = sequential_dense_model(n_layers=4, n_nodes=[16,16,16,16], n_features=len(features), n_classes=len(mds.dataset_labels()))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(mds, epochs=1)
