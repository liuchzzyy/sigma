from pathlib import Path

import hyperspy.api as hs
import numpy as np
from exspy.signals import EDSSEMSpectrum

hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()


class BaseDataset:
    def __init__(self, file_path: str | Path):
        self.base_dataset = hs.load(file_path)
        self.nav_img = None
        self.spectra = None
        self.original_nav_img = None
        self.original_spectra = None
        self.nav_img_bin = None
        self.spectra_bin = None
        self.spectra_raw = None
        self.feature_list = []
        self.feature_dict = {}

    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        for s in [self.spectra, self.spectra_bin]:
            if s is not None:
                s.metadata.Sample.xray_lines = self.feature_list
        self.feature_dict = {el: i for (i, el) in enumerate(feature_list)}
        print(f"Set feature_list to {self.feature_list}")

    def rebin_signal(self, size=(2, 2)):
        print(f"Rebinning the intensity with the size of {size}")
        if self.spectra is None:
            raise ValueError("Cannot rebin: spectra is None")
        if self.nav_img is None:
            raise ValueError("Cannot rebin: nav_img is None")

        x, y = size[0], size[1]
        self.spectra_bin = self.spectra.rebin(scale=(x, y, 1))
        self.nav_img_bin = self.nav_img.rebin(scale=(x, y))
        if self.spectra_bin is None:
            raise ValueError("Rebinning failed: spectra_bin is None")

        self.spectra_raw = self.spectra_bin.deepcopy()
        return (self.spectra_bin, self.nav_img_bin)

    def remove_fist_peak(self, end: float):
        print(
            f"Removing the fisrt peak by setting the intensity to zero until the energy of {end} keV."
        )
        for spectra in (self.spectra, self.spectra_bin):
            if spectra is None:
                continue
            else:
                scale = spectra.axes_manager[2].scale  # type: ignore
                offset = spectra.axes_manager[2].offset  # type: ignore
                end_ = int((end - offset) / scale)
                for i in range(end_):
                    spectra.isig[i] = 0

    def peak_intensity_normalisation(self) -> EDSSEMSpectrum:
        print(
            "Normalising the chemical intensity along axis=2, so that the sum is wqual to 1 along axis=2."
        )
        if self.spectra_bin:
            spectra_norm = self.spectra_bin
        else:
            spectra_norm = self.spectra

        if spectra_norm is None:
            raise ValueError("Cannot normalise: no spectra data found.")

        # Ensure correct broadcasting and handle potential nans/zeros
        total_intensity = spectra_norm.data.sum(axis=2, keepdims=True)
        # Avoid division by zero
        total_intensity[total_intensity == 0] = 1.0

        spectra_norm.data = spectra_norm.data / total_intensity

        if np.isnan(np.sum(spectra_norm.data)):
            spectra_norm.data = np.nan_to_num(spectra_norm.data)
        return spectra_norm

    def peak_denoising_PCA(
        self, n_components_to_reconstruct=10, plot_results=True
    ) -> EDSSEMSpectrum:
        print("Peak denoising using PCA.")
        if self.spectra_bin:
            spectra_denoised = self.spectra_bin
        else:
            spectra_denoised = self.spectra

        if spectra_denoised is None:
            raise ValueError("Cannot denoise: no spectra data found.")

        spectra_denoised.decomposition(  # type: ignore
            normalize_poissonian_noise=True,
            algorithm="SVD",
            random_state=0,
            output_dimension=n_components_to_reconstruct,
        )

        if plot_results == True:
            spectra_denoised.plot_decomposition_results()  # type: ignore
            spectra_denoised.plot_explained_variance_ratio(log=True)  # type: ignore
            spectra_denoised.plot_decomposition_factors(comp_ids=4)  # type: ignore

        return spectra_denoised

    def get_feature_maps(self, feature_list=None, raw_data: bool = False) -> np.ndarray:
        if feature_list is not None:
            self.set_feature_list(feature_list)

        if self.spectra is None:
            raise ValueError("spectra is None")

        num_elements = len(self.feature_list)

        if (self.spectra_bin is None) or (raw_data):
            lines = self.spectra.get_lines_intensity(self.feature_list)  # type: ignore
        else:
            lines = self.spectra_bin.get_lines_intensity(self.feature_list)  # type: ignore

        dims = lines[0].data.shape  # type: ignore
        data_cube = np.zeros((dims[0], dims[1], num_elements))

        for i in range(num_elements):
            data_cube[:, :, i] = lines[i]  # type: ignore

        return data_cube

    def normalisation(self, norm_list=[]):
        self.normalised_elemental_data = self.get_feature_maps(self.feature_list)
        print("Normalise dataset using:")
        for i, norm_process in enumerate(norm_list):
            print(f"    {i + 1}. {norm_process.__name__}")
            self.normalised_elemental_data = norm_process(
                self.normalised_elemental_data
            )
