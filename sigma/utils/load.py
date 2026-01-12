import os
from os.path import join
from pathlib import Path
import hyperspy.api as hs
import numpy as np
from exspy.signals import EDSSEMSpectrum, EDSTEMSpectrum
from hyperspy.signals import Signal1D, Signal2D
from PIL import Image, ImageOps
from skimage.transform import resize

from .base import BaseDataset


class SEMDataset(BaseDataset):
    def __init__(self, file_path: str | Path, nag_file_path: str | Path = None):
        super().__init__(file_path)

        # for .bcf files:
        if str(file_path).endswith(".bcf"):
            for dataset in self.base_dataset:
                if (self.nav_img is None) and (type(dataset) is Signal2D):
                    self.original_nav_img = dataset
                    self.nav_img = dataset  # load BSE data
                elif (self.nav_img is not None) and (type(dataset) is Signal2D):
                    old_w, old_h = self.nav_img.data.shape
                    new_w, new_h = dataset.data.shape
                    if (new_w + new_h) < (old_w + old_h):
                        self.original_nav_img = dataset
                        self.nav_img = dataset
                elif type(dataset) is EDSSEMSpectrum:
                    self.original_spectra = dataset
                    self.spectra = dataset  # load spectra data from bcf file

        # for .hspy files:
        elif str(file_path).endswith(".hspy"):
            if nag_file_path is not None:
                assert str(nag_file_path).endswith(".hspy")
                nav_img = hs.load(nag_file_path)
            else:
                nav_img = Signal2D(self.base_dataset.sum(axis=2).data).T

            self.original_nav_img = nav_img
            self.nav_img = nav_img
            self.original_spectra = self.base_dataset
            self.spectra = self.base_dataset

        self.spectra.change_dtype(
            "float32"
        )  # change spectra data from unit8 into float32

        # reserve a copy of the raw data for quantification
        self.spectra_raw = self.spectra.deepcopy()

        self.feature_list = self.spectra.metadata.Sample.xray_lines
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}


class IMAGEDataset:
    def __init__(self, chemical_maps_dir: str | Path, intensity_map_path: str | Path):
        chemical_maps_paths = [
            join(chemical_maps_dir, f)
            for f in os.listdir(chemical_maps_dir)
            if not f.startswith(".")
        ]
        chemical_maps = [Image.open(p) for p in chemical_maps_paths]
        chemical_maps = [ImageOps.grayscale(p) for p in chemical_maps]
        chemical_maps = [np.asarray(img) for img in chemical_maps]

        self.chemical_maps = np.stack(chemical_maps, axis=2).astype(np.float32)
        self.intensity_map = np.asarray(
            Image.open(intensity_map_path).convert("L")
        ).astype(np.int32)

        self.chemical_maps_bin = None
        self.intensity_map_bin = None

        self.feature_list = [
            f.split(".")[0]
            for f in os.listdir(chemical_maps_dir)
            if not f.startswith(".")
        ]
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}

    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        print(f"Set feature_list to {self.feature_list}")

    def rebin_signal(self, size: tuple = (2, 2)):
        for i, maps in enumerate([self.chemical_maps, self.intensity_map]):
            w, h = maps.shape[:2]
            new_w, new_h = int(w / size[0]), int(h / size[1])
            maps = resize(maps, (new_w, new_h))
            if i == 0:
                self.chemical_maps_bin = maps
            else:
                self.intensity_map_bin = maps

    def normalisation(self, norm_list: list = []):
        self.normalised_elemental_data = (
            self.chemical_maps_bin
            if self.chemical_maps_bin is not None
            else self.chemical_maps
        )
        print("Normalise dataset using:")
        for i, norm_process in enumerate(norm_list):
            print(f"    {i + 1}. {norm_process.__name__}")
            self.normalised_elemental_data = norm_process(
                self.normalised_elemental_data
            )


class PIXLDataset(IMAGEDataset):
    def __init__(self, file_path: str | Path):
        self.base_dataset = hs.load(file_path)
        self.chemical_maps = self.base_dataset.data.astype(np.float32)
        self.intensity_map = self.base_dataset.data.sum(axis=2).astype(np.float32)
        self.intensity_map = self.intensity_map / self.intensity_map.max()

        self.chemical_maps_bin = None
        self.intensity_map_bin = None

        self.feature_list = self.base_dataset.metadata.Signal.phases
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}


class TEMDataset(BaseDataset):
    def __init__(self, file_path: str):
        super().__init__(file_path)

        if type(self.base_dataset) == Signal2D:
            self.stem = self.base_dataset
        elif type(self.base_dataset) == Signal1D:
            self.spectra = hs.load(file_path, signal_type="EDS_TEM")
            if self.spectra is None:
                raise ValueError("Failed to load EDS_TEM signal.")

            self.nav_img = Signal2D(self.spectra.data.sum(axis=2))
            self.spectra.change_dtype("float32")
            self.spectra_raw = self.spectra.deepcopy()

            self.spectra.metadata.set_item("Sample.xray_lines", [])
            self.spectra.axes_manager["Energy"].scale = 0.01 * 8.07 / 8.08
            self.spectra.axes_manager["Energy"].offset = -0.01
            self.spectra.axes_manager["Energy"].units = "keV"

            self.feature_list = []

        # if data format is .emd file
        elif type(self.base_dataset) is list:  # file_path[-4:]=='.emd' and
            try:
                # Check for sparse capability which is required for some EMD files
                import sparse
            except ImportError:
                # We don't block here, but if loading fails later with a sparse error, the user will know.
                pass

            emd_dataset = self.base_dataset
            self.nav_img = None
            for dataset in emd_dataset:
                if (self.nav_img is None) and (
                    dataset.metadata.General.title == "HAADF"
                ):
                    self.original_nav_img = dataset
                    self.nav_img = dataset  # load HAADF data
                elif isinstance(dataset, EDSTEMSpectrum):
                    self.original_spectra = dataset
                    self.spectra = dataset  # load spectra data from .emd file

            if self.spectra is None:
                raise ValueError("No EDSTEMSpectrum found in the EMD file.")

            self.spectra.change_dtype("float32")
            self.spectra_raw = self.spectra.deepcopy()

            elements = self.spectra.metadata.Sample.elements
            # Ensure elements is not None before iteration
            if elements:
                self.spectra.metadata.set_item(
                    "Sample.xray_lines", [e + "_Ka" for e in elements]
                )  # type: ignore
                self.feature_list = self.spectra.metadata.Sample.xray_lines  # type: ignore
            else:
                self.feature_list = []

            self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}

    def set_xray_lines(self, xray_lines: list[str]):
        """
        Set the X-ray lines for the spectra analysis.

        Parameters
        ----------
        xray_lines : List
            A list consisting of a series of elemental peaks. For example, ['Fe_Ka', 'O_Ka'].

        """
        self.feature_list = xray_lines
        if self.spectra is None:
            raise ValueError("spectra is None")
        self.spectra.set_lines(self.feature_list)  # type: ignore
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}
        print(f"Set xray_lines to {self.feature_list}")

    def set_axes_scale(self, scale: float):
        """
        Set the scale for the energy axis.

        Parameters
        ----------
        scale : float
            The scale of the energy axis. For example, given a data set with 1500 data points corresponding to 0-15 keV, the scale should be set to 0.01.

        """
        if self.spectra is None:
            raise ValueError("spectra is None")
        self.spectra.axes_manager["Energy"].scale = scale  # type: ignore

    def set_axes_offset(self, offset: float):
        """
        Set the offset for the energy axis.

        Parameters
        ----------
        offset : float
            the offset of the energy axis.

        """
        if self.spectra is None:
            raise ValueError("spectra is None")
        self.spectra.axes_manager["Energy"].offset = offset  # type: ignore

    def set_axes_unit(self, unit: str):
        """
        Set the unit for the energy axis.

        Parameters
        ----------
        unit : float
            the unit of the energy axis.

        """
        if self.spectra is None:
            raise ValueError("spectra is None")
        self.spectra.axes_manager["Energy"].unit = unit  # type: ignore

    def remove_NaN(self):
        """
        Remove the pixels where no values are stored.
        """
        if self.spectra is None or self.nav_img is None:
            raise ValueError("spectra or nav_img is None")

        index_NaN = np.argwhere(np.isnan(self.spectra.data[:, 0, 0]))[0][0]
        self.nav_img.data = self.nav_img.data[: index_NaN - 1, :]
        self.spectra.data = self.spectra.data[: index_NaN - 1, :, :]

        if self.nav_img_bin is not None:
            self.nav_img_bin.data = self.nav_img_bin.data[: index_NaN - 1, :]
        if self.spectra_bin is not None:
            self.spectra_bin.data = self.spectra_bin.data[: index_NaN - 1, :, :]
