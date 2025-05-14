import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import os
import warnings

class fractionation:
    def __init__(self, data_path, name=None):
        self.data = self._data_parser(data_path)
        if name is None:
            name = data_path.split("/")[-1].split(".")[0]
        
        self.name = name

    def _data_parser(self, data_path):
        reached_data = False
        os.makedirs("temp", exist_ok=True)
        temp_data_path = "temp/temp_data.csv"
        temp_metadata_path = "temp/temp_metadata.csv"
        self.wavelengths_in_nm = {}
        with open(data_path, 'r') as file:
            if os.path.exists(temp_data_path):
                os.remove(temp_data_path)
            if os.path.exists(temp_metadata_path):
                os.remove(temp_metadata_path)
            for line in file:
                if reached_data:
                    with open(temp_data_path, 'a') as temp:
                        temp.write(line.replace(" ", ""))
                        continue
                else:
                    with open(temp_metadata_path, 'a') as temp:
                        temp.write(line)
                        if line.startswith("Channel A (LED1) Wavelength:"):
                            self.wavelengths_in_nm["A"] = line.split(":")[1].replace("nm", "").replace("\n", "").replace(" ", "")
                        if line.startswith("Channel B (LED2) Wavelength:"):
                            self.wavelengths_in_nm["B"] = line.split(":")[1].replace("nm", "").replace("\n", "").replace(" ", "")
                        if line == "Data Columns:\n":
                            reached_data = True

        return pl.read_csv(temp_data_path, null_values=["A"])
    
    def _get_fractions(self):
        fraction_positions = \
        self.data[[i for 
                    i, val in enumerate(self.data.get_column("FractionNumber")) 
                    if val is not None], 
                    "Position"].to_list()
        fraction_positions = [0] + fraction_positions

        self.fraction_labels_positions = np.diff(fraction_positions) / 3 + fraction_positions[:-1]
        self.fraction_labels_text = self.data[[i for 
                                            i, val in enumerate(self.data.get_column("FractionNumber")) 
                                            if val is not None], 
                                            "FractionNumber"].to_list()
        
        self.fraction_positions = fraction_positions

    def _n_gaussians(self, x, *params):
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
            mu = params[i]
            sigma = params[i+1]
            amplitude = params[i+2]
            mu = np.exp(mu)
            sigma = np.exp(sigma)
            amplitude = np.exp(amplitude)
            y += amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        return y

    def _get_gaussian_parameters(self, mu_array, sigma_array, amplitude_array, absorbance_column="A"):
        yval="Abs" + absorbance_column
        reformatted_initial_values = np.log(np.asarray([[x, y, z] for x, y, z in zip(mu_array, sigma_array, amplitude_array)]).flatten())
        vals = opt.curve_fit(self._n_gaussians, self.data.get_column("Position").to_numpy(), self.data.get_column(yval).to_numpy(), p0=reformatted_initial_values)
        
        opt_mu = np.exp(vals[0][0::3])
        opt_sigma = np.exp(vals[0][1::3])
        opt_amplitude = np.exp(vals[0][2::3])
        return vals, [opt_mu, opt_sigma, opt_amplitude]
    
    def _get_gaussian_curves(self, mu, sigma, amplitude, c="black"):
        x = np.linspace(self.data.get_column("Position").min(), self.data.get_column("Position").max(), 1000)
        # x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
        y = amplitude * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        return [x, y]

    

    def plot(self, ymin, ymax, x_offset=0, y_offset=0, absorbance_column="A", include_fractions=False, label="gradient", path_to_save="temp/temp.svg", ax=None):
        if ax is None:
            _, ax = plt.subplots()

        yval="Abs" + absorbance_column
        ax.plot(self.data["Position"] + x_offset, self.data[yval] + y_offset, label=label)

        if include_fractions:
            if x_offset != 0:
                warnings.warn(f"Fractions are plotted with an offset of 0 and will not be accurate for any x-offset profiles. Hence the fractions are not accurate for {self.name}")
            self._get_fractions()
            for i, pos in enumerate(self.fraction_positions):
                ax.vlines(x=pos, ymin=ymin, ymax=ymin + 0.1 * (ymax - ymin), color="black", linestyle="-")
            for i in range(len(self.fraction_labels_positions)):
                ax.text(self.fraction_labels_positions[i], ymin + 0.11 * (ymax - ymin), s=self.fraction_labels_text[i], size=8, rotation=90)
        

        ax.set_ylim(ymin, ymax)
        ax.set_xticks(ticks=[])
        ax.set_xticklabels("")
        ax.set_xlabel("")
        ax.set_ylabel(f"Absorbance at {self.wavelengths_in_nm[absorbance_column]} nm")
        if path_to_save is not None:
            plt.savefig(path_to_save, dpi=300, transparent=True)
        if ax is None:
            plt.show()

    
            
            

class fractionation_set:
    def __init__(self, fractionation_list):
        self.fractionation_list = fractionation_list
    
    def plot(self, ymin, ymax, x_offsets=None, y_offsets=None, absorbance_column="A", include_fractions=False, path_to_save=None):
        fig, ax = plt.subplots()
        if x_offsets is None:
            x_offsets = [0] * len(self.fractionation_list)
        if y_offsets is None:
            y_offsets = [0] * len(self.fractionation_list)
        for i, fractionation in enumerate(self.fractionation_list):
            fractionation.plot(ymin, ymax, x_offsets[i], y_offsets[i], absorbance_column, include_fractions, label=fractionation.name, ax=ax)

        self.plotted_wavelengths = [frac.wavelengths_in_nm[absorbance_column] for frac in self.fractionation_list]
        if len(np.unique(self.plotted_wavelengths)) == 1:
            ax.set_ylabel(f"Absorbance at {self.plotted_wavelengths[0]} nm")
        else:
            warnings.warn(f"The absorbance wavelength plotted is not the same for all samples.")
            ax.set_ylabel(f"Absorbance {absorbance_column}")
               
        plt.legend()
        if path_to_save is not None:
            plt.savefig(path_to_save, dpi=300, transparent=True)
        plt.show()