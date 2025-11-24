import pandas as pd
import polysome.utils as poly
import numpy as np


class QFeatures:
    def __init__(self, abundances: pd.DataFrame, col_info: pd.DataFrame, row_info: pd.DataFrame, go_res_all: pd.DataFrame = None):
        self._abundances = abundances
        self._col_info = col_info
        self._row_info = row_info

        self.col_info = col_info
        self.row_info = row_info

        self.go_res_all = go_res_all

        self.normalized = False

    def reset_to_original(self):
        self.row_info = self._row_info
        self.col_info = self._col_info

    def normalize_abundances(self, norm_factor):
        
        self._abundances = self._abundances / norm_factor
        self.normalized=True
    @property
    def abundances(self):
        cols = self.col_info.index
        return self._abundances.loc[self.row_info.index, :].iloc[:, self.col_info.index]
    
    def filter_rows(self, row_condition, inplace=False, **kwargs):
        if inplace:
            self.row_info = self._row_info.query(row_condition, **kwargs)
        else:
            row_info = self.row_info.query(row_condition, **kwargs)
            
            return QFeatures(
                abundances=self._abundances,
                col_info=self.col_info,
                row_info=row_info,
                go_res_all=self.go_res_all
            )


    def filter_cols(self, col_condition, inplace=False, **kwargs):
        if inplace:
            self.col_info = self._col_info.query(col_condition, **kwargs)
        else:
            col_info = self.col_info.query(col_condition, **kwargs)
    
            return QFeatures(
                abundances=self._abundances,
                col_info=col_info,
                row_info=self.row_info,
                go_res_all=self.go_res_all
            )
        
    def filter_rows_by_GO(self, query, inplace=False):

        accessions_for_GO = self.go_res_all.query(query).UNIPROTKB
        accessions_for_GO_STR = "['" + "', '".join(accessions_for_GO) + "']"

        

        res = self.filter_rows(f"Accession.isin({accessions_for_GO_STR})", inplace=inplace)

        if not inplace:
            return res



class PolysomeProfileWithProteins:
    def __init__(self, fractionation: poly.fractionation, qfeatures: QFeatures, name: str = None):
        self.fractionation = fractionation
        self.frac_data = fill_in_fraction_numbers(fractionation.data.to_pandas())
        self.qfeatures = qfeatures
        self.name = name

        self.make_heatmap(inplace=True)

    @property
    def extent(self):
        return [0, self.frac_data.CumulativeVolume_ml.max(), 0, self.qfeatures.abundances.shape[0]]

    def make_heatmap(self, row_condition: None | str = None, inplace: bool = False):
        if row_condition is  None:
            qfeatures_for_heatmap = self.qfeatures
        else:
            qfeatures_for_heatmap = self.qfeatures.filter_rows(row_condition=row_condition)
        heatmap = np.zeros((qfeatures_for_heatmap.abundances.shape[0], self.frac_data.shape[0]))
        for col_ind, col in enumerate(qfeatures_for_heatmap.col_info.itertuples()):
            fractions = col.Fractions
            fraction_region = self.frac_data.loc[self.frac_data.FilledInFractionNumber.isin(fractions)].index.tolist()
            for row_ind, val in enumerate(qfeatures_for_heatmap.abundances.iloc[:, col_ind]):
                heatmap[row_ind, fraction_region] += val / len(fractions)
            
        extent = [0, self.frac_data.CumulativeVolume_ml.max(), 0, qfeatures_for_heatmap.abundances.shape[0]]
        
        if inplace:
            self.heatmap = heatmap
        else:
            return heatmap, extent, qfeatures_for_heatmap.row_info["Gene Symbol"][::-1]
        
    def filter_rows(self, query, inplace=False):
        res = self.qfeatures.filter_rows(query, inplace=inplace)
        if inplace:
            self.make_heatmap(inplace=True)
        else:
            new_instance = PolysomeProfileWithProteins(
                fractionation=self.fractionation,
                qfeatures=res,
                name=self.name
            )
            return new_instance
    def filter_rows_by_GO(self, query, inplace=False):
        res = self.qfeatures.filter_rows_by_GO(query, inplace=inplace)
        if inplace:
            self.make_heatmap(inplace=True)
        else:
            new_instance = PolysomeProfileWithProteins(
                fractionation=self.fractionation,
                qfeatures=res,
                name=self.name
            )
            return new_instance


    def plot(self, row_condition=None, ax=None, ymin=0, ymax=1, log_transform=False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        if row_condition is None:
            hm, ext, row_names = self.heatmap, self.extent, self.qfeatures.row_info["Gene Symbol"]
        else:
            hm, ext, row_names = self.make_heatmap(row_condition=row_condition)

        if log_transform:
            hm = np.log2(hm)
        
        self.fractionation.plot(ymin=ymin, ymax=ymax, ax=ax[0], label='0h')
        ax[1].imshow(hm, extent=ext, cmap="gray_r", aspect="auto", interpolation="none", **kwargs)
        ax[1].hlines(y=np.arange(1, hm.shape[0]), xmin=0, xmax=ext[1], colors='lightgray', linewidth=0.5)
        ax[1].set_yticks(ticks=np.arange(len(row_names)))
        ax[1].set_yticklabels(labels=row_names[::-1])


def fill_in_fraction_numbers(data):
    frac_boundary_index = np.where(~data.FractionNumber.isna())[0]
    frac_boundary_val = data.FractionNumber.iloc[frac_boundary_index].tolist()

    frac_nums = np.full(len(data), np.nan)
    lower_index = 0
    higher_index = frac_boundary_index[0]
    for ind, num in enumerate(frac_boundary_val):
        frac_nums[lower_index:higher_index+1] = frac_boundary_val[ind]
        lower_index = frac_boundary_index[ind]+1
        higher_index = frac_boundary_index[ind+1] if ind+1 < len(frac_boundary_index) else len(data)

    data['FilledInFractionNumber'] = frac_nums
    return data