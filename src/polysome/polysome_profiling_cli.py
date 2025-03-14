import typer
from typing import List
from polysome.utils import fractionation, fractionation_set

def cli(fractionation_paths: List[str], lower_limit: float =0, upper_limit: float =1, path_to_save=None):
    fractionations = []
    for path in fractionation_paths:
        fractionations.append(fractionation(path))
    dataset = fractionation_set(fractionations)
    dataset.plot(lower_limit, upper_limit, include_fractions=True, absorbance_column="A", path_to_save=path_to_save)
    

if __name__ == "__main__":
    typer.run(cli)