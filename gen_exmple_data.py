import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import norm

# Set base path, but do not change working directory
basedir = Path("data_fake")

oo = ["data", "logs", "results", "ssd", "table"]
organs = ["COADREAD"]

# Manage directory structure
for x in oo:
    dir_path = basedir / x
    if dir_path.exists():
        shutil.rmtree(dir_path)  # Odpowiednik unlink(recursive = T)
    for y in organs:
        path = dir_path / y
        path.mkdir(parents=True, exist_ok=True)  # Odpowiednik dir.create(recursive = T)

def gen_data():
    n = 1000  # number of samples
    f = 500   # number of features/genes
    
    sample_names = [f's_{i}' for i in range(1, n + 1)]
    gene_names = [f'g_{i}' for i in range(1, f + 1)]
    dna_names = [f'd_{i}' for i in range(1, f + 1)]
    
    # Equivalent of qnorm(runif(n*f, min=pnorm(0), max=pnorm(1)))
    # Generates random data from a truncated normal distribution in [0, 1]
    low = norm.cdf(0)
    high = norm.cdf(1)
    
    def generate_truncated_norm(rows, cols):
        u = np.random.uniform(low, high, size=(rows, cols))
        return norm.ppf(u)

    # Create data frames
    features_v1 = pd.DataFrame(generate_truncated_norm(f, n), 
                               index=gene_names, 
                               columns=sample_names)
    
    # features_v2 is created in the R version, but only v1 is written to files
    # Keep the same logic here.
    
    # Generate labels
    cms = ['CMS1', 'CMS2', 'CMS3', 'CMS4', 'NOLBL']
    labels = pd.DataFrame({
        'barcode': sample_names,
        'subtype': np.random.choice(cms, n, replace=True)
    })
    
    # Add 'type' column based on subtype
    labels['type'] = 'tumor'
    labels.loc[labels['subtype'] == 'NOLBL', 'type'] = 'cell'
    
    # Set index (equivalent of rownames in R)
    labels.index = labels['barcode']
    
    # Save tab-separated files
    # In R, row.names=T and col.names=T are used; this is the default in pandas
    out_dir = basedir / "COADREAD"
    out_dir.mkdir(parents=True, exist_ok=True)
    features_v1.to_csv(out_dir / "features_exp.txt", sep="\t")
    features_v1.to_csv(out_dir / "features_mut_cnv_comb.txt", sep="\t")
    labels.to_csv(out_dir / "dataset_labels.txt", sep="\t")

# Uruchomienie funkcji
gen_data()
