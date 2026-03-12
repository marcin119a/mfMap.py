import os
import shutil
import pandas as pd
import numpy as np
from scipy.stats import norm

# Odpowiednik rm(list = ls()) nie jest bezpośrednio potrzebny w Pythonie, 
# ale definiujemy bazową ścieżkę
basedir = 'mfMap.py'
os.chdir(basedir)

oo = ["data", "logs", "results", "ssd", "table"]
organs = ["COADREAD"]

# Zarządzanie strukturą katalogów
for x in oo:
    if os.path.exists(x):
        shutil.rmtree(x)  # Odpowiednik unlink(recursive = T)
    for y in organs:
        path = os.path.join(basedir, x, y)
        os.makedirs(path, exist_ok=True) # Odpowiednik dir.create(recursive = T)

def gen_data():
    n = 1000  # liczba próbek
    f = 500   # liczba cech/genów
    
    sample_names = [f's_{i}' for i in range(1, n + 1)]
    gene_names = [f'g_{i}' for i in range(1, f + 1)]
    dna_names = [f'd_{i}' for i in range(1, f + 1)]
    
    # Odpowiednik qnorm(runif(n*f, min=pnorm(0), max=pnorm(1)))
    # Generuje losowe dane z uciętego rozkładu normalnego w zakresie [0, 1]
    low = norm.cdf(0)
    high = norm.cdf(1)
    
    def generate_truncated_norm(rows, cols):
        u = np.random.uniform(low, high, size=(rows, cols))
        return norm.ppf(u)

    # Tworzenie ramek danych (DataFrame)
    features_v1 = pd.DataFrame(generate_truncated_norm(f, n), 
                               index=gene_names, 
                               columns=sample_names)
    
    # features_v2 jest tworzone w R, ale w Twoim kodzie do plików trafia tylko v1
    # Zachowuję tę logikę.
    
    # Generowanie etykiet
    cms = ['CMS1', 'CMS2', 'CMS3', 'CMS4', 'NOLBL']
    labels = pd.DataFrame({
        'barcode': sample_names,
        'subtype': np.random.choice(cms, n, replace=True)
    })
    
    # Dodawanie kolumny 'type' na podstawie subtype
    labels['type'] = 'tumor'
    labels.loc[labels['subtype'] == 'NOLBL', 'type'] = 'cell'
    
    # Ustawienie indeksu (odpowiednik rownames w R)
    labels.index = labels['barcode']
    
    # Zapisywanie plików (tab-separated)
    # W R użyto row.names=T i col.names=T, co jest domyślne w pandas
    features_v1.to_csv('data/COADREAD/features_exp.txt', sep='\t')
    features_v1.to_csv('data/COADREAD/features_mut_cnv_comb.txt', sep='\t')
    labels.to_csv('data/COADREAD/dataset_labels.txt', sep='\t')

# Uruchomienie funkcji
gen_data()
