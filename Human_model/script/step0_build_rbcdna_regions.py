import argparse
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

def existing_cols(df, cols):
    return [c for c in cols if c in df.columns]

def row_median(df, cols):
    if not cols:
        return pd.Series(np.nan, index=df.index)
    num = df[cols].apply(pd.to_numeric, errors='coerce')
    return num.median(axis=1, skipna=True)

def main(args):
    gDNA15 = ['GLGHD1069', 'GLGHD0046', 'GLGHD0053', 'GLGHD0014', 'GLGHD0015', 'GLGHD0058', 'GLGHD0822', 'GLGHD1049', 'GLGHD1111', 'GLGHD0068']
    trn_hd = ['GLRHD1069', 'GLRHD0046', 'GLRHD0053', 'GLRHD0014', 'GLRHD0015', 'GLRHD0058', 'GLRHD0822', 'GLRHD1049', 'GLRHD1111', 'GLRHD0068']
    base_dir = './rbcDNA_regions'
    all_samples = gDNA15 + trn_hd
    n_perm = 1000

    cpm_nor_all = []
    sub = args.tab_id

    path = f"{base_dir}/train.tab.{sub}"
    cpm = pd.read_table(path, sep='\t', header=0, dtype=str)

    cpm.columns = cpm.columns.str.replace('.uniq.nodup.bam', '', regex=False)
    cpm.columns = cpm.columns.str.replace("'", "", regex=False)
    cpm.columns = cpm.columns.str.replace('"', "", regex=False)

    cols_hd = existing_cols(cpm, trn_hd)
    cols_gDNA = existing_cols(cpm, gDNA15)

    cpm['median.cpm'] = row_median(cpm, cols_hd) / sub
    cpm['median.gDNA'] = row_median(cpm, cols_gDNA) / sub

    cpm['FC_hdvsgDNA'] = cpm['median.cpm'] / cpm['median.gDNA']
    eps = 1e-6
    cpm['logFC_hdvsgDNA'] = np.log2(cpm['median.cpm'] + eps) - np.log2(cpm['median.gDNA'] + eps)
    obs_logFC = cpm['logFC_hdvsgDNA'].values.reshape(-1, 1)

    cols_exist = existing_cols(cpm, all_samples)

    if len(cols_exist) == 20:
        X = cpm[cols_exist].apply(pd.to_numeric, errors="coerce")
        perm_FC = []
        for seed in range(n_perm):
            rng = np.random.default_rng(seed)
            rand_hd = (
                    rng.choice([s for s in gDNA15 if s in cols_exist], 5, replace=False).tolist() +
                    rng.choice([s for s in trn_hd if s in cols_exist], 5, replace=False).tolist()
            )
            rand_gDNA = [s for s in cols_exist if s not in rand_hd]

            median_hd = X[rand_hd].median(axis=1, skipna=True)
            median_gDNA = X[rand_gDNA].median(axis=1, skipna=True)

            perm_logFC = np.log2(median_hd + eps) - np.log2(median_gDNA + eps)
            perm_FC.append(perm_logFC)

        perm_FC = np.stack(perm_FC, axis=1)

        med = np.nanmedian(perm_FC, axis=1)
        mad = 1.4826 * np.nanmedian(np.abs(perm_FC - med[:, None]), axis=1)
        z_robust = (obs_logFC.ravel() - med) / (mad + 1e-12)
        p_robust = 2 * (1 - stats.norm.cdf(np.abs(z_robust)))
        cpm['pval_perm'] = p_robust

        pvals = cpm['pval_perm'].values
        reject, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
        cpm['pval_adj'] = pvals_adj
        cpm['significant'] = reject

    out_cols = ['#chr', 'start', 'end', 'median.cpm', 'median.gDNA', 'FC_hdvsgDNA', 'logFC_hdvsgDNA', 'pval_perm', 'pval_adj', 'significant']
    cpm_tmp = cpm[out_cols].copy()
    cpm_nor_all.append(cpm_tmp)

    cpm_nor_all = pd.concat(cpm_nor_all, axis=0, ignore_index=True)
    output_file = f'{base_dir}/region_fc_part.bed.{args.tab_id}'
    cpm_nor_all.to_csv(output_file, sep='\t', header=False, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tab_id", type=int, default=100, required=True)
    args = parser.parse_args()
    main(args)