import argparse
import numpy as np
import pandas as pd

import sklearn as sci
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import lightgbm as gbm
import scipy 
import matplotlib as mpl
from pathlib import Path
import lightgbm as gbm


def elastRegression(C, l1Ratio):
    return Pipeline([
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(solver='saga', C=C, l1Ratio=l1Ratio,
                                  max_iter=5000, tol=1e-4))
    ])



class GBM:

    def __init__(self):
        pass



class analyzerBlender:
    
    def __init__(self):
        pass


##################      Dataframe Creation for Quicker Processing       ##################
# CSV cols: id, predicted_ancestry, breast_cancer (target)
ID, ANC, TARGET = 'id', 'predicted_ancestry', 'breast_cancer'
SCRIPT_DIR = Path(__file__).resolve().parent

def loadData(trainCSV, testCSV):
    '''Return train, test DataFrames; caches each CSV as .parquet next to it.'''
    out = []
    for csv in (Path(trainCSV), Path(testCSV)):
        pq = csv.with_suffix('.parquet')            # data/train.csv -> data/train.parquet
        if pq.exists():
            df = pd.read_parquet(pq)
        else:
            cols = pd.read_csv(csv, nrows=0).columns
            dtypes = {c: 'float32' for c in cols if c.startswith('ZSCORE_')}
            dtypes[ANC] = 'category'
            df = pd.read_csv(csv, dtype=dtypes)
            df.to_parquet(pq, index=False)
        out.append(df)
    train, test = out
    train[TARGET] = train[TARGET].astype('int8')
    return train, test


def buildFeatureSets(train, test, addConsensus=True):
    '''
    Input:  train, test DataFrames from loadData()
    Output: featureSets {'A'|'B'|'C': (Xtr, Xte, cols)}, y, ancTr, ancTe, test_ids

        y        <- train['breast_cancer']
        ancTr/Te <- train/test['predicted_ancestry']
        test_ids <- test['id']

        A = ZSCORE_breast_cancer-*   B = any PRS col containing 'breast'   C = every ZSCORE_*
        Every set gets ancestry one-hot (+ optional consensus mean of set A).
        Columns chosen by name only -> no label leakage.
    '''
    ######  Shared arrays identical for every feature set   ######
    y = train[TARGET].to_numpy(dtype=np.int8)
    ancTr = train[ANC].to_numpy(dtype=str)   # fixed-width str (npz-safe)
    ancTe = test[ANC].to_numpy(dtype=str)
    test_ids = test[ID].to_numpy()

    ######  Choose PRS cols by name only (label-free)   ######
    prs = [c for c in train.columns if c.startswith('ZSCORE_')]
    setCols = {
        'A': [c for c in prs if c.startswith('ZSCORE_breast_cancer-')],
        'B': [c for c in prs if 'breast' in c],
        'C': prs,
    }

    ######  Ancestry one-hot; levels fixed from train so cols align   ######
    ancLevels = sorted(pd.unique(ancTr))
    def oneHot(anc):
        return np.stack([(anc == lvl) for lvl in ancLevels], axis=1).astype(np.float32)
    ohTr, ohTe = oneHot(ancTr), oneHot(ancTe)
    ohCols = [f'anc_{lvl}' for lvl in ancLevels]

    ######  Consensus score: row mean of the breast_cancer PRS   ######
    if addConsensus:
        consTr = train[setCols['A']].mean(axis=1).to_numpy(np.float32)[:, None]
        consTe = test[setCols['A']].mean(axis=1).to_numpy(np.float32)[:, None]

    ######  Assemble each set   ######
    featureSets = {}
    for name, cols in setCols.items():
        partsTr = [train[cols].to_numpy(np.float32), ohTr]
        partsTe = [test[cols].to_numpy(np.float32), ohTe]
        names = cols + ohCols
        if addConsensus:
            partsTr.append(consTr)
            partsTe.append(consTe)
            names = names + ['breast_cancer_PRS_mean']

        Xtr, Xte = np.hstack(partsTr), np.hstack(partsTe)
        assert Xtr.shape[0] == len(y) and Xte.shape[0] == len(test_ids)
        assert Xtr.shape[1] == Xte.shape[1] == len(names)
        featureSets[name] = (Xtr, Xte, names)
        print(f'set {name}: Xtr {Xtr.shape}, Xte {Xte.shape}')

    return featureSets, y, ancTr, ancTe, test_ids


def getFeatureSets(trainCSV, testCSV, cache, addConsensus=True):
    '''Load feature sets from the .npz cache if present; otherwise build and save it.
    Delete the cache file whenever buildFeatureSets() or --noConsensus changes.'''
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        featureSets = {
            s: (z[f'Xtr_{s}'], z[f'Xte_{s}'], z[f'cols_{s}'].tolist())
            for s in ('A', 'B', 'C')
        }
        print(f'loaded feature sets from {cache}')
        return featureSets, z['y'], z['ancTr'], z['ancTe'], z['test_ids']

    train, test = loadData(trainCSV, testCSV)
    featureSets, y, ancTr, ancTe, test_ids = buildFeatureSets(train, test, addConsensus)

    arrays = {'y': y, 'ancTr': ancTr, 'ancTe': ancTe, 'test_ids': test_ids}
    for s, (Xtr, Xte, cols) in featureSets.items():
        arrays[f'Xtr_{s}'] = Xtr
        arrays[f'Xte_{s}'] = Xte
        arrays[f'cols_{s}'] = np.array(cols)
    np.savez(cache, **arrays)
    print(f'saved feature sets to {cache}')
    return featureSets, y, ancTr, ancTe, test_ids
    

def makeFolds(y, ancTr, seed, nFolds=5):
    '''
    Input:  y <- train['breast_cancer]              (n_train,)
            ancTr <- train['predicted_ancestry']    (n_train,)
            seed <- args.seed
    Output: folds: list of nFolds (trainIdx, valIdx) integer index arrays
    Stratifies on target x ancestry so every fold has the same case rate and the same 
    share of each anc (keeps afr/amr in every fold)
    build once & pass the same folds to every model fo OOF predictions align
    '''

    strata = np.char.add(ancTr.astype(str), '_' + y.astype(str))

    labels, counts = np.unique(strata, return_counts=True)
    small = labels[counts<nFolds]
    if small.size:
        print(f'  merging small strata into target-only labels: {list(small)}')
        strata = np.where(np.isin(strata, small), 'RARE_' + y.astype(str), strata)
    
    skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), strata))

    allVal = np.concatenate([va for _, va in folds])
    assert np.array_equal(np.sort(allVal), np.arange(len(y))), 'each row must be validated exactly once'
    for k, (tr, va) in enumerate(folds):
        assert np.intersect1d(tr, va).size == 0
        print(f'  fold {k}: train {len(tr):,}  val {len(va):,}  '
              f'val case rate {y[va].mean():.3f}  '
              'val ancestry ' + ' '.join(f'{a}:{n}' for a, n in zip(*np.unique(ancTr[va], return_counts=True))))

    return folds


def main():
    parser = argparse.ArgumentParser(description='BME205 PRS breast cancer prediction')
    parser.add_argument('--trainCSV', type=Path, default=SCRIPT_DIR / 'data' / 'train.csv')
    parser.add_argument('--testCSV',  type=Path, default=SCRIPT_DIR / 'data' / 'test.csv')
    parser.add_argument('--output',   type=Path, default=SCRIPT_DIR / 'submission.csv')
    parser.add_argument('--seed',     type=int,  default=205)
    parser.add_argument('--noConsensus', action='store_true',
                        help='drop the breast_cancer_PRS_mean feature')
    parser.add_argument('--rebuild', action='store_true',
                        help='ignore the feature-set cache and rebuild it')
    args = parser.parse_args()

    addConsensus = not args.noConsensus

    ######  Feature sets (CSV -> Parquet -> arrays, cached as .npz)   ######
    # Separate cache per consensus setting, so toggling the flag never loads stale features
    cache = args.trainCSV.parent / f'featuresets_cons{int(addConsensus)}.npz'
    if args.rebuild and cache.exists():
        cache.unlink()

    featureSets, y, ancTr, ancTe, test_ids = getFeatureSets(
        args.trainCSV, args.testCSV, cache, addConsensus=addConsensus)

    print(f'y: {len(y):,} samples, case fraction {y.mean():.3f}')
    print(f'test ids: {len(test_ids):,}')

    ######  Make folds   ######
    folds = makeFolds(y, ancTr, args.seed)

    ######  Models   ######
    # baseline LR, elastRegression, GBM -> OOF + test predictions
    model = elastRegression(C=0.05, l1Ratio=0.5).fit()

    ######  Analyze + blend   ######

    ######  Write submission   ######
    # writeSubmission(test_ids, preds, args.output)


if __name__ == '__main__':
    main()