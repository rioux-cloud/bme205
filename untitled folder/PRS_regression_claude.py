import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr

import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

import lightgbm as lgb

import matplotlib
matplotlib.use('Agg')                 # write PNGs without opening a window
import matplotlib.pyplot as plt


##################      Constants       ##################
# CSV cols: id, predicted_ancestry, breast_cancer (target), ZSCORE_* (PRS)
ID, ANC, TARGET = 'id', 'predicted_ancestry', 'breast_cancer'
SCRIPT_DIR = Path(__file__).resolve().parent

# sklearn >= 1.8 infers the elastic-net penalty from l1_ratio; older versions need penalty='elasticnet'
_SK_MAJOR, _SK_MINOR = (int(v) for v in sklearn.__version__.split('.')[:2])
_NEEDS_PENALTY_ARG = (_SK_MAJOR, _SK_MINOR) < (1, 8)

ENET_CS = np.logspace(-3, 1, 7)               # inverse penalty strength
ENET_L1_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)  # 0 = ridge, 1 = LASSO


##################      Models       ##################
def baselineLR():
    '''Plain logistic regression (sklearn default L2, C=1): the reference to beat.'''
    return Pipeline([
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(max_iter=5000)),
    ])


def elastRegression(C, l1Ratio):
    '''Elastic-net logistic regression. Scaler is fit inside each fold (no leakage).
    lambda = 1 / (C * n_train),  alpha = l1Ratio  (see ElasticNet_explained.pdf)'''
    kw = dict(solver='saga', C=C, l1_ratio=l1Ratio, max_iter=5000, tol=1e-4)
    if _NEEDS_PENALTY_ARG:
        kw['penalty'] = 'elasticnet'
    return Pipeline([
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(**kw)),
    ])


class GBM:
    '''
    LightGBM wrapper with an sklearn-like interface, so runCV treats it like elastRegression.

    fit(X, y):        X, y <- training-fold rows from runCV. For each seed, a stratified
                      esFrac of those rows is held out for early stopping (the CV
                      validation fold is never touched), then lgb.train() builds a booster.
    predict_proba(X): seed-averaged P(case), returned as an (n, 2) array [1-p, p].
    importance():     mean gain importance across seeds, indexed by cols.
    '''

    DEFAULT_PARAMS = dict(
        objective='binary', metric='auc',
        learning_rate=0.02, num_leaves=15, max_depth=4, min_data_in_leaf=100,
        feature_fraction=0.2, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=1.0, lambda_l1=0.0, verbose=-1,
    )

    def __init__(self, params=None, seeds=(0, 1, 2), esFrac=0.1, cols=None,
                 numThreads=8, maxRounds=5000, esRounds=100, monotone=False):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.seeds = tuple(seeds)
        self.esFrac = esFrac
        self.cols = cols
        self.numThreads = numThreads
        self.maxRounds = maxRounds
        self.esRounds = esRounds
        self.monotone = monotone

    def _monotoneConstraints(self, nFeatures):
        # +1 (risk never decreases with the score) on breast-cancer PRS and the consensus mean
        if self.cols is None:
            raise ValueError('monotone=True needs cols')
        return [1 if 'breast_cancer' in c else 0 for c in self.cols][:nFeatures]

    def fit(self, X, y):
        self.boosters, self.bestIters = [], []
        for s in self.seeds:
            Xfit, Xes, yfit, yes = train_test_split(
                X, y, test_size=self.esFrac, stratify=y, random_state=s)
            params = {**self.params, 'seed': s, 'num_threads': self.numThreads}
            if self.monotone:
                params['monotone_constraints'] = self._monotoneConstraints(X.shape[1])
            # feature names left out on purpose: some trait names contain characters
            # LightGBM rejects; importances are mapped back to self.cols by position
            dtrain = lgb.Dataset(Xfit, label=yfit)
            dvalid = lgb.Dataset(Xes, label=yes, reference=dtrain)
            booster = lgb.train(
                params, dtrain,
                num_boost_round=self.maxRounds,
                valid_sets=[dvalid], valid_names=['es'],
                callbacks=[lgb.early_stopping(self.esRounds, verbose=False),
                           lgb.log_evaluation(0)],
            )
            self.boosters.append(booster)
            self.bestIters.append(booster.best_iteration)
        return self

    def predict_proba(self, X):
        p = np.mean([b.predict(X, num_iteration=b.best_iteration) for b in self.boosters], axis=0)
        return np.column_stack([1 - p, p])

    def importance(self):
        gain = np.mean([b.feature_importance(importance_type='gain') for b in self.boosters], axis=0)
        index = self.cols if self.cols is not None else range(len(gain))
        return pd.Series(gain, index=index).sort_values(ascending=False)


##################      Cross-validation       ##################
def runCV(modelFactory, Xtr, Xte, y, folds, name='', verbose=True):
    '''
    Input:  modelFactory <- e.g. lambda: elastRegression(C, r)   (fresh model per fold)
            Xtr, Xte     <- featureSets[s][0], featureSets[s][1]
            y, folds     <- getFeatureSets(), makeFolds()
    Output: dict with
            oof      (n_train,) out-of-fold P(case), each row predicted by a model that never saw it
            test     (n_test,)  mean of the fold models' test predictions
            foldAucs (nFolds,)  AUC on each validation fold
            oofAuc   float      AUC over all OOF predictions
            models   list of the fitted fold models (coefficients / importances)
    '''
    oof = np.zeros(len(y), dtype=np.float64)
    test = np.zeros(len(Xte), dtype=np.float64)
    foldAucs, models = [], []
    t0 = time.time()
    for k, (tr, va) in enumerate(folds):
        model = modelFactory()
        model.fit(Xtr[tr], y[tr])
        oof[va] = model.predict_proba(Xtr[va])[:, 1]
        test += model.predict_proba(Xte)[:, 1] / len(folds)
        foldAucs.append(roc_auc_score(y[va], oof[va]))
        models.append(model)
    res = dict(oof=oof, test=test, foldAucs=np.array(foldAucs),
               oofAuc=roc_auc_score(y, oof), models=models)
    if verbose:
        print(f'  {name:<22} OOF AUC {res["oofAuc"]:.5f}   folds {np.round(res["foldAucs"], 4)}'
              f'   ({time.time() - t0:.0f}s)')
    return res


def tuneEnet(Xtr, Xte, y, folds, Cs=ENET_CS, l1Ratios=ENET_L1_RATIOS, nJobs=-1, label=''):
    '''
    Grid search over (C, l1Ratio) by OOF AUC; grid points run in parallel (joblib).
    Selection rule: among settings within 1 fold-SD of the best OOF AUC, take the smallest C
    (strongest regularization = safest on the private leaderboard).
    Output: (best result dict incl. 'C' and 'l1Ratio', grid DataFrame)
    '''
    grid = [(C, r) for C in Cs for r in l1Ratios]
    print(f'  tuning elastic net {label}: {len(grid)} settings x {len(folds)} folds')

    def one(C, r):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')          # ConvergenceWarnings from saga
            res = runCV(lambda: elastRegression(C, r), Xtr, Xte, y, folds, verbose=False)
        res['C'], res['l1Ratio'] = C, r
        res['nonzero'] = int(np.mean([(m['lr'].coef_ != 0).sum() for m in res['models']]))
        return res

    results = Parallel(n_jobs=nJobs)(delayed(one)(C, r) for C, r in grid)

    table = pd.DataFrame([{
        'C': r['C'], 'l1Ratio': r['l1Ratio'], 'oofAuc': r['oofAuc'],
        'foldMean': r['foldAucs'].mean(), 'foldSd': r['foldAucs'].std(ddof=1),
        'nonzero': r['nonzero'],
    } for r in results]).sort_values('oofAuc', ascending=False).reset_index(drop=True)

    top = table.iloc[0]
    nearTop = table[table['oofAuc'] >= top['oofAuc'] - top['foldSd']]
    pick = nearTop.sort_values(['C', 'oofAuc'], ascending=[True, False]).iloc[0]
    best = next(r for r in results if r['C'] == pick['C'] and r['l1Ratio'] == pick['l1Ratio'])
    print(f'  best {label}: C={best["C"]:.4g} l1Ratio={best["l1Ratio"]} '
          f'OOF AUC {best["oofAuc"]:.5f} nonzero {best["nonzero"]} '
          f'(top of grid {top["oofAuc"]:.5f})')
    if best['C'] in (min(Cs), max(Cs)):
        print(f'  WARNING: best C={best["C"]:.4g} is at the edge of the grid; '
              f'extend ENET_CS past it and rerun')
    return best, table


##################      Analysis + blending       ##################
class analyzerBlender:
    '''
    Input:  results  {name: runCV() dict}  - every model must use the SAME folds
            y, ancTr <- getFeatureSets();  folds <- makeFolds()
            baseline name of the reference model;  outDir where CSV/PNG outputs go
    '''

    def __init__(self, results, y, ancTr, folds, baseline='baseline', outDir=SCRIPT_DIR / 'results'):
        self.results = results
        self.y, self.ancTr, self.folds = y, ancTr, folds
        self.baseline = baseline
        self.outDir = Path(outDir)
        self.outDir.mkdir(parents=True, exist_ok=True)

    def summary(self):
        '''OOF AUC, fold mean/SD, and paired per-fold gain vs the baseline. -> cv_results.csv'''
        base = self.results[self.baseline]['foldAucs']
        rows = []
        for name, r in self.results.items():
            d = r['foldAucs'] - base                     # paired: same folds, shared noise cancels
            rows.append(dict(
                model=name, oofAuc=r['oofAuc'],
                foldMean=r['foldAucs'].mean(), foldSd=r['foldAucs'].std(ddof=1),
                deltaVsBase=r['oofAuc'] - self.results[self.baseline]['oofAuc'],
                pairedMean=d.mean(), pairedSd=d.std(ddof=1), foldsWon=int((d > 0).sum()),
            ))
        df = pd.DataFrame(rows).sort_values('oofAuc', ascending=False).reset_index(drop=True)
        df['passesRule'] = (df['pairedMean'] > df['pairedSd']) & (df['foldsWon'] >= 4)
        df.to_csv(self.outDir / 'cv_results.csv', index=False)
        return df

    def byAncestry(self):
        '''AUC within each ancestry group. -> auc_by_ancestry.csv'''
        rows = []
        for name, r in self.results.items():
            for g in sorted(np.unique(self.ancTr)):
                m = self.ancTr == g
                auc = roc_auc_score(self.y[m], r['oof'][m]) if len(np.unique(self.y[m])) == 2 else np.nan
                rows.append(dict(model=name, ancestry=g, n=int(m.sum()),
                                 nCases=int(self.y[m].sum()), auc=auc))
        df = pd.DataFrame(rows)
        df.to_csv(self.outDir / 'auc_by_ancestry.csv', index=False)
        return df.pivot(index='model', columns='ancestry', values='auc')

    def correlation(self):
        '''Spearman rank correlation between models' OOF predictions.'''
        names = list(self.results)
        M = np.array([[spearmanr(self.results[a]['oof'], self.results[b]['oof'])[0]
                       for b in names] for a in names])
        return pd.DataFrame(M, index=names, columns=names)

    def blend(self, names, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        '''
        Rank-average two models: w*rank(A) + (1-w)*rank(B), ranks scaled to (0, 1].
        w chosen on OOF AUC; an honest check picks w on 4 folds and scores the 5th.
        Adds the blend to self.results so summary()/pickFinal() include it.
        '''
        a, b = names
        rOof = {n: rankdata(self.results[n]['oof']) / len(self.y) for n in names}
        rTest = {n: rankdata(self.results[n]['test']) / len(self.results[n]['test']) for n in names}

        def mix(r, w):
            return w * r[a] + (1 - w) * r[b]

        gridAucs = {w: roc_auc_score(self.y, mix(rOof, w)) for w in grid}
        wBest = max(gridAucs, key=gridAucs.get)

        # honest check: choose w without the fold being scored
        cvOof = np.zeros(len(self.y))
        allIdx = np.arange(len(self.y))
        for tr, va in self.folds:
            wk = max(grid, key=lambda w: roc_auc_score(self.y[tr], mix(rOof, w)[tr]))
            cvOof[va] = mix(rOof, wk)[va]
        cvAuc = roc_auc_score(self.y, cvOof)

        oof, test = mix(rOof, wBest), mix(rTest, wBest)
        name = f'blend({a},{b})'
        self.results[name] = dict(
            oof=oof, test=test,
            foldAucs=np.array([roc_auc_score(self.y[va], oof[va]) for _, va in self.folds]),
            oofAuc=gridAucs[wBest], models=[],
        )
        print(f'  blend {a} / {b}: w={wBest}  OOF AUC {gridAucs[wBest]:.5f}  '
              f'(w chosen per fold: {cvAuc:.5f})   grid ' +
              ' '.join(f'{w}:{v:.4f}' for w, v in gridAucs.items()))
        return dict(name=name, w=wBest, oofAuc=gridAucs[wBest], cvAuc=cvAuc,
                    oof=oof, test=test, gridAucs=gridAucs)

    def plotRoc(self, names=None):
        '''ROC curves from OOF predictions. -> roc.png'''
        names = names or list(self.results)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for n in names:
            fpr, tpr, _ = roc_curve(self.y, self.results[n]['oof'])
            ax.plot(fpr, tpr, lw=1.4, label=f'{n} (AUC {self.results[n]["oofAuc"]:.4f})')
        ax.plot([0, 1], [0, 1], ls='--', lw=1, color='grey')
        ax.set(xlabel='False positive rate', ylabel='True positive rate',
               title='Out-of-fold ROC (5-fold CV)')
        ax.legend(fontsize=7, loc='lower right')
        fig.tight_layout()
        path = self.outDir / 'roc.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def pickFinal(self, summary=None):
        '''Highest OOF AUC among models that pass the paired rule; else the baseline.'''
        df = summary if summary is not None else self.summary()
        passing = df[df['passesRule'] & (df['model'] != self.baseline)]
        name = passing.iloc[0]['model'] if len(passing) else self.baseline
        if not len(passing):
            print('  no model clearly beats the baseline; falling back to baseline')
        return name, self.results[name]['test']


##################      Data loading       ##################
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
        for s, (Xtr, Xte, _) in featureSets.items():
            print(f'set {s}: Xtr {Xtr.shape}, Xte {Xte.shape}')
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
    Input:  y <- train['breast_cancer']              (n_train,)
            ancTr <- train['predicted_ancestry']     (n_train,)
            seed <- args.seed
    Output: folds: list of nFolds (trainIdx, valIdx) integer index arrays
    Stratifies on target x ancestry so every fold has the same case rate and the same
    share of each ancestry (keeps AFR/AMR in every fold).
    Build once & pass the same folds to every model so OOF predictions align.
    '''
    strata = np.char.add(ancTr.astype(str), '_' + y.astype(str))

    labels, counts = np.unique(strata, return_counts=True)
    small = labels[counts < nFolds]
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


##################      Output       ##################
def writeSubmission(test_ids, preds, path):
    '''id,breast_cancer CSV; checks row count, NaNs and [0, 1] range before writing.'''
    preds = np.asarray(preds, dtype=np.float64)
    assert len(preds) == len(test_ids), 'one prediction per test id'
    assert np.isfinite(preds).all(), 'NaN/inf in predictions'
    assert (preds >= 0).all() and (preds <= 1).all(), 'predictions must be in [0, 1]'
    pd.DataFrame({ID: test_ids, TARGET: preds}).to_csv(path, index=False)
    print(f'wrote {path}  ({len(preds):,} rows)')


def savePreds(res, path):
    '''Save a runCV result (without fitted models) so later runs can reuse it.'''
    np.savez(path, oof=res['oof'], test=res['test'], foldAucs=res['foldAucs'],
             oofAuc=res['oofAuc'])


def loadPreds(path):
    z = np.load(path)
    return dict(oof=z['oof'], test=z['test'], foldAucs=z['foldAucs'],
                oofAuc=float(z['oofAuc']), models=[])


##################      Main       ##################
def main():
    parser = argparse.ArgumentParser(description='BME205 PRS breast cancer prediction')
    parser.add_argument('--trainCSV', type=Path, default=SCRIPT_DIR / 'data' / 'train.csv')
    parser.add_argument('--testCSV',  type=Path, default=SCRIPT_DIR / 'data' / 'test.csv')
    parser.add_argument('--output',   type=Path, default=SCRIPT_DIR / 'submission.csv')
    parser.add_argument('--outDir',   type=Path, default=SCRIPT_DIR / 'results')
    parser.add_argument('--seed',     type=int,  default=205)
    parser.add_argument('--noConsensus', action='store_true',
                        help='drop the breast_cancer_PRS_mean feature')
    parser.add_argument('--rebuild', action='store_true',
                        help='ignore the feature-set cache and rebuild it')
    parser.add_argument('--enetSets', nargs='*', default=['A', 'B'],
                        help='feature sets to tune elastic net on (C is slow)')
    parser.add_argument('--gbmSet', default='B', help="feature set for LightGBM, or 'none'")
    parser.add_argument('--gbmThreads', type=int, default=8,
                        help='LightGBM threads = physical performance cores (8 M1 Max, 16 13700K)')
    parser.add_argument('--nJobs', type=int, default=-1, help='parallel elastic-net grid fits')
    parser.add_argument('--quick', action='store_true',
                        help='tiny grid + 1 GBM seed, for a fast smoke test')
    parser.add_argument('--reuse', action='store_true',
                        help='load saved model predictions from outDir instead of refitting')
    args = parser.parse_args()

    addConsensus = not args.noConsensus
    args.outDir.mkdir(parents=True, exist_ok=True)
    predDir = args.outDir / 'preds'
    predDir.mkdir(exist_ok=True)

    ######  Feature sets (CSV -> Parquet -> arrays, cached as .npz)   ######
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
    results = {}

    def cached(name, fit):
        '''Reuse saved predictions with --reuse; otherwise fit and save them.'''
        path = predDir / f'{name}_cons{int(addConsensus)}_seed{args.seed}.npz'
        if args.reuse and path.exists():
            res = loadPreds(path)
            print(f'  {name:<22} OOF AUC {res["oofAuc"]:.5f}   (reused {path.name})')
            return res
        res = fit()
        savePreds(res, path)
        return res

    print('\n== baseline ==')
    XtrA, XteA, _ = featureSets['A']
    results['baseline'] = cached('baseline', lambda: runCV(
        baselineLR, XtrA, XteA, y, folds, name='baseline (LR, set A)'))

    Cs, ratios = (ENET_CS[::3], (0.0, 0.5, 1.0)) if args.quick else (ENET_CS, ENET_L1_RATIOS)
    for s in args.enetSets:
        print(f'\n== elastic net, set {s} ==')
        Xtr, Xte, cols = featureSets[s]

        def fitEnet():
            best, table = tuneEnet(Xtr, Xte, y, folds, Cs, ratios, nJobs=args.nJobs, label=f'set {s}')
            table.to_csv(args.outDir / f'enet_grid_{s}.csv', index=False)
            coefs = np.mean([m['lr'].coef_.ravel() for m in best['models']], axis=0)
            pd.Series(coefs, index=cols).loc[lambda c: c != 0] \
              .sort_values(key=np.abs, ascending=False) \
              .to_csv(args.outDir / f'enet_coefs_{s}.csv', header=['coef'])
            return best
        results[f'enet_{s}'] = cached(f'enet_{s}', fitEnet)

    if args.gbmSet.lower() != 'none':
        s = args.gbmSet
        print(f'\n== LightGBM, set {s} ==')
        Xtr, Xte, cols = featureSets[s]
        seeds = (0,) if args.quick else (0, 1, 2)

        def fitGbm():
            res = runCV(lambda: GBM(cols=cols, seeds=seeds, numThreads=args.gbmThreads),
                        Xtr, Xte, y, folds, name=f'gbm (set {s})')
            print(f'  trees per fold (seed-avg best_iteration): '
                  f'{[int(np.mean(m.bestIters)) for m in res["models"]]}')
            imp = pd.concat([m.importance() for m in res['models']], axis=1).mean(axis=1)
            imp.sort_values(ascending=False).to_csv(args.outDir / f'gbm_importance_{s}.csv',
                                                    header=['gain'])
            return res
        results[f'gbm_{s}'] = cached(f'gbm_{s}', fitGbm)

    ######  Analyze + blend   ######
    print('\n== analysis ==')
    ab = analyzerBlender(results, y, ancTr, folds, baseline='baseline', outDir=args.outDir)

    enetNames = [n for n in results if n.startswith('enet_')]
    gbmNames = [n for n in results if n.startswith('gbm_')]
    if enetNames and gbmNames:
        bestEnet = max(enetNames, key=lambda n: results[n]['oofAuc'])
        ab.blend([bestEnet, gbmNames[0]])

    summary = ab.summary()
    with pd.option_context('display.width', 140, 'display.float_format', '{:.5f}'.format):
        print(summary.to_string(index=False))
        print('\nAUC by ancestry:')
        print(ab.byAncestry().to_string())
        print('\nSpearman correlation of OOF predictions:')
        print(ab.correlation().round(3).to_string())
    print(f'\nROC plot: {ab.plotRoc()}')

    ######  Write submission   ######
    finalName, preds = ab.pickFinal(summary)
    print(f'\nfinal model: {finalName}  (OOF AUC {results[finalName]["oofAuc"]:.5f})')
    writeSubmission(test_ids, preds, args.output)


if __name__ == '__main__':
    main()
