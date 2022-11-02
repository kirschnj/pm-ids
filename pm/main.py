import argparse
import time

import numpy as np
import os
import shutil
import json
import pandas as pd
import uuid
from hashlib import md5


from pm import aggregate
from pm.benchmarks import Camelback, SquaredExponential
from pm.games.dueling import DuelingBandit, LocalizedDuelingBandit
from pm.strategies.e2d import E2D
from pm.strategies.gaps import GapEstimator
from pm.strategies.gcb import GCB
from pm.strategies.lls import RegularizedLeastSquares
from pm.game import GameInstance, Game, ContextualGame
from pm.games.confounded import ConfoundedToDuelingGame, ConfoundedToDuelingInstance
from pm.games.continuous import ContinuousGame, ContinuousInstance
from pm.games.noise import noise_normal, PeriodicDrift, NegativeDrift, NegativeRepeat, NegativeRepeatTwo, MinusBest, \
    AlternatingMinusBest, NegativeBernoulli, Bernoulli, PhasedOffset, PositiveRepeat, AutoCalibration
from pm.strategies.bose import DoubleRobustLLS, Bose
from pm.strategies.maxinp import MaxInP
from pm.strategies.pege import PEGE
from pm.strategies.semits import SemiparametricTS
from pm.strategies.ids import IDS
from pm.strategies.ts import TS
from pm.strategies.ucb import UCB, GPUCB
from pm.strategies.solid import Solid
from pm.utils import query_yes_no, timestamp, fixed_seed, lower_bound, to_name_dict, flatten_data_dict, get_binary_array

from pm.strategies import infogain
from pm.strategies import gaps

import logging

def noise_factory(**params):
    noise_var = params.get('noise_var')
    noise = noise_normal(noise_var)
    confounder = params.get('confounder', None)
    if confounder is not None:
        confounders = dict([(f.__name__, f) for f in CONFOUNDERS])
        confounder = confounders[confounder]
        confounder = confounder()

    return noise, confounder


def rbf(x, y):
    """
    shape (n,d)
    """
    return np.exp(- np.sum((x - y) ** 2, axis=1))


def phi(x, X0):
    """
    rbf features for point x and domain X0
    """
    return rbf(x, X0)


def compute_min_gap(X, theta):
    mean = np.inner(X,theta)
    gap = np.max(mean) - mean
    winner = np.argmin(gap)
    gap[winner] = np.inf
    return np.min(gap)


def localized_dueling(**params):
    """
    game factory for simple bandit game
    """
    noise, confounder = noise_factory(**params)
    seed = params.get('seed')
    d = params.get('d')
    normalize_actions = params.get('normalize_actions')

    theta = np.zeros(d)
    theta[0] = 1.
    theta[-1] = 1. - params['min_gap']
    X = np.eye(d)

    _id = f"localized_dueling_d{d}_v{params.get('noise_var')}"

    game = LocalizedDuelingBandit(X, name=_id)
    instance = GameInstance(game, theta=theta, noise=noise, confounder=confounder)
    return game, instance

def random_dueling(**params):
    """
    game factory for simple bandit game
    """

    noise, confounder = noise_factory(**params)
    seed = params.get('seed')
    d = params.get('d')
    k = params.get('k')
    normalize_actions = params.get('normalize_actions')

    theta = np.zeros(d)
    theta[0] = 1.


    with fixed_seed(seed):
        X = np.random.normal(size=d*k).reshape(k, d)
        if normalize_actions:
            X = (X.T / np.linalg.norm(X, axis=1)).T

        _id = f"random_dueling_d{d}_k{k}_v{params.get('noise_var')}"
        if seed is not None:
            _id += f"_s{seed}"

    game = DuelingBandit(X, name=_id)
    instance = GameInstance(game, theta=theta, noise=noise, confounder=confounder)
    return game, instance


def simple_bandit(**params):
    """
    game factory for simple bandit game
    """

    noise, confounder = noise_factory(**params)
    seed = params.get('seed')
    d = params.get('d')
    k = params.get('k')
    random_noise = params.get('random_noise')
    normalize_actions = params.get('normalize_actions')

    theta = np.zeros(d)
    theta[0] = 1.

    env_2018 = params.pop('read_2018_env')
    if env_2018:
        with open(env_2018) as f:
            data = json.load(f)
        domain = data['domain']
        X = np.array([x[0] for x in domain])
        rho = np.array([x[1] for x in domain])
        theta = np.array(data['theta'])
        if params['env_force_homoscedastic']:
            M = X
        else:
            M = (X.T/rho).T
        name = os.path.dirname(env_2018).rsplit(os.sep)[-1]
        _id = f'2018_{name}'

    else:
        with fixed_seed(seed):
            X = np.random.normal(size=d*k).reshape(k, d)
            if normalize_actions:
                X = (X.T / np.linalg.norm(X, axis=1)).T
            if random_noise:
                rho = np.random.uniform(0., random_noise, size=k)
                M = (X.T / rho).T
            else:
                M = X


        _id = f"simple_bandit_d{d}_k{k}_v{params.get('noise_var')}"
        if seed is not None:
            _id += f"_s{seed}"
        if confounder is not None:
            _id += f"_c{type(confounder).__name__}"
        if random_noise:
            _id += f'_rn{random_noise}'



    game = Game(X, M=M, name=_id)
    instance = GameInstance(game, theta=theta, noise=noise, confounder=confounder)

    return game, instance

def noise_example(**params):
    noise, _ = noise_factory(**params)
    seed = params.get('seed')
    d = params.get('d')
    rhoi = params.get('noise_example_rhoi')
    eps = params.get('noise_example_eps')
    X_base = get_binary_array(d)
    X_base = 2*X_base - np.ones_like(X_base)
    X_base = (X_base.T/np.linalg.norm(X_base, axis=1)).T
    M = np.vstack((X_base*rhoi, X_base))
    X = np.vstack((X_base, (1-eps)*X_base))


    with fixed_seed(seed):
        theta = np.random.normal(size=d)
        theta /= np.linalg.norm(theta)

    game = Game(X, M=M, name='noise_example')
    instance = GameInstance(game, theta=theta, noise=noise)

    return game, instance

def camelback(**params):
    noise, confounder = noise_factory(**params)
    _camelback = Camelback()
    points_per_dim = params.get('points_per_dim')
    label = f"camelback_v{params.get('noise_var')}"
    if params.get('confounder'):
        label += f"_c{params.get('confounder')}"
    game = ContinuousGame(_camelback.bounds, points_per_dim=points_per_dim, name=label)
    instance = ContinuousInstance(game, _camelback, noise, confounder)
    return game, instance

def se(**params):
    d = params.get('se_d', 2)
    points_per_dim = params.get('points_per_dim')
    noise, confounder = noise_factory(**params)
    _se = SquaredExponential(d=d)
    label = f"se{d}_v{params.get('noise_var')}"
    if params.get('confounder'):
        label += f"_c{params.get('confounder')}"
    game = ContinuousGame(_se.bounds, points_per_dim=points_per_dim, name=label)
    instance = ContinuousInstance(game, _se, noise, confounder)
    return game, instance

def large_gaps(**params):
    """
    game factory for a simple_bandit game with fixed design and large gap
    4 arms
    """
    # with fixed_seed(3):
    #     X = np.random.normal(size=12).reshape(6, 2)
    #     X = X / np.linalg.norm(X, axis=1)[:, None]
    noise_var = params.get('noise_var')

    X = np.array([[0.97, 0.23], [0.09, -0.9], [-0.09, 0.9], [-0.8, 0.5]])

    _id = f"large_gaps_v{noise_var}"
    game = Game(X, X, name=_id)
    instance = GameInstance(game, theta=np.array([1., 0.]), noise=noise_normal(noise_var))
    return game, instance


def eoo(**params):
    """
    game factory for the counter-example in the End of Optimism
    """
    # alpha = 0.25 such that 8\alpha\epsilon =2\epsilon as in Figure 1
    noise_var = params.get('noise_var')
    eps = params.get('eoo_eps', 0.05)
    alpha = params.get('eoo_alpha', 1.)
    X = np.array([[1.,0.],[1-eps, 8*alpha*eps],[0.,1.]])

    game = Game(X, X, name=f"counter_example_v{noise_var}_e{eps}_a{alpha}")
    instance = GameInstance(game, theta=np.array([1.,0.]), noise=noise_normal(noise_var))
    return game, instance

def contextual_simple_bandit(**params):
    noise, confounder = noise_factory(**params)
    d = params.get('d')
    k = params.get('k')
    l = params.get('l')
    seed = params.get("seed")
    normalize_actions = params.get('normalize_actions')

    theta = np.zeros(d)
    theta[0] = 1.

    X_all = []
    M_all = []
    with fixed_seed(seed):
        for i in range(l):
            X = np.random.normal(size=d * k).reshape(k, d)
            if normalize_actions:
                X = (X.T / np.linalg.norm(X, axis=1)).T
            M = X.reshape(k, 1, d)
            X_all.append(X)
            M_all.append(M)

    _id = f"contextual_simple_bandit_d{d}_k{k}_l{l}_v{params.get('noise_var')}"
    if seed is not None:
        _id += f"_s{seed}"

    X = np.array(X_all)
    M = np.array(M_all)

    game = ContextualGame(X, M=M, cdistr=np.ones(l)/l, name=_id)
    instance = GameInstance(game, theta=theta, noise=noise)

    return game, instance

class IceCreamInstance(GameInstance):

    def __init__(self, game, obs, keys):
        self.game = game
        self.obs = obs
        self._num_obs = len(obs)
        self._num_actions = keys.shape[1]
        self.keys = keys
        self._rewards = np.zeros_like(keys)

        for i, row in enumerate(keys):
            for j, key in enumerate(row):
                self._rewards[i,j] = np.mean((obs[obs[:,0] == key][:,1]).astype(float))
        super().__init__(game, None, None)

    def get_reward(self, actions=None, context=None):
        rewards = self._rewards
        if actions is not None:
            rewards = rewards[:, actions % self._num_actions]
        if context is not None:
            rewards = rewards[context]
        return rewards

    def get_noisy_observation(self, action, context=None):
        target = self.keys[context, action % self._num_actions]
        # no need to distinquish between informative / non-informative since the features are zero for the latter
        while True:
            i = np.random.randint(self._num_obs)
            if self.obs[i,0] == target:
                return np.array([float(self.obs[i,1])])/5.

def icecream(**params):
    file = params.pop('file')
    num_context = params.get('icecream_flavours')
    actions = pd.read_csv(os.path.join(file, f'icecream_actions_{num_context}.csv'), header=None).values
    obs = pd.read_csv(os.path.join(file, f'icecream_obs_{num_context}.csv'), header=None).values
    direct = params.get('icecream_direct')
    actions = actions.reshape(4, num_context, -1).swapaxes(0,1)
    X0 = actions[:,:,1:].astype(float)

    # normalize features
    for i in range(X0.shape[0]):
        X0[i] = (X0[i].T / np.linalg.norm(X0[i], axis=-1)).T

    k = X0.shape[1]
    if direct:
        X = M = X0
        M = M.reshape(num_context, k, 1, -1)
    else:
        X = np.zeros(X0.shape * np.array([1,2,1]))
        M = np.zeros(X0.shape * np.array([1,2,1]))
        X[:,:k,:] = X0
        M[:,k:,:] = X0
        M = M.reshape(num_context, 2*k, 1, -1)

    keys = actions[:,:,0].reshape(4, num_context).T
    cdistr= np.zeros(num_context)
    for review in obs:
        context = int(review[0].split('_')[1])
        cdistr[context] += 1
    cdistr = cdistr/np.sum(cdistr)
    game = ContextualGame(X,M, cdistr=cdistr, name=f'icecream_{num_context}')
    instance = IceCreamInstance(game, obs, keys)
    return game, instance


def laser(**params):
    """
    game factory for laser experiment
    """
    indirect = params['laser_indirect']
    noise, confounder = noise_factory(**params)

    grid = np.arange(-1, 1.1, 0.5)  # 1.1 to include 1
    d = len(grid) ** 2
    X0 = np.array(np.meshgrid(grid, grid)).T.reshape(d, 2)

    # these are parameter settings that can be played
    grid2 = np.arange(-0.5, 0.6, 0.5)
    X1 = np.array(np.meshgrid(grid2, grid2)).T.reshape(len(grid2) ** 2, 2)

    # compute features for invasive measurements
    A_obs = []
    for x in X1:
        # shift X1 by x
        # for each element in X1 + x, compute the feature
        A_obs.append(np.array([phi(y, X0) for y in X1 + x]))

    X_obs = np.zeros((len(A_obs), d))
    A_obs = np.array(A_obs)

    A_int = []
    X_int = []
    # compute featurs for integrated measurements
    for x in X1:
        # shift X1 by x
        # for each element in X1 + x, compute the features for each point
        # then sum up the features over the points in X1 + x to "integrate" the reward
        features = np.sum(np.array([phi(y, X0) for y in X1 + x]), axis=0).reshape(1, -1)
        features /= len(X1)
        X_int.append(features.flatten())

        if indirect:
            A_int.append(np.zeros((len(X1), d)))
        else:
            # append zero observations to have same output dimension
            features = np.vstack([features, np.zeros((len(X1) - 1, d))])
            A_int.append(features)

    X_int = np.array(X_int)
    A_int = np.array(A_int)

    X = np.vstack([X_obs, X_int])
    A = np.vstack([A_obs, A_int])

    # set features s.t. the true function is exp(-x^2)
    theta = np.zeros(d)
    theta[18] = 1.

    # create game and instance
    _id = "laser"
    if indirect:
        _id += "_indirect"

    game = Game(X, A, name=_id)
    instance = GameInstance(game, theta, noise=noise)
    return game, instance


def estimator_factory(game_, **params):
    noise_var = params.get('noise_var')
    scale_obs = params.get('lls_scale_obs', None)
    delta = params.get('delta')
    beta_asymptotic = params.get('beta_asymptotic', False)
    beta_factor = params.get('beta_factor', 1.)
    beta = params.get('lls_beta')
    lls = RegularizedLeastSquares(d=game_.d, beta_logdet=not beta_asymptotic, noise_var=noise_var, scale_obs=scale_obs, beta_factor=beta_factor, delta=delta, beta=beta)
    return lls

def ts(game_, **params):
    """
    strategy factory for UCB
    """
    lls = estimator_factory(game_, **params)
    ts_scale = params.get('ts_scale')
    strategy = TS(game_, lls=lls)
    return strategy

def ucb(game_, **params):
    """
    strategy factory for UCB
    """
    lls = estimator_factory(game_, **params)
    strategy = UCB(game_, lls, force_homoscedastic=params.get('ucb_force_homoscedastic'))
    return strategy

def e2d(game_, **params):
    """
    strategy factory for UCB
    """
    lls = estimator_factory(game_, **params)
    strategy = E2D(game_, lls, gamma_power=params.get('e2d_gamma_power'))
    return strategy

def gpucb(game_, **params):
    delta = params.get('delta')
    reg = params.get('reg')
    lengthscale = params.get('lengthscale')
    beta = params.get('beta')
    strategy = GPUCB(game_, delta=delta, reg=reg, beta=beta, lengthscale=lengthscale)
    return strategy


def dueling_kids(game_, **params):
    delta = params.get('delta')
    reg = params.get('reg')
    beta = params.get('beta')
    lengthscale = params.get('lengthscale')
    strategy = DuelingKernelIDS(game_, delta=delta, reg=reg, beta=beta, lengthscale=lengthscale)
    return strategy


def bose(game_, **params):
    delta = params.get('delta')
    robust_lls = DoubleRobustLLS(game_.d)
    strategy = Bose(game_, robust_lls, delta=delta)
    return strategy

def semiparametric_ts(game_, **params):
    delta = params.get('delta')
    strategy = SemiparametricTS(game_, delta)
    return strategy

def ids(game_, **params):
    """
    strategy factory for IDS
    params: --infogain {full,directed2,...} --dids
    """
    infogain_cls = INFOGAIN[params.get('ids_info')]
    if infogain_cls is infogain.AsymptoticInfoGain:
        info = infogain_cls(eta=params.get('ids_eta'), correction=params.get('ids_info_correction'))
    else:
        info = infogain_cls()
    gap_estimator_cls = GAP_ESTIMATORS[params.get('ids_gap')]
    gap_estimator = gap_estimator_cls(alternatives=params.get('ids_alternatives'), truncate=params.get('ids_truncate'))
    lls = estimator_factory(game_, **params)
    strategy = IDS(game_, lls, gap_estimator, info, sampling_strategy=params.get('ids_sampling'), exploit=params.get('ids_exploit'), discard_exploit_data=params.get('ids_discard'), fw_steps=params.get('ids_fw_steps'))
    return strategy

def maxinp(game_, **params):
    lls = estimator_factory(game_, **params)
    strategy = MaxInP(game_, lls)
    return strategy

def gcb(game_, **params):
    lls = estimator_factory(game_, **params)
    strategy = GCB(game_, lls)
    return strategy

def pege(game_, **params):
    lls = estimator_factory(game_, **params)
    strategy = PEGE(game_, lls, mode=params.get('pege_mode'))
    return strategy

def dueling_ids(game_, **params):
    lls, estimator = estimator_factory(game_, **params)
    strategy = DuelingIDS(game_, estimator)
    return strategy

# def asymptotic_ids(game_, **params):
#     lls, estimator = estimator_factory(game_, **params)
#
#     fast_ratio = params.get('fast_ratio', False)
#     lower_bound_gap = params.get('lower_bound_gap', False)
#     opt2 = params.get('opt2', False)
#     ucb_switch = params.get('ucb_switch', False)
#     fast_info = params.get('ids_fast_info', False)
#     alpha = params.get('alpha', 1.0)
#
#     if params.get('delta') is not None:
#         logging.warning("Setting delta has no effect for asymptotic_ids")
#     # anytime estimator
#     strategy = AsymptoticIDS(game_, estimator=estimator, fast_ratio=fast_ratio, lower_bound_gap=lower_bound_gap, opt2=opt2, alpha=alpha, ucb_switch=ucb_switch, fast_info=fast_info)
#     return strategy

def solid(game_, **params):
    lls = estimator_factory(game_, **params)
    reset = params['solid_reset']
    z0 = params.get('solid_z0', 100)
    opt = params.get('solid_opt', False)
    logging.info(f"Using solid with reset={reset}")
    noise_var = params.get('noise_var')
    strategy = Solid(game_, lls=lls, reset=reset, noise_var=noise_var, z_0=z0, opt=opt)  # default values already set
    return strategy

def confounded_reduction(game_, instance, **params):
    two_point = params.get('to_dueling') == 'two'
    compensate = params.pop('compensate')
    params['compensated'] = True
    cgame = ConfoundedToDuelingGame(game_, two_point=two_point)
    cinstance = ConfoundedToDuelingInstance(instance, two_point=two_point, compensate=compensate)
    return cgame, cinstance


# list of available strategies
STRATEGIES = to_name_dict(ucb, ts, ids, solid, dueling_ids, bose, semiparametric_ts, gpucb, dueling_kids, maxinp, gcb, pege, e2d)

# list of available info gains for IDS
INFOGAIN = to_name_dict(infogain.WorstCaseInfoGain, infogain.AsymptoticInfoGain, infogain.SampleMIInfoGain, infogain.VarInfoGain, infogain.UCBInfoGain, infogain.DirectedInfoGain)
GAP_ESTIMATORS = to_name_dict(gaps.ValueGap, gaps.FastValueGap, gaps.DiffGap, gaps.FastDiffGap, gaps.BayesianGap)

# list of available games
GAMES = to_name_dict(simple_bandit, large_gaps, eoo, camelback, se, noise_example, random_dueling, localized_dueling, contextual_simple_bandit, icecream, laser)

CONFOUNDERS = to_name_dict(PeriodicDrift, NegativeDrift, NegativeRepeat, NegativeRepeatTwo, MinusBest, AlternatingMinusBest, NegativeBernoulli, Bernoulli, PhasedOffset, PositiveRepeat, AutoCalibration)


def run(game_factory, strategy_factory, **params):
    """
    run a game
    """
    # setup game and instance
    game, instance = game_factory(**params)
    game_copy = game

    if not isinstance(instance, ContinuousInstance) and game.__class__ is not ContextualGame:
        logging.info(f"Minimum gap: {instance.min_gap():0.3f}")
    # reduction
    if params.get('to_dueling'):
        game, instance = confounded_reduction(game, instance, **params)

    # setup strategy
    strategy = strategy_factory(game, **params)


    # other parameters
    n = params.get('n')
    # lb_return = params.get('lb_return')
    outdir = params.pop('outdir')
    create_only = params.pop('create_only', False)
    delete_runs = params.pop('delete_runs', False)
    force_seed = params.pop('force_seed')
    timeit = params.pop('timeit')
    record_info_gain = params.pop('ids_record_info')

    # create a hash of the parameter configuration
    store_params = {k :v for k,v in sorted(params.items(), key=lambda i: i[0]) if v is not None}
    hash = md5(json.dumps(store_params, sort_keys=True).encode()).hexdigest()

    # outdir: game_name-{n}/strategy_name-{hash}
    game_name = str(game_copy)
    if params.get('compensate'):
        game_name = game_name + '_compensate'
    outdir = os.path.join(outdir, f"{game_name}-{n}", f"{str(strategy)}-{hash}")

    # setup output directory
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(os.path.join(outdir, 'params.json')):
        # save params
        with open(os.path.join(outdir, 'params.json'), 'w') as outfile:
            json.dump(store_params, outfile, indent=2)

    if create_only:
        print(f"Outdir is {outdir}")
        exit()

    if delete_runs:
        if query_yes_no(f"Do you want to delete the runs in {outdir}?"):
            shutil.rmtree(outdir)
            exit()

    stamp = timestamp()
    hex = uuid.uuid4().hex
    outfile = os.path.join(outdir, f"run-{stamp}-{hex}.csv")
    extra_outfile = os.path.join(outdir, f"extra-{stamp}-{hex}.csv")
    time_outfile = os.path.join(outdir, f"time-{stamp}-{hex}.csv")

    data = np.empty(shape=(n), dtype=([('regret', 'f8'), ('cum_regret', 'f8')]))  # store data for the csv file
    cumulative_regret = 0
    extra_data = dict()

    if record_info_gain:
        info_gain_functions = dict(
            info_mi_sample=infogain.SampleMIInfoGain(),
            # info_mi_laplace=infogain.LaplaceMIInfoGain(),
            info_worst_case=infogain.WorstCaseInfoGain(),
            info_ucb=infogain.UCBInfoGain(),
            info_var=infogain.VarInfoGain(),
            info_asymptotic_half_wc=infogain.AsymptoticInfoGain(correction='worst_case'),
            info_asymptotic_half_ucb=infogain.AsymptoticInfoGain(correction='ucb'),
            info_asymptotic_half_wc_cell=infogain.AsymptoticInfoGain(correction='worst_case', force_cell=True),
            info_asymptotic_half_ucb_cell=infogain.AsymptoticInfoGain(correction='ucb', force_cell=True),
            # info_asymptotic_cell_freq=infogain.AsymptoticInfoGain(),
            info_asymptotic_half_laplace=infogain.AsymptoticInfoGain(correction='none', eta=0.5),
            # info_asymptotic_cell_bays=infogain.AsymptoticInfoGain(correction='none', learning_rate=0.5),
        )
        for key in info_gain_functions.keys():
            extra_data[key] = []

    compensate = params.get('compensate') and not params.get('to_dueling')
    last_obs = 0.

    if force_seed:
        np.random.seed(force_seed)
    t_start = time.time()
    # run game
    for t in range(n):
        # call strategy
        if game.__class__ is ContextualGame:
            context = np.random.choice(game.num_context, p=game.cdistr)
            x = strategy.get_action(context=context)
        else:
            context = None
            x = strategy.get_action()

        # compute reward and regret
        if context is None:
            reward = instance.get_reward(x)
            regret = instance.max_reward() - reward
        else:
            reward = instance.get_reward(x, context)
            regret = instance.max_reward(context) - reward

        cumulative_regret += regret

        # store data : regret, cumulative regret, index of arm pulled
        data[t] = regret, cumulative_regret

        # get observation and update strategy
        if context is None:
            observation = instance.get_noisy_observation(x)
            strategy.add_observations(x, observation - last_obs)
        else:
            observation = instance.get_noisy_observation(x, context)
            strategy.add_observations(x, observation - last_obs, context)

        # print(observation, last_obs)
        # if --compensate, we subtract the last observation
        if compensate:
            last_obs = observation

        if record_info_gain:
            for key, info_gain in info_gain_functions.items():
                I = info_gain.info(strategy.gap_estimator)
                I /= np.sum(I)
                extra_data[key].append(I)

    t_total = time.time() - t_start
    # outfile
    np.savetxt(outfile, data, fmt=['%f', '%f'])
    if extra_data:
        flatten_data_dict(extra_data)
        df = pd.DataFrame.from_dict(extra_data)
        df.to_csv(extra_outfile, index=False)

    if timeit:
        np.savetxt(time_outfile, [t_total], fmt=['%f'])

    return outdir


def main():
    # # store available games and strategies as a dict
    # games = dict([(f.__name__, f) for f in GAMES])
    # strategies = dict([(f.__name__, f) for f in STRATEGIES])
    # confounders = dict([(f.__name__, f) for f in CONFOUNDERS])

    # setup argument parse
    parser = argparse.ArgumentParser(description='run a partial monitoring game.')

    # basic settings
    parser.add_argument('game', choices=GAMES.keys())
    parser.add_argument('strategy', choices=STRATEGIES.keys())
    parser.add_argument('--n', type=int, required=True, help="Horizon")
    parser.add_argument('--rep', type=int, default=1, help="Repetitions")
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--force_seed', type=int, default=None)
    parser.add_argument('--timeit', action='store_true')

    # environment
    parser.add_argument('--seed', type=int)
    parser.add_argument('--d', type=int, default=3)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--l', type=int, default=3)
    parser.add_argument('--confounder', choices=CONFOUNDERS.keys())
    parser.add_argument('--to_dueling', choices=['one', 'two'])
    parser.add_argument('--compensate', action='store_true')
    parser.add_argument('--min_gap', type=float, default=0.1)
    parser.add_argument('--noise_var', type=float, default=1.)
    parser.add_argument('--random_noise', type=float, default=None)
    parser.add_argument('--normalize_actions', action='store_true')
    parser.add_argument('--read_2018_env', type=str)
    parser.add_argument('--env_force_homoscedastic', action='store_true')
    parser.add_argument('--file', type=str, default=None)

    # environment specfic settings
    parser.add_argument('--se_d', type=int, default=2)
    parser.add_argument('--points_per_dim', type=int, default=50)

    # estimator
    parser.add_argument('--delta', type=float, default=None)
    parser.add_argument('--lls_beta', type=float, default=None)
    parser.add_argument('--beta_factor', type=float, default=1.)
    parser.add_argument('--lengthscale', type=float, default=1.0)
    parser.add_argument('--regularizer', type=float, default=1.)

    parser.add_argument('--e2d_gamma_power', type=float, default=0.5)

    # ids
    parser.add_argument('--ids_sampling', choices=['full', 'fast', 'deterministic', 'contextual'], default='fast')
    parser.add_argument('--ids_exploit', action='store_true')
    parser.add_argument('--ids_discard', action='store_true')
    parser.add_argument('--ids_alternatives', choices=['halfspace', 'cell'], default='halfspace')
    parser.add_argument('--ids_eta', type=float, default=None)
    parser.add_argument('--ids_truncate', type=float, default=None)
    parser.add_argument('--ids_fw_steps', type=int, default=100)
    parser.add_argument('--ids_info_correction', choices=['worst_case', 'ucb'], default='worst_case')
    parser.add_argument('--ids_info', choices=INFOGAIN.keys())
    parser.add_argument('--ids_gap', choices=GAP_ESTIMATORS.keys())
    parser.add_argument('--ids_record_info', action='store_true')

    # ucb
    parser.add_argument('--ucb_force_homoscedastic', action='store_true')

    parser.add_argument('--pege_mode', choices=['worst_case', 'log'], default='worst_case')

    parser.add_argument('--icecream_direct', action='store_true', default=False)
    parser.add_argument('--icecream_flavours', type=int, default=5)
    parser.add_argument('--laser_indirect', action='store_true', default=False)
    parser.add_argument('--aggr', choices=[f.__name__ for f in aggregate.AGGREGATORS])
    parser.add_argument('--lb_return', type=bool, default=False)
    # parser.add_argument('--fast_ratio', action='store_true')
    # parser.add_argument('--ids_fast_info', help="fast version of info gain", action="store_true")
    # parser.add_argument('--lower_bound_gaps', type=bool, default=False)
    # parser.add_argument('--opt2', type=bool, default=False)
    # parser.add_argument('--alpha', type=float, default=1.)


    parser.add_argument('--eoo_eps', type=float, default=0.01)
    parser.add_argument('--eoo_alpha', type=float, default=1.)
    parser.add_argument('--noise_example_rhoi', type=float, default=1.)
    parser.add_argument('--noise_example_eps', type=float, default=0.)
    parser.add_argument('-v', '--verbose', help="show info output", action="store_true")
    parser.add_argument('-vv', '--verbose2', help="show debug output", action="store_true")
    parser.add_argument('--create_only', help="only create output directory and exit", action="store_true")
    parser.add_argument('--delete_runs', help="delete runs", action="store_true")

    # parameter for lls
    parser.add_argument('--beta_asymptotic', help="set beta=2log(1/delta) + d log log (n)", action="store_true")
    parser.add_argument('--lls_scale_obs', action='store_true')
    # parameters for solid
    parser.add_argument('--solid_reset', action="store_true")
    parser.add_argument('--solid_opt', action="store_true", help="default values from paper for alpha^l, alpha^w")
    parser.add_argument('--solid_z0', type=int, default=100)

    parser.add_argument('--ts_scale', type=float, default=1.)

    # parse arguments
    args = vars(parser.parse_args())

    # set up logging
    logging.basicConfig(format='%(levelname)s: %(message)s')
    if args.pop('verbose'):
        logging.getLogger().setLevel(level=logging.INFO)
    if args.pop('verbose2'):
        logging.getLogger().setLevel(level=logging.DEBUG)

    # create game and strategy factories
    game_factory = GAMES[args['game']]
    strategy_factory = STRATEGIES[args['strategy']]

    # repetition number
    rep = args.pop('rep')

    # aggregation flag
    aggr = args.pop('aggr')

    lb_return = args['lb_return']

    # run game, possibly multiple times
    for i in range(rep):
        if rep > 1:
            logging.info(f"Running iteration {i}.")
        path = run(game_factory, strategy_factory, **args)

    if lb_return:
        n = args['n']
        game, instance = game_factory(**args)
        outdir = args['outdir']
        outdir = outdir = os.path.join(outdir, f"{str(game)}-{n}", "lb")
        # setup output directory
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, f"lb.csv")
        lb = lower_bound(game,instance)
        lbn = np.log(np.array(range(n))+1) * lb
        np.savetxt(outfile, lbn)

    # aggregate if requested pm2 laser ids --n=10000 --outdir=pm-runs/ --infogain=full
    if aggr:
        aggregator = aggregate.AGGREGATORS[[f.__name__ for f in aggregate.AGGREGATORS] == aggr]
        aggregate.aggregate(path, aggregator)


if __name__ == "__main__":
    main()
