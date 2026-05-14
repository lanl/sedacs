import numpy as np

# Model Weights for MLSP2 defined at beta_0 = 1500, mu_0 = 1/3

#      theta_{i,1}              theta_{i,2}             theta_{i,3}             theta_{i,4}
weights = [
[ -1.353996666080607e+00, +2.437540893737664e+00, -2.038466804171965e-01, +1.291156147536192e-11 ],
[ +1.608677030263155e+00, -2.987544634415441e-01, -6.746741346599809e-02, -7.006084032292678e-12 ],
[ -2.351636735892118e+00, +2.613813908880740e+00, -2.081115324436639e-03, -6.572840812953572e-13 ],
[ +1.810031371507046e+00, -3.733036049410451e-02, -5.442179841399564e-02, +4.318503541775794e-11 ],
[ -2.732204264661287e+00, +2.825886342002125e+00, +8.167975559885774e-03, -1.662456055798004e-11 ],
[ -1.873987421378235e+00, +2.138798318408699e+00, +3.573114610928145e-02, +4.542623791264443e-12 ],
[ +1.402261079424850e+00, +4.890736627204146e-02, -4.237220339751432e-02, +2.424648524635838e-11 ],
[ +1.713089400763156e+00, -1.034037621506357e-01, -2.438094942357169e-02, +7.873381643056254e-11 ],
[ -3.252595736449542e+00, +2.628649626717414e+00, +3.165347918592593e-02, -6.312379678783267e-11 ],
[ -1.878046620274372e+00, +2.074880349608742e+00, +3.449163838903798e-02, -6.649574392039178e-11 ],
[ +1.310464008049612e+00, +5.496499701340592e-02, +3.310087064751345e-03, +2.825980776677432e-10 ],
[ +1.563608995897714e+00, -8.606990671650539e-03, -1.301065468049911e-02, -1.424083924407828e-07 ],
[ -3.038983738437103e+00, +2.477686862799397e+00, +2.636871201950337e-02, -1.330100571490071e-07 ],
[ -1.699377329573085e+00, +2.076565284757246e+00, +3.442469802192470e-02, +4.896446563691430e-03 ],
[ +1.256940670391469e+00, +5.777573303514125e-02, +4.902596975631270e-02, -2.302054912581413e-03 ],
[ -1.390281989291434e+00, +1.966243706278270e+00, +1.910503489826946e-02, +1.606259300834474e-02 ],
[ +1.172415492052006e+00, +2.982302801604828e-02, +4.530045685625948e-02, -1.539021333689731e-02 ],
[ +1.351448045975672e+00, +5.773064380022182e-02, +3.753306163777001e-02, -1.276795931125504e-01 ],
[ -1.212734900653146e+00, +1.861660915086903e+00, +1.242573869144004e-02, -1.100276149057915e-01 ],
[ +8.610663421144649e-01, +1.048747203546148e-02, +4.701135000272372e-02, -4.946843804326570e-02 ],
[ -1.369173339062334e+00, +1.699077704335121e+00, -7.732969926664974e-03, -1.761228530611925e-01 ],
[ -1.452079661340262e+00, +1.993171071701545e+00, -1.661115887219701e-02, -5.688902388963853e-01 ],
[ +1.347262922911275e+00, +2.198074427424086e-02, +1.995615256168758e-01, +5.964487309001580e-01 ],
[ -1.114238353358810e+00, +1.896937623313116e+00, -7.280733281872145e-02, +1.530915295291056e+00 ],
[ +9.411010738180853e-01, -1.554719335561439e-01, +6.584253835393120e-02, -2.059339861429303e-01 ],
[ +8.321693583250559e-01, +5.698259888759222e-01, -4.234806053199446e-01, +5.698259888760550e-01 ]
]

beta0, mu0 = 1500, 1/3

# Gershgorin circle theorem providing minimum & maximum bounds for a real spectrum
def Gershgorin(A):

    # extract diagonal entries of matrix (NOT diagonalization)
    D = np.diag(A)

    # radii of Gershgorin circles for each 
    R = np.sum(np.abs(A),axis=1) - np.abs(D)  

    # return minimum and maximum of all bounds of all circles
    return np.min(D-R), np.max(D+R)

# defined for (emax - emin) * beta < 1000, emin < mu < emax
def mlsp2(H, beta, mu):

    # estimate bounds on spectrum of true H
    emin, emax = Gershgorin(H)
    
    # Identity matrix
    I = np.eye(H.shape[0])

    # primed variables given by Eqs. 27 - 29
    H_prime    = (emax * I - H)/(emax - emin)
    mu_prime   = (emax - mu)/(emax - emin)
    beta_prime = (emax - emin) * beta

    # condition for validity by Eq. 43
    assert beta_prime <= (2/3) * beta0
    assert emin < mu < emax

    # flip given by Eq. 44 if mu' > 0.5
    mu_switch = mu_prime > 0.5
    if mu_switch:
        H_prime  = I - H_prime
        mu_prime = 1 - mu_prime

    # H0 given by Eq. 30
    X = (H_prime - mu_prime * I)*(beta_prime/beta0) + mu0 * I
    A = np.zeros_like(X)

    # MLSP2 as given by Eq. 16
    for a,b,c,d in weights:
        A += d * X
        X = a * np.matmul(X,X) + b * X + c * I
        # for faster routines in cuBLAS see Sec. IV Numerics
    
    # corresponding flip given by Eq. 43
    return I - (A + X) if mu_switch else (A + X)

def newton(H, beta, mu_guess, N_occ, eps=1e-6):

    err = eps * 10
    while abs(err) > eps:

        D = mlsp2(H, beta, mu_guess)

        Tr  = np.trace(D)
        Tr2 = np.sum(D*D)

        # by Netwon's method and f'(H) = -beta D * (I - D)
        d_mu = np.clip( (N_occ - Tr) / (Tr - Tr2) / beta, -0.5 * abs(mu_guess), 0.5 * abs(mu_guess) )

        mu_guess += d_mu

        err = N_occ - Tr

        print(err, d_mu)

    return D

def diag(H, beta, mu_guess, N_occ, eps=1e-6):

    evals, evecs = np.linalg.eigh(H)                     

    err = eps * 10
    while abs(err) > eps:

        occupation = 1 / (1 + np.exp(beta * (evals - mu_guess)))

        Tr  = np.sum(occupation)
        Tr2 = np.sum(occupation**2)

        # by Netwon's method and f'(H) = -beta D * (I - D)
        d_mu = np.clip( (N_occ - Tr) / (Tr - Tr2) / beta, -0.5 * abs(mu_guess), 0.5 * abs(mu_guess) )

        mu_guess += d_mu

        err = N_occ - Tr

        print(err, d_mu)

    return (evecs * occupation ) @ evecs.T

# test cases
if __name__ == '__main__':

    for i in range(10):

        # seed random number generation for reproducibility and fix matrix size
        rng = np.random.default_rng(seed=i)
        N = 100

        # generate random matrix of size (N,N) and make it symmetric
        H = rng.uniform(0, 1, (N,N))                            
        H = H + H.T

        # estimate spectral bounds
        emin, emax = Gershgorin(H)

        # choose beta and mu such that beta' and mu' fall inside region of validity
        beta = rng.uniform(1, (2/3 * beta0)/(emax - emin))
        f_occ = rng.uniform(0.3,0.7)

        mu_guess = emax * f_occ + emin * (1-f_occ)

        # construct density matrix with MLSP2
        D_mlsp2 = newton(H, beta, mu_guess, N*f_occ)

        # compare to diagonalization
        D_diag = diag(H, beta, mu_guess, N*f_occ)

        # print matrix 2-norm of error (maximal eigenvalue difference)
        print( np.linalg.norm(D_diag - D_mlsp2, ord=2) )
