import numpy as np
from matplotlib import pylab as plt
from torch.linalg import svd
from PIL import Image
import math
import torch
from skimage import io
import torch.nn.functional as F
# from tensorflow.python.ops.gen_array_ops import diag, transpose
from rpca_ADMM import rpcaADMM

torch.random.seed()
device = torch.device("cuda")



class TRPCA:

    def converged(self, L, E, X, L_new, E_new):
        '''
        judge convered or not
        '''
        condition1 = torch.max(abs(L_new - L))
        condition2 = torch.max(abs(E_new - E))
        condition3 = torch.max(abs(L_new + E_new - X))

        return max(condition1, condition2, condition3)

    def SoftShrink(self, X, tau):
        '''
        apply soft thesholding
        '''
        #z = torch.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)
        z = torch.max(abs(X) - tau, torch.zeros_like(X))
        #z = abs(X) - tau
        return z

    def SVDShrink2(self, Y, tau):
        '''
        for heterophilic datasets, tensor along 0-way
        apply tensor-SVD and soft thresholding
        '''
        [n1, n2, n3] = Y.shape
        XX = torch.complex(torch.empty(size=Y.shape), torch.empty(size=Y.shape)).to(device)
        # for i in range(n1):
        #    Y[i, :, :] = F.normalize(Y[i, :, :].float(), p=1, dim=0)
        X = torch.fft.fft(Y)
        for i in range(n1):
           X[i, :, :] = F.normalize(X[i, :, :].float(), p=1, dim=0)
        #Z =torch.matmul(X[:,:,0],X[:,:,0])
        tnn = 0
        trank = 0

        #X[:, :, 0] = F.normalize(X[:, :, 0].float(), p=1, dim=0)
        U, S, V = svd(X[0, :, :])
        print("rank_before = {}".format(len(S)))
        r = torch.count_nonzero(S > tau)
        if r == 0: r = r + 1
        print("rank_after = {}".format(r))
        S = S.type(torch.complex64)
        if r >= 1:
            S = torch.diag(S[0:r])
            # print(XX[:,:,0].dtype,U[:,0:r].dtype, S.dtype,V[0:r,:].dtype)
            XX[0, :, :] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
            tnn += tnn + torch.sum(S)
            trank = max(trank, r)

        halfn3 = round(n1 / 2)
        for i in range(1, halfn3):
            #X[:, :, i] = F.normalize(X[:, :, i].float(), p=1, dim=0)
            U, S, V = svd(X[i, :, :])
            r = torch.count_nonzero(S > tau)
            S = S.type(torch.complex64)
            if r == 0: r = r + 1
            if r >= 1:
                S = torch.diag(S[0:r])
                XX[i, :, :] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
                tnn += tnn + torch.sum(S) * 2
                trank = max(trank, r)

            XX[n1-i, :, :] = XX[i, :, :].conj()

        if n1 % 2 == 0:
            i = halfn3
            #X[:, :, i] = F.normalize(X[:, :, i].float(), p=1, dim=0)
            U, S, V = svd(X[i, :, :])
            r = torch.count_nonzero(S > tau)
            S = S.type(torch.complex64)
            if r == 0: r=r+1
            if r >= 1:
                S = torch.diag(S[0:r])
                XX[i, :, :] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
                tnn += tnn + torch.sum(S)
                trank = max(trank, r)

        tnn = tnn / n1
        XX = torch.fft.ifft(XX).real
        #XX = F.normalize(torch.abs(XX), p=2, dim=1)
        return XX, tnn



    def  SVDShrink2_D0(self, Y, tau):
        '''
        for homophilic datasets , construct tensor along 0-way
        apply tensor-SVD and soft thresholding
        '''
        [n1, n2, n3] = Y.shape
        XX = torch.complex(torch.empty(size=Y.shape), torch.empty(size=Y.shape)).to(device)
        #X = Y#torch.fft.fft(Y)
        X = torch.fft.fft(Y)
        ifft_XX = torch.empty(size=Y.shape).to(device)
        tnn = 0
        trank = 0

        U, S, V = svd(torch.fft.fft(Y[0, :, :]))
        print("rank_before = {}".format(len(S)))
        #r = torch.count_nonzero(S > tau)
        r = len(S)
        if r == 0: r = r + 1
        print("rank_after = {}".format(r))
        S = S.type(torch.complex64)
        if r >= 1:
            #S = torch.diag(S[0:r] - tau)
            S = torch.diag(S[0:r]) #- 1/tau
            XX[0, :, :] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
            ifft_XX[0, :, :] = torch.fft.ifft(XX[0, :, :]).real #XX[:, :, 0]#
            #ifft_XX[:, :, 0] = Y[:, :, 0]
            tnn += tnn + torch.sum(S.type(torch.complex64))
            trank = max(trank, r)

        halfn3 = round(n1 / 2)
        for i in range(1, halfn3):
            U, S, V = svd(X[i, :, :])
            r = torch.count_nonzero(S > tau)
            #r = len(S)
            S = S.type(torch.complex64)
            if r == 0: r = r + 1
            if r >= 1:
                S = torch.diag(S[0:r]-tau)
                #S = torch.diag(S[0:r])
                XX[i, :, :] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
                ifft_XX[i, :, :] = torch.fft.ifft(XX[i, :, :]).real
                tnn += tnn + torch.sum(S) * 2
                trank = max(trank, r)

            XX[n1 - i, :, :] = XX[i, :, :].conj()
            ifft_XX[n1 - i, :, :] = torch.fft.ifft(XX[n1 - i, :, :]).real

        if n1 % 2 == 0:
            i = halfn3
            U, S, V = svd(X[i, :, :])
            r = torch.count_nonzero(S > tau)
            #r = len(S)
            S = S.type(torch.complex64)
            if r == 0: r=r+1
            if r >= 1:
                S = torch.diag(S[0:r]-tau)
                XX[i, :, :] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
                ifft_XX[i, :, :] = torch.fft.ifft(XX[i, :, :]).real
                tnn += tnn + torch.sum(S)
                trank = max(trank, r)

        tnn = tnn / n1
        XX = torch.fft.ifft(XX).real
        #XX = F.normalize(torch.abs(XX), p=2, dim=1)
        return ifft_XX, tnn

    def SVDShrink2_D2(self, Y, tau):
        '''
        for homophilic datasets , construct tensor along 2-way
        apply tensor-SVD and soft thresholding
        '''
        [n1, n2, n3] = Y.shape
        XX = torch.complex(torch.empty(size=Y.shape), torch.empty(size=Y.shape)).to(device)
        #X = Y#torch.fft.fft(Y)
        X = torch.fft.fft(Y)
        ifft_XX = torch.empty(size=Y.shape).to(device)
        tnn = 0
        trank = 0

        # U, S, V = svd(torch.fft.fft(X[:, :, 0]))
        U, S, V = svd(X[:, :, 0])
        print("rank_before = {}".format(len(S)))
        #r = torch.count_nonzero(S > tau)
        r = torch.count_nonzero(S)
        if r == 0: r = r + 1
        print("rank_after = {}".format(r))
        S = S.type(torch.complex64)
        if r >= 1:
            #S = torch.diag(S[0:r] - tau)
            S = torch.diag(S[0:r]) #- 1/tau
            XX[:, :, 0] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
            #ifft_XX[:, :, 0] = torch.fft.ifft(XX[:, :, 0]).real #XX[:, :, 0]#
            #ifft_XX[:, :, 0] = Y[:, :, 0]
            tnn = tnn + torch.sum(S.type(torch.complex64))
            trank = max(trank, r)

        halfn3 = round(n3 / 2)
        for i in range(1, halfn3):
            U, S, V = svd(X[:, :, i])
            r = torch.count_nonzero(S > tau)
            S = S.type(torch.complex64)
            if r == 0: r = r + 1
            if r >= 1:
                #S = torch.diag(S[0:r] - tau)
                S = torch.diag(S[0:r])
                XX[:, :, i] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
                #ifft_XX[:, :, i] = torch.fft.ifft(XX[:, :, i]).real
                tnn = tnn + torch.sum(S) * 2
                trank = max(trank, r)

            XX[:, :, n3 - i] = XX[:, :, i].conj()
            #ifft_XX[:, :, n3 - i] = torch.fft.ifft(XX[:, :, n3 - i]).real

        if n3 % 2 == 0:
            i = halfn3
            U, S, V = svd(X[:, :, i])
            r = torch.count_nonzero(S > tau)
            S = S.type(torch.complex64)
            if r == 0: r=r+1
            if r >= 1:
                S = torch.diag(S[0:r] - tau)
                XX[:, :, i] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
                #ifft_XX[:, :, i] = torch.fft.ifft(XX[:, :, i]).real
                tnn = tnn + torch.sum(S)
                trank = max(trank, r)

        tnn = tnn / n3
        XX = torch.fft.ifft(XX).real
        #XX = F.normalize(torch.abs(XX), p=2, dim=1)
        return XX, tnn

    def SVDShrink3(self, Y, tau):
        '''
        for chameleon and squirrel
        apply tensor-SVD and soft thresholding
        '''
        [n1, n2, n3] = Y.shape
        XX = torch.complex(torch.empty(size=Y.shape), torch.empty(size=Y.shape)).to(device)
        X = torch.fft.fft(Y)
        ifft_XX = torch.empty(size=Y.shape).to(device)
        tnn = 0
        trank = 0

        try:
            U, S, V = svd(X[0, :, :])
        except torch._C._LinAlgError as e:
            X[0, :, :][X[0, :, :] != X[0, :, :]] = 0
            inf_mask = torch.isinf(X[0, :, :])
            X[0, :, :][inf_mask] = torch.tensor(1e6)
            try:
                U, S, V = svd(X[0, :, :])
            except torch._C._LinAlgError as e:
                import scipy
                U, S, V = scipy.linalg.svd(X[0, :, :].cpu())
                U = torch.tensor(U)
                S = torch.tensor(S)
                V = torch.tensor(V)
            print("存在nan，已转化为0")
        print("rank_before = {}".format(len(S)))
        XX[0, :, :] = torch.fft.fft(Y[0, :, :])
        res = rpcaADMM(XX[0, :, :].cpu().numpy())['X3_admm']
        ifft_XX[0, :, :] = torch.fft.ifft(torch.from_numpy(res).type(torch.complex64)).real * 0#XX[:, :, 0]#

        halfn1 = round(n1 / 2)
        for i in range(1, halfn1):
            try:
                U, S, V = svd(X[i, :, :])
            except torch._C._LinAlgError as e:
                X[i, :, :][X[i, :, :] != X[i, :, :]] = 0
                try:
                    U, S, V = svd(X[i, :, :])
                    print("存在nan，已转化为0")
                except torch._C._LinAlgError as e:
                    inf_mask = torch.isinf(X[i, :, :])
                    X[i, :, :][inf_mask] = torch.finfo(torch.float32).max
                    import scipy
                    try:
                        U, S, V = scipy.linalg.svd(X[i, :, :].cpu())
                    except torch._C._LinAlgError as e:
                        print('x中有无穷大，已处理，并使用scipy内置算法svd')
                    U = torch.tensor(U)
                    S = torch.tensor(S)
                    V = torch.tensor(V)
            # r = torch.count_nonzero(S > tau)
            S = S.type(torch.complex64)
            tnn = torch.tensor(tnn, dtype=torch.complex64).cuda()
            if torch.isinf(tnn.real) and torch.isinf(tnn.imag):
                tnn = torch.tensor([100000000.+100000000.j], dtype=torch.complex64).cuda()
                tnn = tnn + torch.sum(S) * 2
            elif torch.isinf(tnn.real) and torch.isnan(tnn.imag):
                tnn = torch.tensor([100000000.+0.j], dtype=torch.complex64).cuda()
                tnn = tnn + torch.sum(S) * 2
            elif torch.isnan(tnn.real) and torch.isinf(tnn.imag):
                tnn = torch.tensor([0.+1e8.j], dtype=torch.complex64).cuda()
                tnn = tnn + torch.sum(S) * 2
            elif torch.isnan(tnn.real) and torch.isnan(tnn.imag):
                tnn = torch.tensor([0.+0.j], dtype=torch.complex64).cuda()
                tnn = tnn + torch.sum(S) * 2
            else:
                tnn += tnn + torch.sum(S) * 2
            XX[i, :, :] = torch.fft.fft(Y[i, :, :])
            res = rpcaADMM(XX[i, :, :].cpu().numpy())['X3_admm']
            ifft_XX[i, :, :] = torch.fft.ifft(torch.from_numpy(res).type(torch.complex64)).real * 0.1 * (i * 2)

            ifft_XX[n1 - i, :, :] = ifft_XX[i, :, :].conj()

        if n1 % 2 == 0:
            i = halfn1
            try:
                U, S, V = svd(X[i, :, :])
            except torch._C._LinAlgError as e:
                X[i, :, :][X[i, :, :] != X[i, :, :]] = 0
                inf_mask = torch.isinf(X[i, :, :])
                X[i, :, :][inf_mask] = torch.finfo(torch.float32).max
                try:
                    U, S, V = svd(X[i, :, :])
                except torch._C._LinAlgError as e:
                    import scipy
                    U, S, V = scipy.linalg.svd(X[i, :, :].cpu())
                    U = torch.tensor(U)
                    S = torch.tensor(S)
                    V = torch.tensor(V)
            r = torch.count_nonzero(S > tau)
            S = S.type(torch.complex64)
            if r == 0: r=r+1
            if r >= 1:
                S = torch.diag(S[0:r])
                XX[i, :, :] = torch.matmul(torch.matmul(U[:, 0:r], S), V[0:r, :])
                ifft_XX[i, :, :] = torch.fft.ifft(XX[i, :, :]).real
                tnn += tnn + torch.sum(S)
                trank = max(trank, r)
        tnn = tnn / n1
        #XX = torch.fft.ifft(XX).real
        return ifft_XX, tnn

    def T_SVD(self, Y, k=50):
        print('=== t-SVD: rank={} ==='.format(k))
        try:
            [n1, n2, n3] = Y.shape
            XX = torch.complex(torch.empty(size=Y.shape), torch.empty(size=Y.shape)).to(device)
            X = torch.fft.fft(Y)

            U, S, V = torch.svd(X[:, :, 0])
            # print(U,S,V)
            print("rank_before = {}".format(len(S)))
            S = S.type(torch.complex64)
            #k = round(len(S)*0.6)+1
            if k >= 1:
                S = torch.diag(S[0:k])
                # print(XX[:,:,0].dtype,U[:,0:r].dtype, S.dtype,V[0:r,:].dtype)
                XX[:, :, 0] = torch.matmul(torch.matmul(U[:, 0:k], S), V[:, :k].T)

            halfn3 = round(n3 / 2)
            for i in range(1, halfn3):
                U, S, V = torch.svd(X[:, :, i])
                S = S.type(torch.complex64)
                # k = round(len(S) * 0.6) + 1
                if k >= 1:
                    S = torch.diag(S[0:k])
                    XX[:, :, i] = torch.matmul(torch.matmul(U[:, 0:k], S), V[:, :k].T)

                XX[:, :, n3 - i] = XX[:, :, i].conj()

            if n3 % 2 == 0:
                i = halfn3
                U, S, V = torch.svd(X[:, :, i])
                S = S.type(torch.complex64)
                #k = round(len(S) * 0.6) + 1
                if k >= 1:
                    S = torch.diag(S[0:k])
                    XX[:, :, i] = torch.matmul(torch.matmul(U[:, 0:k], S), V[:, :k].T)

            XX = torch.fft.ifft(XX).real
            # XX[XX<0]=0
            print("rank_after = {}".format(k))
            return XX, True#XX
        except:
            print("pass")
            return Y, False#torch.zeros_like(Y)

    def ADMM(self, X):
        '''
        Solve
        min (nuclear_norm(L)+lambda*l1norm(E)), subject to X = L+E
        L,E
        by ADMM
        '''
        m, n, l = X.shape
        eps = 1e-4
        rho = 1.1
        mu = 1e-2
        mu_max = 1e10
        max_iters = 5#10#200
        lamb = 1 / math.sqrt(max(m, n) * l)
        #lamb = 1 / math.sqrt(max(n,l) * m)
        L = torch.zeros((m, n, l)).to(device) # 低秩
        E = torch.zeros((m, n, l)).to(device) # 噪声
        Y = torch.zeros((m, n, l)).to(device) # 拉格朗日乘子
        iters = 0
        while True:
            iters += 1

            # update L(recovered image)
            #L_new, tnn = self.SVDShrink2_D0(X - E - (1 / mu) * Y, 1 / mu)
            L_new, tnn = self.SVDShrink3(X - E - (1 / mu) * Y, 1 / mu)

            # update E(noise)
            E_new = self.SoftShrink(X - L_new - (1 / mu) * Y, lamb / mu)
            #for wisconsin/texas/cornell
            #E_new = X - L_new

            dY = L_new + E_new - X
            Y += mu * dY
            mu = min(rho * mu, mu_max)

            if self.converged(L, E, X, L_new, E_new).item() < eps or iters >= max_iters:
                return L_new, E_new
            else:
                L, E = L_new, E_new
                L = torch.nan_to_num(L)
                E = torch.nan_to_num(E)
                obj1, obj2 = 0, 0
                for j in range(l):
                    obj1 += torch.norm(E[:, :, j], p=1)
                    obj2 += torch.norm(dY[:, :, j], p=2)


                if iters == 1 or iters % 10 == 0:
                    print(iters, ": ", "mu=", mu, "obj=", (tnn + lamb * obj1).item()
                          , "err=", obj2.item(), "chg=", self.converged(L, E, X, L_new, E_new).item())
                if np.isnan(self.converged(L, E, X, L_new, E_new).item()) or np.isnan(obj2.item()):
                    return X,False

                if iters > 50 and obj2.item()>5:
                    return X, False




