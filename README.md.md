
# Tracking performance clusters over time
--- 

This repository contains an extract from part of my master thesis, in which I used cluster analysis to study employee performance over time in the Mercedes-Benz CAC. This section proposes a novel approach to re-estimate the cluster parameters from the model-based clustering for each point in time. This enables us to study the development of entire clusters over time and hence provides a completely new approach to interpreting results from cluster analysis. 

# The issue of label switching

The main issue with tracking time varying clusters over time is that one cannot simply re-estimate the clusters at each given point in time. Even running the same clustering algorithm multiple times on the same exact data can result in a similar partition, but with a different permutation of the cluster labels. Cluster labels, to that sense, are completely arbitrary. This problem is better known in the literature as 'label switching'. Imagine a mixture model of $x = (x_1, ..., x_n)$ independent observations, from a model with $k$ components such that:

\begin{equation}
p(x|\pi,\phi,\eta) = \pi_1f(x;\phi_1, \eta)+ ... + \pi_kf(x;\phi_k, \eta)
\end{equation}

Here $\pi = (\pi_1,...,\pi_k)$ are the mixture proportions, or the likelihood of an observation $x$ being in a certain cluster $k$, which is constrained to be non-negative and sum up to 1. $\eta$ is a parameter that is common to all components, and $\phi_k$ are the component specific parameters. Then for any permutation $\nu$ of the vector of cluster labels $1,...,k$ the corresponding permutation of the parameter vector $\theta : (\pi, \phi, \eta)$ can be written as follows:

\begin{equation}
\nu(\theta) = \nu(\pi, \phi, \eta) = ((\pi_{\nu(1)},...,\pi_{\nu(k)})(\phi_{\nu(1)},...,\phi_{\nu(k)}),\eta)
\end{equation}

The problem arises from the fact that for each permutation of $\theta$ the likelihood function of that permutation is the same:
\begin{equation}
L(\theta;x)=\prod\limits_{i=1}^n\{\pi_1f(x_i;\phi_1,\eta)+ ... +\pi_kf(x_i;\phi_k,\eta) \}
\end{equation}


Previous work on the label switching problem has mainly focused on trying to recover a link between different permutations of the labelling. One of the most common approaches is the use of a so called 'identifiability constraint' which can only be satisfied by a certain permutation of $\theta$. For instance $\mu_1<\mu_2<...<\mu_k$. Other, more sophisticated relabelling methods were also proposed. This type of approach poses many difficulties when estimating a mixture model for the same data set, but it poses an even greater problem when re-estimating the mixture model for different points in time. Because there is no mathematical link between the prior of the first estimation and the posterior of the subsequent estimations there is no guarantee this identifiability constraint holds over time. 

In the re-estimation approach proposed in this thesis we do not attempt to recover a link between different estimations by means of a relabelling model. Rather, we propose to use Bayesian updating to recover the parameters for each specification of the mixture model at different points in time. In order to do so we can represent the time varying cluster means and variances as a state-space model\footnotemark. Let $y_{it}$ be the $k \times 1$ vector of attributes for an individual $i$ at time $t$. $j \in \{1,...,J\}$ denotes the number of clusters. In our model we let the number of clusters be fixed over time.  $\mu_{j,t}$ denotes the $k \times 1$ vector of cluster centers for cluster $j$ at time $t$, and $\Sigma_{j,t}$ denotes the $k \times k$ diagonal covariance matrix. By stating the clusters have a diagonal covariance matrix we assume the clusters to be complete independent. As a matter of fact one of the most important reasons for using the EM algorithm to define thte CSR performance profiles was the ability to detect highly overlapping clusters. So this is one of the important limitations to the proposed estimation procedure. In reality, and especially for the data set used in this study, cluster variance parameters from the EM estimation are not completely independent. 

To aid the estimation we redefine the elements of the covariance matrix to be $h_{j,t,nn} = ln(\sigma^2_{j,t,nn})$, for $nn = 1,...,n$. The complete state space model can thus be represented as:

\begin{equation}
y_{i,t} = \left\{
	\begin{matrix}
	N(\mu_{1,t}, \Sigma_{1,t}) \mbox{ with probability } p_{i,1}\\
	\vdots \\
	N(\mu_{J,t}, \Sigma_{J,t}) \mbox{ with probability } p_{i,J}
	\end{matrix}
	\right.
\end{equation}

where the probabilities satisfy $p_{i,j}\in(0,1), \forall i,j$ and $\sum_{j=1}^J p_{i,j} = 1, \forall i$.

The observation equation can thus be defined as follows:

\begin{equation}
y_{i,t} = 
	\begin{pmatrix}
	z_{i,1} & z_{i,2} &...&z_{i,J} 
	\end{pmatrix}
	\begin{pmatrix}
	\mu_{1,t} \\ \mu_{2,t} \\...\\\mu_{J,t} 
	\end{pmatrix}\\
+
	\begin{pmatrix}
	\exp(h_{1,t}) &...& 0& 0\\
	0	& \exp(h_{2,t}) & 0& 0\\
	0 	& &\ddots &&\\
	&&&\exp(h_{J,t}) 
	\end{pmatrix}	
    \times \begin{pmatrix}
	\epsilon_{1,t}\\ \vdots \\ \epsilon_{J,t}
	\end{pmatrix}
\end{equation}


where $\epsilon_{1,t}\sim NID(0,1)$, and where $z_{i,1}\in \{0,1\}$ and all observations belong to one group i.e.\ $\sum_{j}z_{i,j} = 1$. The state equations of the model can be defined as:


\begin{equation}
	\begin{pmatrix}
	\mu_{1,t} \\ \mu_{2,t} \\ ... \\\mu_{J,t} 
	\end{pmatrix}
	= 	\begin{pmatrix}
	\mu_{1,t-1} \\ \mu_{2,t-1} \\ ... \\\mu_{J,t-1} 
	\end{pmatrix}
	+ 
	\begin{pmatrix}
	\eta_{\mu,1} \\ \eta_{\mu,2} \\ \vdots \\ \eta_{\mu,J}
	\end{pmatrix} \\
\end{equation}


\begin{equation}
  \begin{pmatrix}
	h_{1,t} \\ h_{2,t} \\ ... \\h_{J,t} 
	\end{pmatrix}
	= 	\begin{pmatrix}
	h_{1,t-1} \\ h_{2,t-1} \\ ... \\h_{J,t-1} 
	\end{pmatrix}
	+ 
	\begin{pmatrix}
	\eta_{h,1} \\ \eta_{h,2} \\ \vdots \\ \eta_{h,J}
	\end{pmatrix}
\end{equation}





## Model estimation procedure

The state-space model representation is estimated using a 2-step estimation procedure, which is based on a step involving the estimation of cluster membership by the EM-algorithm and a step involving the updating of the model parameter estimates using Maximum Likelihood Estimation. The pseudo code of the algorithm is given by:

---

* **Step 0:** Initialize all model parameters and memberships, using EM clustering, such that:
\[ p^*_{i,j,t} \in \{p^*_{i,j,t}| 0 \ge p^*_{i,j,t} \le 1\} \]
 In this step the initial parameter estimates and memberships are based on the performance clustering methodology using the EM algorithm. 
* **Step 1:** Time average the cluster membership likelihoods and round them to get the time invariant cluster membership 	$Z =\begin{pmatrix}	z_{i,1} & z_{i,2} & ... &z_{i,J} \end{pmatrix}$, 

    where:
     $z_{i,J} = ||\frac{1}{T}\sum p^*_{i,j,t}||$

    Where $z_{i,1}\in \{0,1\}$ 

    and  $\sum_{j}z_{i,j} = 1$. 

    Use **MLE** to obtain the observation and state variances $V = \left(\sigma^2_{mu,1},...,\sigma^2_{mu,J}, \sigma^2_{h,1},...,\sigma^2_{h,J}\right)$ and $h_{J,t}$ as represented in formula (5). Use the **Kalman Smoother** to retrieve the time varying cluster means $\mu_{J,T}$
 
* **Step 2:** Calculate the time invariant cluster means and variances as:
 \[\mu_{J} = \frac{1}{T}\sum \mu_{J,T} \text{ \:\: and  \:\: } h_{J} = \frac{1}{T}\sum h_{J,T}\]
 Initialize the EM clustering algorithm with the updated time invariant parameters and determine the updated cluster membership likelihoods $p^*_{i,j,t}$. 

* **End:** Repeat Step 1 and Step 2 until convergence. The vectorization of the parameters is defined as
$\theta = \left(\mbox{vec}(Z), \mbox{vec}(\mu), \mbox{vec}(h), V'\right)$ and the stopping criteria as (where $m$ denotes the iteration number):
\[ |\theta^{m} - \theta^{m-1}| \leq \epsilon\]
\end{itemize}
\rule{\linewidth}{1.5pt}

---

The EM algorithm is implemented using the Mclust package, where the updated parameter estimates are substituted in the model at each iteration. The Maximum Likelihood estimation and Kalman smoother are implemented using the popular R package 'DLM'. The use of the Kalman smoother, rather than the Kalman filter is motivated by the ability of the smoother to iterate over the different time periods and hence to retrieve the most appropriate initial parameter estimates. In this case the MLE does not provide these estimates. Note that in each iteration the MLE estimates for $2 \cdot n\cdot T \cdot J$ parameters need to be estimated, this means that as the amount of individuals and time periods in the sample goes up the computational complexity increases exponentially. 

The amount of iterations required until convergence depends on several factors. First of all, as the amount of individuals in the sample increases, so does the computational complexity, hence more iterations are required. But the number of required iterations also depends on the characteristics of the EM clustering. For completely separated clusters the algorithm will converge very fast because the updated cluster parameters won't affect the cluster results. However, in a setting with highly overlapping clusters, the updating of the cluster parameters will cause a significant shift in cluster membership, hence the algorithm will require more iterations until convergence. 

The proposed 2-step estimation procedure involves the use of two different models with differing underlying assumptions. The EM clustering algorithm assumes that the true DGP can be represented as a set of multivariate Gaussians. Furthermore, in the EM clustering procedure it is implicitly assumed that the amount of clusters is constant over time. Though this need not be the case in real life. Furthermore, the EM procedure assumes constant cluster parameters over time. The state-space model on the other hand assumes that the cluster parameters are time varying. And that the cluster parameters over time can be represented by an AR(1) process. In this instance with a unit root, as can be noted from equation 6 and 7. We thus assume that the time varying cluster parameters can be modeled by a random walk. Though these two models pose different core assumptions the combination of the two allows for the translation of the static clustering to the time domain.


## Code
In this repository you can find an example code containing a simulation dataset and the required code to run the algorithm mentioned above. In case of quesitons/comments please contact me.
