Choose one of the following four questions.  Of course, you can try more than one of them and you will certainly be a star among the candidates.  Here are the tasks for each project:

2 weeks after( By 25th October 2024) you receive this email, please email us to indicate if you are interested in attempting the exercise.  Send us your GitHub address where you will upload your report and your code.  
8 weeks after( by 8th Dec 2024)  you receive this email, please send us code base on part 1 as well as a brief report.  The report should include,
A literature review of potential methods for solving the problem
Propose your choice of methods and the reason for the choice
Present the answers of the question in clear and understandable language.
Testing plans and testing results for checking your implementation is correct and your results are valid. 
16 weeks( 3rd February 2025) after your receive this email, send us the final code and your final report for both part 1 and part 2.   The code has to be written in Python 3 and Tensorflow.  No other options allowed.
 

Question 1: Hamiltonian Neural Network and Hamiltonian Monte Carlo

Part 1:

a.       Consider the paper “Efficient Bayesian Inference with Latent Hamiltonian Neural Networks in No-U-Turn Sampling” Dhulipala(23) attached.

b.       Please write a paper explaining parts that are hard to understand in the original paper.  Please point out any typos and any details that are essential for the implementation but missing in the paper.

c.       Replicate the results with Tensorflow and Tensorflow Probability.  Can you get the same results as the Pytorch code of the authors? 

d.       is the TFP code https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/mcmc/hmc.py#L175-L540 on HMC useful for your implementation?

Part 2:

a.       Consider the paper “Pseudo-Marginal Hamiltonian Monte Carlo” Alenlov(21)

b.       Can we train a Hamiltonian Neural Network to learn the extended Hamiltonian as defined in equation 10 of Alenlov(21) by constructing a loss function based on the equations of motions as in equation (12) ?

c.       Is it possible to construct an algorithm based on this trained Hamiltonian Neural Network in part b above similar to that of Dhulipala(23) ?

d.       Is there any advantage of this Hamiltonian Neural Network approach versus the numerical integrator in equation 19 of Alenlov(21)?

e.       Can your Hamiltonian Neural Network augmented Pseudo-Marginal HMC algorithm replicate the results of the Generalized Linear Mixed Model of section 4.3 of Alenlov(21)?

Bonus part: Can you think of any financial model that is beneficial to use the algorithm of Dhulipala(23) or that of Alenlov(21) for estimation?  Why?

 

Question 2: Income statement and balance sheet forecast

Part 1)

1)      We would like to forecast the balance sheet of a company.  Unfortunately, the different fields of a balance sheet are not independent.  Hence we have to construct a model that respect these identities.  For a short introduction of the problem, please consider the papers Velez-Pareja(09) and Velez-Pareja(10).  For a much more detail exposition of the problem, please consult Shahnazarian(04) and the textbook “financial forecasting, analysis and modelling” by Samonas, as well as other standard accounting textbooks. 

2)      Construct a very simple model of the balance sheet based on the tools of Velez-Pareja(09) and Velez-Pareja(10).  Please write down the mathematical equations government the evolution of the fields of balance sheet.  Is it possible to model this problem as a time series?  How do we handle the accounting identities? 

3)      Implement the model in Tensorflow and python

4)      You can get income statement and balance sheet data from yahoo finance.  This blog post may help you.  https://rfachrizal.medium.com/how-to-obtain-financial-statements-from-stocks-using-yfinance-87c432b803b8

5)      Choose some companies to apply your model to.  How are you going to train your model?  How can you test if your model is good at forecasting the balance sheet of the company?  How can you ensure that your forecast at least respect the accounting identities, and at least satisfying the asset = liability + equity identity as other relationship stated in the papers quoted here?

6)      Can you use your model to forecast earnings?

7)      What are the ML techniques we can use to your model to make it better?

8)      Hint: simulation is highly related to prediction.  Suppose that you can simulate y(t+1) given y(t).  The prediction problem is very simple to implement numerically.  A general form of the model can be written as y(t+1) = f( x(t), y(t) ) + n(t),  where n(t) is some noise term to be specified, and x(t) are additional sets of variables that are relevant for the simulation.  What should x(t) be? 

 

Part 2) Consider the problem of paper Kim(24) applying large language model to financial statement analysis.

a)       Choose your favourite LLM to apply the problem of financial statement analysis.

b)      Let’s try the task of balance sheet forecast using the same set of data as collected in part 1, does the LLM you picked perform better or worse than your model?

c)       Is it possible to combine your model in part 1 and LLM to create an ensemble model that performs better than the individual model in balance sheet forecast?

d)      Given the results of your analysis, pick a company you have analysed, what would you recommend to the CFO or CEO of this particular company given your results? 

 

Question 3: Dynamic causal Bayesian optimization and optimal intervention.

Part 1

Consider the paper Dynamic Causal Bayesian Optimization https://arxiv.org/abs/2110.13891 by Agiletti et al.

 

1.       In the paper, the authors considered very small and simple graphs, which might not be the case in practice. Can you give an example of a causal graph with 15 nodes at each time step -- 7 non-manipulable, 7 manipulable, and 1 target variable, how would you get the exploration set (a key input to the algorithm)? Would you write a program for this purpose?  Is it enough to have simply the causal diagram to get the exploration set?  What additional specifications do you need ?

2.       Replicate their synthetic experiment results in Tensorflow probability.  https://github.com/neildhir/DCBO. Hint: Check how they used the GPy package and think about how we can use TFP to replicate it.

3.       Can you write a document discussing the following:

a.       What are the key ideas of this paper? What are the ideas that excite you the most? Why do you find them interesting and critical?

b.       Do you think the acquisition function is correct?  Is there any typo? If yes, can you explain?  If not, can you derive the correct one?

c.       Do you find any errors or questionable issues? 

d.       Are there any parts that are not clear in the paper?

e.       Can you please include derivations that are not clear in their paper?

f.        Is there any important part of their code that is not mentioned in the paper?

g.       Can you explain the paper based on details you have learnt from their code and your result replication experience?

h.       Can you think of a possible application of such techniques for a bank? 

i.         Hint:

                                                                                       i.      Definition 2.1 in the CBO paper might be helpful for understanding Assumption 1 and Definition 4 in this paper.

                                                                                     ii.      There is some abuse of notation in this paper. Don’t get confused by the M_t in Definition 2 and bold M_t in in Section 3.2.

                                                                                   iii.      This paper assumes the SEM to be known, which it means both the causal diagram and the distributions are known.

 

Part 2:

a.       Consider this paper:  https://arxiv.org/abs/2406.10917 Bayesian Intervention Optimization for Causal Discovery. 

b.       Suppose you were a serious reviewer, and a chair professor of a famous university, of this paper of a major ML conference.   Do you find any error?  Does the paper have any theoretical contribution?  If you think they do, are the theoretical results proved?

c.       Will you accept or reject the paper?  Why?  Can you please write a detail review of the paper, point out the pros and cons of the paper, including any possible errors? 

d.       Can you replicate their synthetic data generation results in Tensorflow probability?    There is no need to replicate the actual experiment. 

e.       Is it possible to expand the paper to multiple variables?  How would you advise the authors to do that?    For this multivariable case, is it possible to apply Causal Bayesian Optimization to this Bayesian intervention optimization problem? 

f.        If you believe this paper is correct, can you think of a possible application of this paper for a bank? 

 

 

Question 4: Diffusion Policy Policy Optimization

 

Part 1) Consider this paper https://arxiv.org/abs/2409.00588

a)       Can you replicate their results with Tensorflow Agent?

b)      From your experience of replication of their code, what are the parts that are unclear in the paper?  Can you please write a note explaining in more details the algorithm and the parts that are missing in the paper? 

c)       Have you found any mistakes or errors? 

d)      If you were the reviewer of this paper, would you accept or reject this paper for a major conference?  Could you please write a note commenting the pros and cons of this paper?

 

Part 2)

a)       Consider the Diffusion Policy MDP of DPPO paper, is it possible to construct a soft actor critic (SAC) type algorithm on this special MDP https://arxiv.org/abs/1812.05905

b)      One problem in the construction of the soft actor critic algorithm is the estimation of the entropy of the diffusion policy.   Is it possible to employ some kind GMM estimation like this paper https://arxiv.org/abs/2405.15177 . 

c)       If you think that we cannot construct an SAC algorithm on the diffusion policy MDP, do you think this paper https://arxiv.org/pdf/2405.15177 provides a way to construct such kind of algorithm?  This paper claims that they are practically the same as the SAC with a diffusion policy.  However, a few details are missing.  Could you please more detail description of the algorithm?

d)      Are you able to implement the algorithm or a modified version of the algorithm?  Can you please provide a note describing either your algorithm in a) and b) or c) or your own version of diffusion policy SAC? 
