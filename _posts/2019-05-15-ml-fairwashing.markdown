---
layout: post
title:  Fairwashing in machine learning
date: 2019-04-24 15:51:00
description: Summary of a paper, Fairwashing -- the risk of rationalization
---
Machine learning is now used in every aspect of our life, from entertainment to high stakes decision-making processes such as credit scoring, medical diagnosis, or predictive justice. The potential risk of incorrect decisions has raised the public demand for an explanation of the decisions of machine learning models. In addition to this public demand, all across the world, several communities and government initiatives are emerging, asking for more transparency in machine learning models' decisions, and the development of an ethically-aligned AI. As an example, in Europe, the new General Data Protection Regulation has a provision requiring explanations for the decisions of machine learning models that have a significant impact on individuals [\[Goodman and Flaxman, 2017\]](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2741){:target="\_blank"}. 

We believe that because of this particular combination of regulations and public demand for ethically-aligned development of AI, a dishonest machine learning models' producers may be tempted to perform fairwashing. We define fairwashing as promoting the false perception that a machine learning model complies with a given ethical requirement while it might not be so. The risk of fairwashing is all the more possible because the right to explanation as defined in current regulations does not give precise directives on what it means to provide a ''valid explanation'' [\[Wachter et al., 2017\]](https://academic.oup.com/idpl/article-pdf/doi/10.1093/idpl/ipx005/17932196/ipx005.pdf){:target="\_blank"} [\[Edwards and Veale, 2017\]](https://strathprints.strath.ac.uk/61618/8/Edwards_Veale_DLTR_2017_Slave_to_the_algorithm_why_a_right_to_an_explanation_is_probably.pdf){:target="\_blank"}, leaving a legal loophole that can be exploited by a dishonest model's producer to cover up a possible misconduct of its black-box model by providing misleading explanations.

To demonstrate this risk, we consider two variations of the black-box explanation problem and show that one can forge misleading explanations that comply with a given ethical requirement. In particular, we use fairness as the ethical requirement and demonstrate that given a black-box model that is unfair, one can systematically deduce rule lists with high fidelity to the black-box model while being considerably less unfair at the same time.


### **Prerequisites**
<hr>

#### **Fairness**
There are several definitions for fairness in machine learning [\[Verma and Rubin, 2018\]](http://fairware.cs.umass.edu/papers/Verma.pdf){:target="\_blank"}. Central to all these definitions is the notion of the sensitive attribute $$s$$ for which non-discrimination should be established. An example of such attribute can be the gender, the ethnicity, the religion... For this work, we use demographic parity [\[Calders et al. 2009\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5360534){:target="\_blank"} as fairness metric. Demographic parity requires the prediction $$\hat{y}$$ of the black-box model to be independent of the sensitive attribute $$s$$, that is, $$P(\hat{y}=1 | s= 1) = P(\hat{y}=1 | s= 0)$$. We therefore define unfairness as $$\mathsf{unfairness}=|P(\hat{y}=1 | s= 1) - P(\hat{y}=1 | s= 0)|$$.

#### **Black-box explanation**
Black-box explanation is the problem of explaining how a machine learning model -- whose internal logic is hidden to the auditor and generally complex -- produces its outcomes. An explanation can be viewed as an interface between humans and a decision process that is both an accurate proxy of the decision process and understandable by humans [\[Guidotti et al. 2018\]](https://dl.acm.org/citation.cfm?id=3236009){:target="\_blank"}. Figure 1 shows an example of a black-box explanation pipeline. First, a black-box model $$M_b$$ is obtained using a particular learning algorithm (e.g., random forest, SVM, neural network...) on a particular dataset $$D_{train}$$. Then, the black-box model is used to label another dataset $$D_{labeled} \neq D_{train}$$. Finally, an explanation algorithm takes as input $$D_{labeled}$$ and produces an interpretable model $$M_i$$. Examples of interfaces accepted in the literature as interpretable models include linear models, decision trees, rule lists, and rule sets. In this work, we use rule lists as explanation models. We consider two variations of the black-box explanation problem, namely model explanation and outcome explanation. The former consists in providing an explanation model that explains all the decisions of the black-box model, while the latter consists of producing an explanation model that explains a particular decision of the black-box model. To assess the performance of an explanation algorithm, we rely on the notion of fidelity [\[Craven and Shavlik, 1996\]](http://papers.nips.cc/paper/1152-extracting-tree-structured-representations-of-trained-networks.pdf){:target="\_blank"}, which is the accuracy of the explanation model $$M_i$$ relative to the black-box model $$M_b$$ on some instances $$X$$. Simply put, $$\mathsf{fidelity} = \frac{1}{|X|} \sum_{x \in X} \mathbb{I}(M_i(x)=M_b(x))$$.
<div >
    <img align="middle"  src="{{ site.baseurl }}/assets/img/blog/fairwashing/explanation.png" height="400" width="700">
</div>
<div >
    Figure 1: Black-box explanation.
</div>

#### **Rule lists**
A rule list $$rl= (d_p, \delta_p, q_0, K)$$ of length $$K \geq 0$$ is a $$(K+1)-$$tuple consisting of $$K$$ distinct association rules $$r_k = p_k \to q_k$$, in which $$p_k \in d_p$$ is the antecedent of the association rule and $$q_k \in \delta_p$$ its corresponding consequent, followed by a default prediction $$q_0$$. The rule list below predicts whether a person is likely to make at least 50k per year. 

    if (capital gain > 7056) then predict (income >= 50k)
    else if (education:HS-grad) then predict (income < 50k)
    else if (occupation:other) then predict (income < 50k)
    else if (occupation:white-collar) then predict (income >= 50k)
    else predict (income < 50k)

To make a prediction using a rule list, the rules are applied sequentially until one rule applies, in which case the associated outcome is reported. If none of the rules applies, then the default prediction is reported.

#### **Computing optimal Rule lists with CORELS**
CORELS [\[Angelino et al. 2018\]](https://arxiv.org/pdf/1704.01701.pdf){:target="\_blank"} is a supervised machine learning algorithm which takes as input a training set with $$n$$ predictors, all assumed to be binary.
First, it represents the search space of the rule lists as a $$n$$-level trie whose root node has $$n$$ children, formed by the $$n$$ predictors, each of which has $$n-1$$ children, composed of all the predictors except the parents, and so on. 
Afterwards, it considers for a rule list $$rl=(d_p, \delta_p, q_0, K)$$, an objective function defined as a regularized empirical risk: $$\mathsf{misc(\cdot)} + \lambda K$$, where $$\mathsf{misc(\cdot)}$$ is the misclassification error of the rule list and $$\lambda \geq 0$$ is a regularization parameter used to penalize longer rule lists. Finally, it finds the optimal rule list by minimizing the so defined objective function. In particular, it uses an efficient branch-and-bound algorithm to prune the search space. 

#### **Enumeration of rule lists**
In this work, we use a rule list enumeration technique introduced in [\[Hara and Ishihata, 2018\]](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16343){:target="\_blank"}.
This algorithm takes as input the set $$T$$ of all binary predictors, and enumerate rule list models in the descending order of their objective function. It maintains a heap $$\mathcal{H}$$, whose priority is the objective function value of the rule lists and a list $$\mathcal{M}$$ for the enumerated models. First, it starts by computing the optimal rule $$m=\mathsf{CORELS}(T) = (d_p, \delta_p, q_0, K)$$ for $$T$$ using the CORELS algorithm. Then, it inserts the tuple $$(m, T, \emptyset)$$ into the heap. 
Finally, the algorithm repeats the following three steps until the stopping criterion is met:
* Extract the tuple $$(m, S, F)$$ with the maximum priority value from $$\mathcal{H}$$.
* Output $$m$$ as the $$i-$$th model if $$m \notin \mathcal{M}$$. 
* Branch the search space: compute and insert $$m' =\mathsf{CORELS}(S \setminus \{t_j\})$$ into $$\mathcal{H}$$ for all $$t_j \in \delta_p$$. 


### **Rationalization**
<hr>

#### **Problem formulation**
We define rationalization as the problem of finding an interpretable surrogate model $$M_i$$ approximating a black-box model $$M_b$$, such that $$M_i$$ is fairer than $$M_b$$. In particular, we distinguish model rationalization and outcome rationalization. 

Given a black-box model $$M_b$$, a set of instances $$X$$ and a sensitive attribute $$s$$, the model rationalization problem consists in finding an intrepretable global model $$M_i^g$$ such that $$\epsilon(M_i^g, X, s) > \epsilon(M_b, X, s)$$, for some fairness evaluation metric $$\epsilon(\cdot, \cdot, \cdot)$$. The outcome rationalization problem is similar to the model rationalaization problem, but consider only a single data point, and provide an explanation for this particular data point and its neighborhood. More precisely, given a black-box model $$M_b$$, an instance $$x$$, a neighborhood $$\mathcal{V}(x)$$ of $$x$$,  and a sensitive attribute $$s$$, the outcome rationalization problem consists in finding an intrepretable local model $$M_i^l$$ such that $$\epsilon(M_i^l, \mathcal{V}(x), s) > \epsilon(M_b, \mathcal{V}(x), s)$$, for some fairness evaluation metric $$\epsilon(\cdot, \cdot, \cdot)$$.

#### **LaundryML**
We propose LaundryML, an algorithm to solve the rationalization problem efficiently. First, LaundryML uses a modified version of CORELS in which we define the new objective function as $$\mathsf{(1-\beta) misc(\cdot)} + \mathsf{\beta unfairness(\cdot)}+ \lambda K$$, where $$\mathsf{misc(\cdot)}$$ is the misclassification error of the rule list, $$\mathsf{unfairness(\cdot)}$$ returns the unfairness of the rule list, $$\lambda \geq 0$$ is a regularization parameter used to penalize longer rule lists, and $$\beta \geq 0$$ is the weight of the unfairness regularization. Then, it uses the customized CORELS algorithm to enumerated rule list models. Finally, it uses a threshold for the unfairness to select the models that can be used for rationalization. 

To solve the model rationalization problem for a set $$X$$ of instances, LaundryML will take as input the tuple $$T=\{X, y\}$$ formed the instances $$X$$ and the predictions $$y$$ of the black-box model on $$X$$. To solve the outcome rationalization for a particular instance $$x$$, the inputs of LaundryML will be the instances $$T_x =  \mathsf{neigh(x, T)}$$, some neighborhood searching algorithm $$\mathsf{neigh(\cdot)}$$.


### **Experimentations**
<hr>
We train a Random Forest model on Adult Income dataset to mimic the black-box model. Our black-box model achieves respectively an accuracy of $$84.31\%$$, a precision of $$79.78\%$$, and an unfairness of $$0.13$$ on a set of instances we use to mimic a suing group (that accuses the black-box model on being unfair). 

#### **Results for model rationalization**

We perform fairwashaing through model rationalization by explaining the decisions of the black-box model with an interpretable model (here a rule list) that is less unfair than the black-box while having a fidelity greater than $$90\%$$. For the model rationalization experiment, we set $$\lambda=0.005$$, and $$\beta =\{0, 0.1, 0.2, 0.5, 0.7, 0.9\}$$, and we enumerate 50 models. 

<div >
    <img align="middle"  src="{{ site.baseurl }}/assets/img/blog/fairwashing/global_adult_unfairness.png" height="300" width="350">
    <img align="right"  src="{{ site.baseurl }}/assets/img/blog/fairwashing/global_adult_fidelity.png" height="300" width="350">

</div>
<div >
    Figure 2: CDFs of the unfairness (left) and the fidelity (right) of rationalized explanation models produced by LaundryML on the suing group of Adult Income. Results are for the demographic parity metric and the random forest black-box model. The vertical line on the left figure represents the unfairness of the black-box model. The CDFs on the right figure are the CDFs of the fidelity of explanation models whose unfairness are less than that of the black box model.
</div>

The results in Figure 2 show that one can obtain fairer explanation models that have good fidelity. In particular, we observe that as $$\beta$$ increases, both the unfairness and the fidelity of the enumerated models decrease. Overall, the best model we obtain has a fidelity of $$0.908$$ and an unfairness of $$0.058$$. In Figure 3, we compare the selected rationalization model to the black-box model in term of relative feature dependence ranking. The observations show that the sensitive attribute is ranked $$2$$nd (respectively $$28$$th) with the black-box model (respectively the rationalization model).

<div >
    <img align="middle"  src="{{ site.baseurl }}/assets/img/blog/fairwashing/audit.png" height="300" width="700">

</div>
<div >
    Figure 3: Relative feature dependence ranking obtained using FairML to audit models trained on the Adult Income dataset. 
    Green indicates that the feature highly contributes to a high salary rating on Adult. 
    Features that characterize the majority groups are highlighted in yellow. 
    Black-box model (left) vs. LaundryML model (middle) and its description (right).
</div>

#### **Results for outcome rationalization**
We perform fairwhasing trough outcome rationalization by explaining each rejected female subject with a rule list whose unfairness (as measured on the neighbourhood of the subject) is lower than that of the black-box model. For the outcome rationalization experiment, we set $$\lambda=0.005$$, and $$\beta =\{0.1, 0.3, 0.5, 0.7, 0.9\}$$, and for each rejected female subjects, we enumerate 50 models to perform outcome rationalization. 

<div >
    <img align="middle"  src="{{ site.baseurl }}/assets/img/blog/fairwashing/local_adult.png" height="300" width="500">

</div>
<div >
    Figure 4: CDFs of the unfairness of the best model found by LaundryML per user on Adult Income.
</div>

Results in Figure 4 show that as the fairness regulation parameter $$\beta$$ increases, the unfairness of the explanation model decreases. 
In particular, with $$\beta=0.9$$, a completely fair model (unfairness = $$0.0$$) was found to explain the outcome for each rejected female subject in Adult Income.

Results on ProPublica Recidivism dataset, as well as generalization of LaundryML to other black-box models and to other fairness metrics, are discussed in the [paper](https://arxiv.org/abs/1901.09749){:target="\_blank"}.


### **Conclusion**
<hr>

This work introduces the risk of fairwashing associated with black-box explanation in machine learning. We hope our work will raise the awareness of the machine learning community and inspire future research towards the ethical issues raised by the possibility of rationalizing.