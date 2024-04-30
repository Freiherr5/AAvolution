# AAvolution

---

### Description
AAvolution is an algorithm performing in silico mutagenesis of amino acid sequences.
The algorithm utilizes data-based amino acid propensities, sequence length features and mechanisms 
inspired by genetics for an efficient sequence optimization.
___

### Elements of AAvolution

#### 1) Initialization
Random generation of a starting population in dependency defined of amino acid propensities and length 
distribution of the domains.

#### 2) Mutagenesis
3 main mutation types drive the diversification of the sequence population:
point-mutations, indels and crossovers

#### 3) Scoring
Translation of sequences via CPP [Breimann S., 24c], prediction and scoring via classification
ensemble (Random Forest, Naive Bayes, Gaussian Process; meta classifier: Logistic Regression).

#### 4) Selection
Selection after each Scoring-iteration and on other features (allowed length-range of sequence domain)

#### 5) Repopulation
After each Selection-step, the survivors of the initial population are used to fill up the population to its
original size. This is achieved by recombining the existing domain-parts of the survivors.

---

### Output

Top 100 in silico generated substrates + plots for visualization (average/max score of the population, t-SNE)



