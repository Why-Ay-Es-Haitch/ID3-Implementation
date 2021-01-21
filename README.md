# ID3-Implementation
A simple implementation of the ID3 decision tree algorithm from scratch.
## ID3
The ID3 Algorithm is used to generate automatically generate a decision tree from a dataset using entropy.
On each iteration of the algorithm, it iterates through every unused attribute of the set and calculates the entropy or the information gain of that attribute. It then selects the attribute which has the smallest entropy (or largest information gain) value.
## Constraints
This is a basic implementation with no tree pruning. As entropy is calculated only at each node recursively, it is a greedy algorithm.
