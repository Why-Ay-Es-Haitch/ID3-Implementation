import numpy as np
import pandas as pd # These are all the import statements required for our algorithm

df = pd.read_csv("car.csv", header=0, skipinitialspace=True) #This line imports our dataset.
dfValid = df[1500:] #creation of training dataset
df = df.head(1500) #creation of testing dataset

class TreeID3: #This is our decision tree class which can begin making our tree.
    def __init__(self):
        self.root = None #Sets it so that there is no root node until the fit function is used
    def fit(self, samples, answers):
        self.root = TreeNode(samples, answers) # Creates a root node.
        self.root.make() #Begins the tree building process
    def predict(self, input): #predicts class of given input
        return self.root.predict(input) #Starts the prediction by telling root node to predict

class TreeNode:
    def __init__(self, samples, answers):
        #Sets the inital variables for each tree node.
        self.samples = samples #Sets which samples are available for the tree node
        self.answers = answers #Sets the target column from the sample dataset
        self.decision = None #Initializies the tree decision
        self.children = {} #Initializies the children of the tree as a dictionary which can be appended to
        self.splitby = None #Initializies the splitby attribute to none as the tree has not been made yet
        self.infomax = 0 #Initializies the maximum infomax
        self.average = None
    
    def make(self):
        if len(self.samples)==0:
            self.decision = "None"
            return #Returns none of there are no more samples left
        elif (len(self.samples[self.answers].unique())==1):
            self.decision = self.samples[self.answers].unique()
            return #Returns the value of the answer column if there is only 1 unique value left.
        else:
            for i in self.samples.keys(): #Loops through all attributes in the sample given
                if i==self.answers: #Makes sure we do not use the target column to split the tree
                    continue
                else:
                    info = compute_info(self.samples, i, self.answers) #computes the information gain for a attribute in the sample given
                    if (info > self.infomax): #Checks if the highest information gain is achieved by splitting by this attribute
                        self.infomax = info #sets new highest information gain
                        self.splitby = i #sets the attribute to splitby
        self.average = self.samples[self.answers].value_counts(normalize=True, sort=True).index.values[0]
        for j in self.samples[self.splitby].unique(): #Iterates through all the unique values in attribute we are spliting by
            index = self.samples[self.splitby] == j #Splits the dataset by the unique values in the split by attribute
            self.children[j] = TreeNode(self.samples[index], self.answers) #Creates a new child node and passes on the newly split dataset to it
            self.children[j].make() #Starts the tree node
        
    def pretty_print(self, prefix=''): #Prints the tree in a pretty way
        if self.splitby is not None:
            for k, v in self.children.items():
                v.pretty_print(f"{prefix}:When {self.splitby} is {k}")
                #v.pretty_print(f"{prefix}:{k}:")
        else:
            print(f"{prefix}:{self.decision}")

    
    def predict(self, sample_input): #predict function for each node
        if self.decision is not None:
            return (self.decision[0])
        else:
            try:
                attrValue = sample_input[self.splitby] #splits data by node attribute
                child = self.children[attrValue[0]] #selects child node which accepts the value of the attribute 
                return child.predict(sample_input) #Passes data to said child node
            except (KeyError, IndexError) as e:
                return self.average
            

def compute_entropy(y): #Function to calculate entropy
    if len(y) < 2:
        return 0 #Checks if there there are not enough values to calculate entropy
    array = y.value_counts(normalize=True) #Determines frequency of unique values in the answer column
    return -(array*np.log2(array + 1e-6)).sum() #Calculates entropy of the sample via entropy formula the +1e-6 ensures that 0 cannot be passed into the log function

def compute_info(samples, i, target): #Function to calculate information gain
    values = samples[i].value_counts(normalize=True) #Determines frequency of unique values in a given attribute
    split_ent = 0 
    for v, fr in values.iteritems(): #Iterates through unique values and their frequencies
        sub_ent = compute_entropy(samples[samples[i]==v][target]) #Calculates entropy by unique value in the given attribute
        split_ent += fr * sub_ent #Calculates entropy by unique value multiplied by frequency of given value
    ent = compute_entropy(samples[target]) #Calculates entropy of dataset
    return ent - split_ent #returns information gain

def predict_validation(data, tree): #Function
    results_df = pd.DataFrame(data=[''], columns=['TargetValue']) #Creates a dataset to write to
    for i in range(len(data)): #Loops through all rows in dataset
        data_input = data[i:i+1] #selects row
        data_input = data_input.reset_index(drop=True) #Drops index of row for compatability
        result = tree.predict(data_input) #Predicts row
        results_df = results_df.append(pd.DataFrame(data=[result], columns=['TargetValue'])) #Saves prediction to dataset
    return results_df #Returns predictions

if __name__=="__main__":
    t = TreeID3() #Creates the tree
    t.fit(df, "Acceptability") #Passes on the data to tree and starts the tree making process
    t.root.pretty_print() #Prints the tree
    validation_ans = (predict_validation(dfValid, t))
    print(dfValid["Acceptability"])
    dfValid.to_csv("dfValid.csv")
    print(validation_ans)
    validation_ans.to_csv("validation_ans.csv")