import pandas as pd
from decisionTree import decisionTree

training = pd.read_csv("training.csv",index_col=0)
# attributes=training.iloc[:,4:].columns #Done in the tree
# print(attributes)
# print(type(attributes))
tree=decisionTree(5)
tree.fit(training)


testing=pd.read_csv("testing.csv",index_col=0)
preds=[]#Correct or incorrect
pstar=0.5
for index, row in testing.iterrows():
    p=tree.predict(row)
    if(p<pstar):
        p=0
    else:
        p=1
    if(row["Male"]==p):
        preds.append(1)
    else:
        preds.append(0)
count=0
for pred in preds:
    count+=pred


print("Accuracy: "+str(count/len(preds)))
print("Most Important Column: "+tree.root.decidingColumn)