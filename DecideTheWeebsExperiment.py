import pandas as pd
from decisionTree import decisionTree

training = pd.read_csv("training.csv",index_col=0)
validation=pd.read_csv("validation.csv",index_col=0)
# attributes=training.iloc[:,4:].columns #Done in the tree
# print(attributes)
# print(type(attributes))
for d in [2,3,5,7,10,20]:
    for q in [[0.5], [0.25,0.5,0.75], [0.17, .33, 0.5, .67, .83], [x/10 for x in range(10)]]:
        tree=decisionTree(d)
        tree.quantiles=q
        tree.fit(training)

        preds=[]#Correct or incorrect
        pstar=0.5
        for index, row in validation.iterrows():
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
        f = open("ExperimentParameterTuning", "a")
        s=""
        s += "depth: "+str(d)+"\n"
        s += "quantile set: "+str(q) + "\n"
        s += "Accuracy: "+str(count/len(preds)) + "\n"
        s += "Most Important Column: "+tree.root.decidingColumn + "\n"
        print(s)
        f.write(s+"\n\n")
        f.close()