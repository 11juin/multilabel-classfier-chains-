import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
df1=pd.read_csv("data.csv")
drugmatching= ['drug{}'.format(i) for i in range(0, df1.shape[0])]
se = pd.Series(drugmatching)
df1['matching'] =se
cols = df1.columns.tolist()
df1 = df1[[cols[-1]] + cols[:-1]] 
#Data visualisation
import matplotlib.pyplot as plt
N = 12
positives =  (df1[df1 == 1.0].count())[2:14]
negatievs =  (df1[df1 == 0.0].count())[2:14]
nan=df1.isnull().sum(axis=0)[2:14]
ind = np.arange(N)  
width = 0.35      
p1 = plt.bar(ind, positives.T, width, color='#d62728')
p2 = plt.bar(ind, negatievs.T, width,
             bottom=positives)
p3 = plt.bar(ind, nan.T, width,
             bottom=negatievs)
plt.ylabel('Targets')
plt.ylabel('Numbers')
plt.title('Number of each class in every label')
plt.xticks(ind, ('1', '2', '3', '4', '5','6','7','8','9','10','11','12'))
plt.legend((p1[0], p2[0],p3[0]), ('positives', 'negatives','nans'))
plt.show()
#draw modulars 
ms = [Chem.MolFromSmiles(x) for x in df1['smiles'][0:5]]
Draw.MolsToGridImage(ms,subImgSize=(300,300))
#Check corelations 
from scipy import stats
label=df1.iloc[:,2:df1.shape[1]].values
cor=stats.spearmanr(label)[0]
p=stats.spearmanr(label)[1]
#Find features (Morgan Fingerprints)
def smiles_to_fps(smiles_tuple, fp_radius, fp_length):
	fps = []
	for smiles in smiles_tuple:
		molecule = Chem.MolFromSmiles(smiles)
		fp = AllChem.GetMorganFingerprintAsBitVect(
			molecule, fp_radius, nBits=fp_length)
		fps.append(fp.ToBitString())
	fps = np.array(fps)
	fps = np.array([list(fp) for fp in fps], dtype=np.float32)
	return fps
sl1=tuple(df1['smiles'])
fp1=smiles_to_fps(sl1,fp_radius=4, fp_length=128)
df2=pd.concat([df1.iloc[:,0:2],pd.DataFrame(fp1)],axis=1)
labelset=df1.iloc[:,2:df1.shape[1]].values
labelset[np.isnan(labelset)] = 0
#split data 
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(df2,labelset, test_size=0.2,random_state=42)
X_train1=X_train.drop(['matching','smiles'], 1).values
X_test1=X_test.drop(['matching','smiles'], 1).values
#training 
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
classifier = ClassifierChain(SVC ( kernel='linear',probability=True,random_state=1234))  
classifier.fit(X_train1, Y_train)
predictions = classifier.predict(X_test1)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,predictions)
#testing
prob = classifier.predict_proba(X_test1)
from sklearn.metrics import roc_curve, roc_auc_score
auc=[]
for  i in range(0,12):    
    y_pred = prob[:,i]
    yi=Y_test[:,i]
    fpr, tpr, _ = roc_curve(yi, y_pred)
    auc.append(roc_auc_score(yi, y_pred, average='macro'))
    plt.plot(fpr, tpr, color='darkorange', lw=0.5)
#significant features 
indexs = [np.argmax(np.abs(classifier.estimators_[i].coef_))
for i in range(12)]



























