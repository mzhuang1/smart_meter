import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA,KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn import svm
import brewer2mpl
from pandas.tools.plotting import parallel_coordinates
from matplotlib import rcParams
from matplotlib.pyplot import savefig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as QDA


##option
##configure brewer2mpl

dark2_cmap = brewer2mpl.get_map('Dark2', 'Qualitative', 7)
dark2_colors = dark2_cmap.mpl_colors

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
#rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

#import scipy.stats as stats
#load excel data
xls = 'trainingData.xlsx'
#xls_test = 'Heat Scan data_IIT_30415.xlsx'
names = ['models','locations',
         'tprt_1','tprt_2','tprt_3','tprt_4','tprt_5','tprt_6',
         'load_1','load_2','load_3','load_4','load_5','load_6',
         'months','hours','label']
tprtlist = ['tprt_1','tprt_2','tprt_3','tprt_4','tprt_5','tprt_6']
loadlist = ['load_1','load_2','load_3','load_4','load_5','load_6'] 
data_0=pd.io.excel.read_excel(xls,sheet_name='Sheet1', names=names).dropna()  

#data cleaning

model = set(data_0.models)
model = sorted(model)

for m in model:
    data_0[m] = [m is meter for meter in data_0.models]
data_0[model].sum()

location = set(data_0.locations)
location = sorted(location)
for m in location:
    data_0[m] = [m is meter for meter in data_0.locations]
data_0[location].sum()
 
month = set(data_0.months)
month = sorted(month)
print(month)
for m in month:
    data_0[m] = [m is meter for meter in data_0.months]
data_0[month].sum()



feature_index = tprtlist + loadlist + model + location + month
X = data_0[feature_index].values
y = data_0['label'].values

list_0 = ['tprt_1','tprt_2','tprt_3','tprt_4','tprt_5','tprt_6','label']
list_1 = ['load_1','load_2','load_3','load_4','load_5','load_6','label']
mkeys=[1,2,3]
mvals=model
rmap={e[0]:e[1] for e in zip(mkeys,mvals)}
#d2=data_0.groupby(['models','locations'])
d2=data_0.groupby('May')
#fig, axes=plt.subplots(figsize=(20,20), nrows=len(location)*len(model), ncols=1)
colors=[dark2_cmap.mpl_colormap(col) for col in np.linspace(1,0,2)]
#
#for m,subset in d2:
#    a='###'
#    print(m)
#    parallel_coordinates(subset[list_0],'label',colors=colors, alpha=0.12)
#    plt.show()
#    parallel_coordinates(subset[list_1],'label',colors=colors, alpha=0.12)
#    plt.show()

#smaller_frame=data_0[loadlist]#tprtlist+loadlist
#axeslist=scatter_matrix(smaller_frame, alpha=0.8, figsize=(12,12), diagonal="kde")
#for ax in axeslist.flatten():
#    ax.grid(False)
    
#PCA
pca = PCA(n_components=6)
X_E = pca.fit_transform(X)
#print(pca.explained_variance_ratio_)
#plt.scatter(X_E[:, 0], X_E[:, 1])
def do_pca(d,n):
    pca = PCA(n_components=n)
    X = pca.fit_transform(d)
    print(pca.explained_variance_ratio_)
    return X, pca
X2, pca2=do_pca(X,2)
df = pd.DataFrame({"x":X2[:,0], "y":X2[:,1],"label":np.where(y==1,
                   "bad","good")})
colors = ["red", "yellow"]
#for label, color in zip(df['label'].unique(), colors):
#    mask = df['label']==label
#    plt.scatter(df[mask]['x'], df[mask]['y'],c=color,label=label,alpha=0.6)
#plt.legend()


##kernel PCA
#def do_kpca(d,n):
#    kpca = KernelPCA(n_components=2,kernel='rbf',fit_inverse_transform=True,gamma=0.1)
#    X_kpca=kpca.fit_transform(d)
#    return X_kpca,kpca
#X2,kpca2=do_kpca(X,2)
#df = pd.DataFrame({"x":X2[:,0], "y":X2[:,1],"label":np.where(y==1,
#                   "bad","good")})   
#colors = ["red", "blue"]
#for label, color in zip(df['label'].unique(), colors):
#    mask = df['label']==label
#    plt.scatter(df[mask]['x'], df[mask]['y'],c=color,label=label,alpha=0.8)
#plt.legend()

#LDA vs QDA





X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_E,y)
X_train, X_test, y_train, y_test = train_test_split(X, y)

gbdt = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1,
                                  max_depth=5, random_state=1234).fit(X_train,y_train)
gbdt.score(X_test, y_test)
pd.crosstab(y_test,gbdt.predict(X_test), rownames=['Actual'], colnames=['Predicted'])
indices = np.argsort(gbdt.feature_importances_)
plt.barh(np.arange(len(feature_index)),gbdt.feature_importances_[indices])
plt.yticks(np.arange(len(feature_index))+0.5, np.array(feature_index)[indices])
_=plt.xlabel('Relative importance')
np.array(feature_index)[indices][::-1]
plt.show()