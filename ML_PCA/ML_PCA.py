import numpy as np
import pandas as pd
import sklearn as skl
from sklearn.decomposition import PCA
from matplotlib import pyplot

FileName = 'File.csv'

#----------------------------------------------------------------------------------------------#
# Data outside #
Files = pd.read_csv(FileName, header=None)
SampleNumber = Files.columns.size - 1   # Sample Number
FeatureNumber = len(Files) - 1                # Feature Number
Labels = np.array(Files.values[0, 1:SampleNumber+1], dtype='int')        # Labels
Features = np.array(Files.values[1:FeatureNumber+1, 0], dtype='int')   # Features
Datas = Files.iloc[1:FeatureNumber+1, 1:SampleNumber+1]                        # Datas
Datas = np.array(Datas.T)  # sample * feature
print('Sample Number : ', SampleNumber)
print('Feature Number : ', FeatureNumber)
#----------------------------------------------------------------------------------------------#
# PCA Model #
PCAmodel = PCA()
PCAmodel.n_components = 2   # Default : None(all) , components number
PCAmodel.copy = False            # Default : True, copy the origin data
PCAmodel.whiten = False         # Default : False, let features with same var
PCAmodel.random_state = 0    # Default : None
PCAmodel.svd_solver = 'auto' # Default : auto
#----------------------------------------------------------------------------------------------#
# PCA Processing #
Scaler = skl.preprocessing.StandardScaler()
Scaler.mean_ = np.mean(Datas, axis=0)
Scaler.scale_ = np.std(Datas, axis=0, ddof=1)
ScaledDatas = Scaler.transform(Datas)
NewDatas = PCAmodel.fit_transform(ScaledDatas)
print('-------------------------------')
print('Pragma : PCA')
print('Reserved Comp Number : ', PCAmodel.n_components_)
print('Reserved Comp Ratio : ', PCAmodel.explained_variance_ratio_) # Larger is better
print('Reserved Comp Var : ', PCAmodel.explained_variance_) # Larger is better
print('Comp with Max Var : ', PCAmodel.components_)
print('Processed Data : ', NewDatas)
print('-------------------------------')
#----------------------------------------------------------------------------------------------#
# Plots #
pyplot.figure()
pyplot.scatter(NewDatas[:, 0], NewDatas[:, 1],c=Labels, marker='o', linewidths=0, alpha=1, edgecolors=None)
pyplot.title('PCA', fontsize='18', color='k')
pyplot.xlabel('First Component', fontsize='16', color='k')
pyplot.ylabel('Second Component', fontsize='16', color='k')
pyplot.legend('Labels', loc="upper right")
pyplot.savefig('PCA_plot.png')
pyplot.show()



