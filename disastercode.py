import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm

import seaborn as sns
import matplotlib.pyplot as plt

#veri seti data frame e yüklenir.
datadf=pd.read_csv("disasters.csv")



#DATA PREPROCESSİNG

#1)Feature engineering
#Başlangıç ve bitiş yılları çok büyük oranla aynı olduğundan o sutunları dusurdum ve ay bilgisini kullanıp mevsim adında yeni bir sutun oluşturdum.
#Bunun için öncelikle NULL olan değerleri doldurdum
datadf.drop(["Start Day","End Day","End Year","Start Year","End Month"],axis=1,inplace=True)
datadf.fillna(value={"Start Month" :datadf["Start Month"].mean().astype(int)},inplace=True)
datadf['Start Month'] = datadf['Start Month'].replace([3,4,5],'Spring')
datadf['Start Month'] = datadf['Start Month'].replace([6,7,8],'Summer')
datadf['Start Month'] = datadf['Start Month'].replace([9,10,11],'Fall')
datadf['Start Month'] = datadf['Start Month'].replace([12,1,2],'Winter')
datadf.rename(columns={"Start Month" :"Season"},inplace=True)

#Country sutunu ile aynı bilgiyi verdiği için Location sutununu dusurduk
datadf.drop(["Location"],axis=1,inplace=True)

#Disaster Magnitude Value sutunu çok büyük oranda boş olduğundan o sutunun birimini gösteren Dis Mag Scale sutunu da dusurulur.
datadf.drop(["Dis Mag Scale"],axis=1,inplace=True)

#Country sutunu ile aynı olduğundan düşürüldü.
datadf.drop(["ISO"],axis=1,inplace=True)

#Continent sutunu ile çok büyük oranda  aynı bilgiyi verdiği için düşürüldü.
datadf.drop(["Region"],axis=1,inplace=True)

#No effected ve Total effected sutunları aynı bilgiyi bize verdiği  için birini düşürdük. Daha buyuk oranda boş olanı dusurduk.
datadf.drop(["No Affected"],axis=1,inplace=True)

# Tüm satırlardaki değeri aynı olduğundan -Natural- Disaster Group sutunu dusurdum.
datadf.drop(["Disaster Group"],axis=1,inplace=True)

#2)Missing data
#doldurulamayacak oranda boş olan columnların veri setinden silinmesi
columnstodrop=["Glide","Disaster Subsubtype","Event Name","Origin","Associated Dis","Associated Dis2","OFDA Response","Appeal","Declaration",
"Aid Contribution","Dis Mag Value","Latitude","Longitude","Local Time","River Basin","No Injured",
"No Homeless","Insured Damages ('000 US$)","Total Damages ('000 US$)","Adm Level","Admin1 Code","Admin2 Code","Geo Locations"]
datadf.drop(columnstodrop,axis=1,inplace=True)   #%51-%96 oranında boş olan columnları veri setinden düşürülür.

#NULL değeri olan columnların doldurulması
#%19 boş olan kategorik sutunun label mode  ile doldurulması
datadf["Disaster Subtype"].fillna(datadf.groupby('Disaster Subgroup')['Disaster Subtype'].transform(lambda x : next(iter(x.mode()), np.nan)),inplace=True)
datadf=datadf.dropna(subset=["Disaster Subtype"])

#%29 u boş olan nümerik sutunun label ortalamasına göre doldurulması-Max değer ile diğer değerler çok farklı olduğundan normal ortalama kullanmadık-
datadf['Total Deaths'].fillna(datadf.groupby('Disaster Subgroup')['Total Deaths'].transform('mean'),inplace=True)
datadf=datadf.dropna(subset=["Total Deaths"])

#%29 u boş olan nümerik sutunun label ortalamasına göre doldurulması-Max değer ile diğer değerler çok farklı olduğundan normal ortalama kullanmadık-
datadf['Total Affected'].fillna(datadf.groupby('Disaster Subgroup')['Total Affected'].transform('mean'),inplace=True)

#%2 si boş olan nümerik sutunun ortalmaya göre doldurulması
datadf.fillna(value={"CPI" :datadf["CPI"].mean()},inplace=True)

#3)Kategorik datanın nümeriğe çevrilmesi
# Label olacak olan sutunu -Disaster Subgroup- Ordinal encoding ile sayılara çevrilmesi
ord_enc = OrdinalEncoder()
datadf["Target Value"] = ord_enc.fit_transform(datadf[["Disaster Subgroup"]])
datadf.drop(["Disaster Subgroup"],axis=1,inplace=True)


#Kategorik datamızda unique değer çok fazla olduğu için direkt olarak One-Hot-Encoding yapılmıyor bu sebeple grupları bir araya topladım.
#Standart %5 ve altının yeni bir grupta toplanması ama ben bizim datamızın özelinde sutunlara göre yeni değerler belirledim.
def combine(percent,name):
    mask = datadf[name].map(datadf[name].value_counts()) < int(datadf.shape[0])*percent
    datadf[name] =  datadf[name].mask(mask, 'other')

combine(0.04,"Disaster Type")
combine(0.04,"Disaster Subtype")
combine(0.02,"Country")

#One Hot Encoding ile kategorik data nümeriğe dönüştürülür
def onehotencode(name):
    encoder=ce.OneHotEncoder(cols=name,handle_unknown='return_nan',return_df=True,use_cat_names=True)
    x = encoder.fit_transform(datadf)
    return x

datadf=onehotencode("Disaster Type")
datadf=onehotencode("Disaster Subtype")
datadf=onehotencode("Country")
datadf=onehotencode("Continent")
datadf=onehotencode("Season")



# Random Forest modelinin oluşturulması
#Ağaç yapısından dolayı random forest de outlier dedection ve scale işlemi yapılmasına gerek yoktur.
X=datadf[['Year', 'Seq', 'Disaster Type_Drought', 'Disaster Type_Earthquake',
       'Disaster Type_other', 'Disaster Type_Storm', 'Disaster Type_Flood',
       'Disaster Type_Epidemic', 'Disaster Type_Landslide',
       'Disaster Subtype_Drought', 'Disaster Subtype_Ground movement',
       'Disaster Subtype_other', 'Disaster Subtype_Tropical cyclone',
       'Disaster Subtype_Riverine flood', 'Disaster Subtype_Bacterial disease',
       'Disaster Subtype_Convective storm', 'Disaster Subtype_Flash flood',
       'Country_other', 'Country_India', 'Country_Bangladesh', 'Country_China',
       'Country_Indonesia', 'Country_United States of America (the)',
       'Country_Japan', 'Country_Philippines (the)', 'Continent_Africa',
       'Continent_Asia', 'Continent_Americas', 'Continent_Europe',
       'Continent_Oceania', 'Season_Summer', 'Season_Spring', 'Season_Fall',
       'Season_Winter', 'Total Deaths', 'Total Affected', 'CPI']]
y=datadf["Target Value"]

# 80% training and 20% test olarak datanın ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf=RandomForestClassifier(n_estimators=10) # 10-100 arası değerleri denedim fakat önemli ölçüde değişim olmadığından 100 de karar kıldım
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
report = metrics.classification_report(y_test, y_pred)
print("Random Forest other metrics :")
print(report)
matrix = metrics.confusion_matrix(y_test, y_pred)
print("Random Forest Confusion Matrix :")
print(matrix)
print("Random Forest AUC score",metrics.roc_auc_score(y_test,clf.predict_proba(X_test),multi_class='ovo'))
"""
#görselleştirme için en yüksek oranda başarı veren algoritma olan RF kullanıldı.
xtestvisual=X_test.copy()
y_predvisual=list(y_pred)
xtestvisual["target"]=y_predvisual
continents=["Africa","Asia","America","Europe","Oceania"]

def visuallist(target):
    l1=list()
    l1.append(len(xtestvisual[(xtestvisual["target"]==target) & (xtestvisual["Continent_Africa"]==1)]))
    l1.append(len(xtestvisual[(xtestvisual["target"]==target) & (xtestvisual["Continent_Asia"]==1)]))
    l1.append(len(xtestvisual[(xtestvisual["target"]==target) & (xtestvisual["Continent_Americas"]==1)]))
    l1.append(len(xtestvisual[(xtestvisual["target"]==target) & (xtestvisual["Continent_Europe"]==1)]))
    l1.append(len(xtestvisual[(xtestvisual["target"]==target) & (xtestvisual["Continent_Oceania"]==1)]))
    return l1

biologicallist=visuallist(0.0)
climatologicallist=visuallist(1.0)
geophysicallist=visuallist(2.0)
hydrologicallist=visuallist(3.0)
meterologicallist=visuallist(4.0)


ax=plt.subplot(2,3,1)
plt.bar(continents,biologicallist,color="pink")
plt.xlabel("Continents")
plt.ylabel("count")
plt.title("Biological Disasters")
plt.xticks(fontsize=9)


ax1=plt.subplot(2,3,2)
plt.bar(continents,climatologicallist,color="lightblue")
plt.xlabel("Continents")
plt.ylabel("count")
plt.title("Climatological Disasters")
plt.xticks(fontsize = 9)

ax2=plt.subplot(2,3,3)
plt.bar(continents,geophysicallist,color="grey")
plt.xlabel("Continents")
plt.ylabel("count")
plt.title("Geophysical Disasters")
plt.xticks(fontsize = 9)

ax3=plt.subplot(2,3,4)
plt.bar(continents,hydrologicallist,color="black")
plt.xlabel("Continents")
plt.ylabel("count")
plt.title("Hydrological Disasters")
plt.xticks(fontsize = 9)

ax4=plt.subplot(2,3,5)
plt.bar(continents,meterologicallist,color="purple")
plt.xlabel("Continents")
plt.ylabel("count")
plt.title("Meterological Disasters")
plt.xticks(fontsize = 9)

plt.suptitle("DİSASTERS")
plt.subplots_adjust(wspace=0.5,hspace=0.5)

plt.show()
"""
#KNN modelinin oluşturulması
#Daha iyi sonuç alabilmek için öncelikle veri setini normalize ettim.
scaler = StandardScaler()
scaler.fit(X)
scaled = scaler.fit_transform(X)
scaled_X = pd.DataFrame(scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)# 3 ile 35 arasınadki değerleri denedim en yüksek sonucu veren komşu sayısana karar verdim
knn.fit(X_train, y_train)
y_pred2 = knn.predict(X_test)

print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred2))
report2 = metrics.classification_report(y_test, y_pred2)
print("KNN other metrics :")
print(report2)
matrix2 = metrics.confusion_matrix(y_test, y_pred2)
print("KNN Confusion Matrix :")
print(matrix2)
print("KNN AUC score",metrics.roc_auc_score(y_test,knn.predict_proba(X_test),multi_class='ovo'))

#SVM modelinin oluşturulması
#Daha yüksek başarı oranı için SVM de de scaled data kullanılır.
svmclf = svm.SVC(kernel='linear',probability=True) #  rbf linear ve poly kernel tipleri denenip linear seçilmiştir.
svmclf.fit(X_train, y_train)
y_pred3 = svmclf.predict(X_test)

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred3))
report3 = metrics.classification_report(y_test, y_pred3)
print("SVM other metrics :")
print(report3)
matrix3 = metrics.confusion_matrix(y_test, y_pred3)
print("SVM Confusion Matrix :")
print(matrix3)
print("SVM AUC score",metrics.roc_auc_score(y_test,svmclf.predict_proba(X_test),multi_class='ovo'))
