import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df=pd.read_csv("C:\VBO_DOSYALAR\ders öncesi notlar\Telco-Customer-Churn.csv")

df.drop(columns="customerID", inplace=True)
#TotalCharges değişkenin içindeki boşluktan dolayı object anladı bunu float olaak değiştiriyoruz
df['TotalCharges'].replace([' '], '0.0', inplace=True)
df["TotalCharges"] = df["TotalCharges"].astype(float)

#hedef değişkenimizle diğer değişkenler arasınaki ilişkiyi görmek için integera çeviriyoruz
df["Churn"].replace(["Yes"], "1", inplace=True)
df["Churn"].replace(["No"], "0", inplace=True)
df["Churn"] = df["Churn"].astype(int)

#############################################################
# GÖREV - 1
#############################################################

#ADIM-1 Genel veriseti incelemesi
df.shape
df.head(10)
df.describe().T

#ADIM-2 Numerik kategorik değişkenlerin yakanlaması

#Kategorik değişkenler, numerik değişkenler, kategorik görünüp kardinal olan değişkenler ve
#numerik görünüp kategorik olan değişkenleri  grap_col_name fonksiyonu ile yakalıyoruz
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols'ta kolon tipi object olan değişkenleri kategorik değişkenler olarak atadık
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    #num_but_cat ile df içindeki kolonlarda tipi object olmayan ve eşsiz sınıf sayısı kullanıcının
     #verdiği cat_th değerinden az olan kolonlar olarak belirledik bu sayede numerik görünümlü
     #kategorikleri elde ettik
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    #cat_but_car ile df içindeki kolonlarda tipi object olup eşsiz sınıf sayısı kullanıcının verdiği
     #car_th değerinden büyük olan değişkenleri belieledik kategorik görünümlü kardinalleri elde ettik
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    #kategorik değişkenler içine numerik görünümlü kategorikleri ekledik
    cat_cols = cat_cols + num_but_cat

    #kategorik kolonlar içinden kategorik görünümlü kardinalleri çıkardık
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #numerik değişkenleri, kolon tipi object olmayan olarak tanımladık
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    #numerik değişkenler içinden numerik görümümlü kategorik değişkenleri çıkardık
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

#fonksiyonu çalıştırarak kategorik,numerik ve kardinal değişkenleri elde ettik
cat_cols, num_cols, cat_but_car= grab_col_names(df)

#ADIM-3 Numerik ve kategorik değişkenlerin analizinin yapılması

#cat_summary fonksiyonu ile kategorik değişkenler için sınıf sayılarını ve sınıf sayılarının oranlarını
#ayrıca istenirse eksik gözlemi ve grafiği gösteriyor
def cat_summary(dataframe, col_name, plot=False, null=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()
    if null:
        print(col_name, "NaN number: ", dataframe[col_name].isnull().sum())

#Tek kategorik değişkenimiz olan Outcome için fonksiyonu uyguluyoruz
cat_summary(df,cat_cols[0],null=True,plot=False)

#num_summary fonksiyonu numerik sütunlar için çeyrek değerleri ve tanımlayıcı istatistikleri
# ayrıca istenirse box-plot grafiğini çizdiriyor
def num_summary(dataframe, numerical_col, boxplot=False):
    quantiles = [0.05, 0.10, 0.50, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if boxplot:
        sns.boxplot(x=dataframe[numerical_col])
        plt.xlabel(numerical_col)
        plt.show()

for i in num_cols:
    num_summary(df,i,boxplot=True)

#ADIM-4 Hedef değişken analizinin yapılması

#target_analyser fonksiyonu ile hedef değişken kırılımında kategorik değişkenler için sınıf sayılarını,
# sınıf sayılarının oranları ve ategorik değişkenlere göre hedef değişkenin ortalaması
#numerik değişkenler için hedef değişkene göre numerik değişkenlerin ortalaması hesaplanır.
def target_analyser(dataframe, target, num_cols, cat_cols):
    for col in dataframe.columns:
        if col in cat_cols:
            print(col, ":", len(dataframe[col].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                                "RATIO": dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
        if col in num_cols:
            print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(target)[col].mean()}), end="\n\n\n")

target_analyser(df, "Churn", num_cols, cat_cols)

#ADIM-5 Aykırı gözlem analizinin yapılması

#df için alt ve üst çeyreklere göre aykırı değelerin olup olmadığını buluyoruz
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,num_cols)

#ADIM-6 Eksik gözlem analizinin yapılması

#Veride eksik gözlem olup olmadığını kontrol ediyoruz.
df.isnull().sum()

#ADIM-6 Korelasyon analizinin yapılması

#sns_heatmap fonksiyonu ile değişkenlerin birbiri arasındaki korelasyon değerlerine baktık
def sns_heatmap(dataset, color):
    heatmap = sns.heatmap(dataset.corr(), vmin=-1, vmax=1, annot=True, cmap=color)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()

sns_heatmap(df,color='Greens')

#tenur değişkeniyle arasında negatif yönlü .35lik bir ilişki var değişken türetirken bunu kullanacağız
#############################################################
# GÖREV - 2
#############################################################

#ADIM-2 Yeni değişkenlerin oluşturulması

df.loc[(df["tenure"] <= 20) , 'New_tenur'] = 'low'
df.loc[(df["tenure"] > 20) &(df["tenure"]<=60) , 'New_tenur'] = 'mid'
df.loc[(60 < df["tenure"]) , 'New_tenur'] = 'high'

df.loc[(df["gender"] == "Female") & (df["SeniorCitizen"] == 1), 'Age_Gender'] = 'Old_Woman'
df.loc[(df["gender"] == "Female") & (df["SeniorCitizen"] == 0), 'Age_Gender'] = 'Young_Woman'
df.loc[(df["gender"] == "Male") & (df["SeniorCitizen"] == 1), 'Age_Gender'] = 'Old_Man'
df.loc[(df["gender"] == "Male") & (df["SeniorCitizen"] == 0), 'Age_Gender'] = 'Young_Man'

df.loc[(df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes"), 'TV_Movie'] = 'Both_have'
df.loc[(df["StreamingTV"] == "No") & (df["StreamingMovies"] == "Yes"), 'TV_Movie'] = 'Only_Movies'
df.loc[(df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "No"), 'TV_Movie'] = 'Only_TV'
df.loc[(df["StreamingTV"] == "No") & (df["StreamingMovies"] == "No"), 'TV_Movie'] = 'Both_have_not'

df.loc[(df["OnlineSecurity"] == "Yes") & (df["DeviceProtection"] == "Yes"), 'Security_Online&Device'] = 'Both_have'
df.loc[(df["OnlineSecurity"] == "Yes") & (df["DeviceProtection"] == "No"), 'Security_Online&Device'] = 'Only_Online'
df.loc[(df["OnlineSecurity"] == "No") & (df["DeviceProtection"] == "Yes"), 'Security_Online&Device'] = 'Only_Device'
df.loc[(df["OnlineSecurity"] == "No") & (df["DeviceProtection"] == "No"), 'Security_Online&Device'] = 'Both_have_not'


#ADIM-3 Encoding işlemlerinin gerçekleştirilmesi

#yeni oluşturulduğumuz değişkenler sonrası dfi tekrar kategorik, numerik, kardinal ayırdık
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#Hücrelerinde sadece iki değişken olan sütunları bulalım
binary_cols = [col for col in df.columns if df[col].dtype == object
               and df[col].nunique() == 2]

#label_encoder ile binary kolonları 0-1 olarak değiştiriyoruz.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)
df.head()

#2den daha fazla string içeren karegorik değişkenleri drop_first=True yani
#dummy değişken tuzağına düşmeden 0-1 olarak dönüştürdük.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, ohe_cols)
df.head()
grab_col_names(df)

#ADIM-4 Numerik değişkenlere standartlaştırma yapılması

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#ADIM-5 Modelin kurulması

#Bağımlı yani hedef değişkeni tanımladık
y = df["Churn"]
X = df.drop(["Churn"], axis=1)

#train ve test setini 70e 30 ayırdık
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier



#radomforest ile model kurduk
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#Hangi değişkenin modelde tahminde daha çok etki ettiğini bulmak için baktık
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)
