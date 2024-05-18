# Çalışan Memnuniyeti ve Performans Verileri ile İşten Ayrılma Olasılığını Tahmin Etme

Bu proje, çalışan memnuniyeti ve performans verilerini kullanarak çalışanların işten ayrılma olasılığını tahmin etmeyi amaçlamaktadır. Bu amaç doğrultusunda veri analizi, görselleştirme teknikleri ve çeşitli makine öğrenimi modelleri kullanılmıştır.

## İçindekiler

- [Kurulum](#kurulum)
- [Veri Seti](#veri-seti)
- [Veri Analizi ve Görselleştirme](#veri-analizi-ve-görselleştirme)
- [Makine Öğrenimi Modelleri](#makine-öğrenimi-modelleri)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunanlar](#katkıda-bulunanlar)
- [Lisans](#lisans)

## Kurulum

Bu projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. Gerekli Python kütüphanelerini yükleyin:
    ```bash
    pip install pandas numpy matplotlib seaborn plotly scikit-learn lazypredict lightgbm xgboost
    ```

2. Proje dosyalarını indirin veya klonlayın:
    ```bash
    git clone https://github.com/kullaniciadi/projeadi.git
    cd projeadi
    ```

## Veri Seti

Veri seti, `HR_capstone_dataset.csv` dosyasından yüklenmiştir. Bu veri seti, çalışanların memnuniyet düzeyi, son değerlendirme, proje sayısı, ortalama aylık çalışma saatleri, şirkette geçirilen süre, iş kazası, son 5 yıldaki terfi durumu, departman ve maaş bilgilerini içermektedir.

## Veri Analizi ve Görselleştirme

Veri analizi ve görselleştirme adımları aşağıdaki gibidir:

1. Veri yükleme ve ilk inceleme:
    ```python
    import pandas as pd

    hr_df = pd.read_csv('HR_capstone_dataset.csv')
    df = hr_df.copy()
    df.head()
    ```

2. Veri setinin genel yapısını inceleme:
    ```python
    df.info()
    df.describe()
    ```

3. Eksik ve yinelenen değerlerin kontrolü ve temizlenmesi:
    ```python
    df.isnull().sum()
    df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    ```

4. Kategorik ve numerik değişkenlerin dağılımlarının görselleştirilmesi:
    ```python
    import plotly.express as px

    for col in df.columns:
        if df[col].dtype == 'object':
            px.histogram(df, x=col).show()
        else:
            px.box(df[col]).show()
            px.histogram(df, x=col, title=f'{col} Histogram').show()
    ```

5. Korelasyon matrisi ve ısı haritası:
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues')
    plt.show()
    ```

## Makine Öğrenimi Modelleri

Çeşitli makine öğrenimi modelleri kullanılarak çalışanların işten ayrılma olasılığı tahmin edilmiştir:

1. Veriyi eğitim ve test setlerine ayırma:
    ```python
    from sklearn.model_selection import train_test_split

    X = df.drop('left', axis=1)
    y = df['left']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=43)
    ```

2. LazyPredict kullanarak model karşılaştırması:
    ```python
    from lazypredict.Supervised import LazyClassifier

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
    ```

3. RandomForest ve LightGBM modelleri ile en iyi sonuçların elde edilmesi:
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import lightgbm as lgb
    from sklearn.model_selection import RandomizedSearchCV

    # RandomForest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # LightGBM
    param_dist = {...}
    lgb_model = lgb.LGBMClassifier()
    lgb_random = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_dist, n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=-1)
    lgb_random.fit(X_train, y_train)
    print(f"En İyi Parametreler: {lgb_random.best_params_}")
    best_lgb = lgb_random.best_estimator_
    y_pred = best_lgb.predict(X_test)
    print(classification_report(y_test, y_pred))
    ```

## Sonuçlar

RandomForest ve LightGBM modelleri, çalışanların işten ayrılma olasılığını yüksek doğruluk oranları ile tahmin etmiştir. Özellikle LightGBM modeli ile %98'in üzerinde doğruluk oranı elde edilmiştir.

## Katkıda Bulunanlar

- [Kullanıcı Adı](https://github.com/yusuffenes)

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
