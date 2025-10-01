import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from joblib import dump


def preprocess_data(data, target_column, save_path):
    """
    Fungsi preprocessing dataset Diabetes dengan SMOTE otomatis
    """
    # Encode target
    le = LabelEncoder()
    data[target_column] = le.fit_transform(data[target_column])

    # Ordinal mapping
    edu_map = {"No formal": 0, "Highschool": 1, "Graduate": 2, "Postgraduate": 3}
    inc_map = {"Low": 0, "Lower-Middle": 1, "Middle": 2, "Upper-Middle": 3, "High": 4}
    if "education_level" in data.columns:
        data["education_level"] = data["education_level"].map(edu_map)
    if "income_level" in data.columns:
        data["income_level"] = data["income_level"].map(inc_map)

    # entukan kolom numerik & kategorikal
    categorical_features = ["gender", "ethnicity", "employment_status", "smoking_status", "activity_level"]
    numeric_features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_column in numeric_features:
        numeric_features.remove(target_column)
    for col in categorical_features.copy():
        if col not in data.columns:
            categorical_features.remove(col)

    # Pipeline numerik
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Pipeline kategorikal
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Split X dan y
    X = data.drop(columns=[target_column,], errors="ignore")
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Transformasi
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Ambil nama kolom hasil transformasi
    feature_names = (
        numeric_features +
        list(preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features))
    )

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.Series(y_train, name=target_column)
    y_test = pd.Series(y_test, name=target_column)

    # erapkan SMOTE langsung
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("✅ SMOTE otomatis diterapkan, data training sudah balance.")

    # Simpan pipeline
    dump(preprocessor, save_path)
    print(f"✅ Pipeline preprocessing disimpan di {save_path}")

    return X_train, X_test, y_train, y_test
