import os
from warnings import filterwarnings

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax, skew
from sklearn.model_selection import train_test_split

#####################################################
# Origine: Nanashi's solution # https://www.kaggle.com/jesucristo/1-house-prices-solution-top-1
# Rewrite and refine some processes
#######################################################

filterwarnings("ignore")


def make_data(test_size):
    dataframe = pd.read_csv(os.path.join("data", "train.csv"))
    dataframe = dataframe.drop(["Id"], 1)

    # dataframe = dataframe.loc[dataframe["GrLivArea"] < 4500]
    dataframe["SalePrice"] = np.log1p(dataframe["SalePrice"])

    y = dataframe["SalePrice"].copy()
    X = dataframe.drop(["SalePrice"], 1).copy()

    X["MSSubClass"] = X["MSSubClass"].apply(str)
    X["YrSold"] = X["YrSold"].astype(str)
    X["MoSold"] = X["MoSold"].astype(str)

    X["Functional"] = X["Functional"].fillna("Typ")
    X["Electrical"] = X["Electrical"].fillna("SBrkr")
    X["KitchenQual"] = X["KitchenQual"].fillna("TA")
    X["Exterior1st"] = X["Exterior1st"].fillna(X["Exterior1st"].mode()[0])
    X["Exterior2nd"] = X["Exterior2nd"].fillna(X["Exterior2nd"].mode()[0])
    X["SaleType"] = X["SaleType"].fillna(X["SaleType"].mode()[0])

    X["PoolQC"] = X["PoolQC"].fillna("None")

    X[["GarageYrBlt", "GarageArea", "GarageCars"]] = X[
        ["GarageYrBlt", "GarageArea", "GarageCars"]
    ].fillna(0)
    X[["GarageType", "GarageFinish", "GarageQual", "GarageCond"]] = X[
        ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    ].fillna("None")
    X[["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]] = X[
        ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
    ].fillna("None")

    X["MSZoning"] = X.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.fillna(x.mode()[0])
    )

    obj_cols = X.select_dtypes(object).columns
    X[obj_cols] = X[obj_cols].fillna("None")
    X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    num_cols = X.select_dtypes(numeric_dtypes).columns
    X[num_cols] = X[num_cols].fillna(0)

    skew_X = X[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skew_X.loc[skew_X > 0.5]
    skew_cols = high_skew.index

    boxcox = lambda col: boxcox1p(col, boxcox_normmax(col + 1))
    X[skew_cols] = X.loc[:, skew_cols].apply(lambda col: boxcox(col), axis=0)

    X = X.drop(["Utilities", "Street", "PoolQC"], axis=1)

    X["YrBltAndRemod"] = X["YearBuilt"] + X["YearRemodAdd"]
    X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]

    X["Total_sqr_footage"] = (
        X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["1stFlrSF"] + X["2ndFlrSF"]
    )

    X["Total_Bathrooms"] = (
        X["FullBath"]
        + (0.5 * X["HalfBath"])
        + X["BsmtFullBath"]
        + (0.5 * X["BsmtHalfBath"])
    )

    X["Total_porch_sf"] = (
        X["OpenPorchSF"]
        + X["3SsnPorch"]
        + X["EnclosedPorch"]
        + X["ScreenPorch"]
        + X["WoodDeckSF"]
    )

    X["haspool"] = X["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    X["has2ndfloor"] = X["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
    X["hasgarage"] = X["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
    X["hasbsmt"] = X["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
    X["hasfireplace"] = X["Fireplaces"].apply(lambda x: 1 if x > 0 else 0)

    X = pd.get_dummies(X).reset_index(drop=True)

    overfit = []
    for i in X.columns:
        counts = X[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(X) * 100 > 99.94:
            overfit.append(i)

    overfit = list(overfit)
    overfit.append("MSZoning_C (all)")

    X = X.drop(overfit, axis=1)

    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test
