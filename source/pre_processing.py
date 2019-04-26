import numpy as np
import pandas

def pre_process():
    # Read the file
    df = pandas.read_csv(r"../data/pre_processed_train.csv")

    # One hot encode genders
    df["gender"] = pandas.Categorical(df["gender"])
    dfDummies = pandas.get_dummies(df['gender'], prefix='is_gender', )
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode races
    df["race/ethnicity"] = pandas.Categorical(df["race/ethnicity"])
    dfDummies = pandas.get_dummies(df["race/ethnicity"], prefix="is_race")
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode parental level of education
    df["parental level of education"] = pandas.Categorical(df["parental level of education"])
    dfDummies = pandas.get_dummies(df['parental level of education'], prefix='is_parent_education')
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode lunch
    df["lunch"] = pandas.Categorical(df["lunch"])
    dfDummies = pandas.get_dummies(df['lunch'], prefix='is_lunch')
    df = pandas.concat([df, dfDummies], axis=1)

    # One hot encode test preparation course
    df["test preparation course"] = pandas.Categorical(df["test preparation course"])
    dfDummies = pandas.get_dummies(df['test preparation course'], prefix='is_prepared')
    df = pandas.concat([df, dfDummies], axis=1)
    df = df.drop(columns=["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"])

    return df

if __name__ == '__main__':
    pre_process()