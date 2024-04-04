import aaanalysis as aa
aa.options["verbose"] = False  # Disable verbosity


def main():
    df_seq = aa.load_dataset(name="DOM_GSEC")
    labels = df_seq["label"].to_list()
    df_feat = aa.load_features(name="DOM_GSEC").head(100)
    print(df_feat)
    print(df_seq.columns)
    # Create feature matrix
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq)
    print(df_parts)
    X = sf.feature_matrix(features=df_feat["feature"], df_parts=df_parts)

    tm = aa.TreeModel()
    tm = tm.fit(X, labels=labels)

    pred, pred_std = tm.predict_proba(X)
    print(pred)
    df_seq["prediction"] = pred
    df_seq["pred_std"] = pred_std
    print(df_seq.columns)
    print(df_seq)
    print("Prediction scores for 5 substrates")
    aa.display_df(df_seq.head(5))

    print("Prediction scores for 5 non-substrates")
    aa.display_df(df_seq.tail(5))


if __name__ == "__main__":
    main()
