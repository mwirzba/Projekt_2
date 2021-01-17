import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import preprocessing


class StudentPerfCol(str):
    gender = "gender"
    race_ethnicity = "race/ethnicity"
    parental_level_of_education = "parental level of education"
    lunch = "lunch"
    test_preparation_course = "test preparation course"
    math_score = "math score"
    reading_score = "reading score"
    writing_score = "writing score"


def encode_column_values_to_bit(df: DataFrame, column_name: str):
    le = preprocessing.LabelEncoder()
    le.fit(df[column_name].astype(str))
    return le.transform(df[column_name].astype(str))


def reverse_encode_column_values_to_string(df: DataFrame, column_name: str):
    le = preprocessing.LabelEncoder()
    le.fit(df[column_name])
    return le.inverse_transform(df[column_name])


def naive_bayes():
    gnb = GaussianNB()
    gnb.fit(train_inputs, train_classes)
    percent = (gnb.score(test_inputs, test_classes))
    return ["GaussianNB", percent, gnb]


def nearest_neighbors(neighbors):
    nbrs = KNeighborsClassifier(neighbors)
    nbrs.fit(train_inputs, train_classes)
    percent = (nbrs.score(test_inputs, test_classes))
    return ["NeighborsClassifier " + str(percent), percent, nbrs]


def decision_tree():
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(train_inputs, train_classes)
    percent = (dtc.score(test_inputs, test_classes))
    return ["DecisionTreeClassifier", percent, dtc]


def neural_network():
    scaler = StandardScaler()

    scaler.fit(train_inputs)

    train_data = scaler.transform(train_inputs)
    test_data = scaler.transform(test_inputs)
    print(train_data[:3])

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

    mlp.fit(train_data, train_classes)

    predictions_train = mlp.predict(train_data)
    predictions_test = mlp.predict(test_data)
    percent = (mlp.score(test_data, test_classes))

    return ["Neural Network", percent, mlp]


def replace_percens_with_grades(df: DataFrame, column_name: str):
    df.loc[df[column_name] < 60, column_name] = 0  # 'F'
    df.loc[(60 < df[column_name]) & (df[column_name] < 70), column_name] = 1  # 'D'
    df.loc[(70 < df[column_name]) & (df[column_name] < 80), column_name] = 2  # 'C'
    df.loc[(80 < df[column_name]) & (df[column_name] < 90), column_name] = 3  # 'B'
    df.loc[(90 < df[column_name]) & (df[column_name] < 100), column_name] = 4  # 'A'
    return df



df = pd.read_csv('StudentsPerformance.csv')

print(df[StudentPerfCol.reading_score].max())
print(df[StudentPerfCol.reading_score].min())
print(df[StudentPerfCol.reading_score].mean())
print(df.nunique())


def show_plot_comparision(column_name: str):
    plt.figure(figsize=(18, 8))
    df[column_name].value_counts(normalize=True)
    fig = df[column_name].value_counts(dropna=False).plot.bar(color=['blue'])
    # color=['black', 'red', 'green', 'blue', 'cyan'])
    """
    for p in fig.patches:
        fig.annotate(format(p.get_height(), '.1f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),
                     textcoords='offset points')

    plt.xticks(rotation=90, ha='right')
    """
    plt.title('Kolumna:' + column_name)
    plt.xlabel('Oceny z testu z czytania')
    plt.ylabel('Ilość')
    plt.show()


def prepare_inputs():
    global train_inputs, test_inputs, train_classes, test_classes
    #df[StudentPerfCol.gender] = encode_column_values_to_bit(df, StudentPerfCol.gender)
    df[StudentPerfCol.race_ethnicity] = encode_column_values_to_bit(df, StudentPerfCol.race_ethnicity)
    df[StudentPerfCol.parental_level_of_education] = encode_column_values_to_bit(df,
                                                                                 StudentPerfCol.parental_level_of_education)
    df[StudentPerfCol.lunch] = encode_column_values_to_bit(df, StudentPerfCol.lunch)
    df[StudentPerfCol.test_preparation_course] = encode_column_values_to_bit(df, StudentPerfCol.test_preparation_course)

    all_inputs = df[[
        StudentPerfCol.race_ethnicity,
        StudentPerfCol.parental_level_of_education,
        StudentPerfCol.lunch,
        StudentPerfCol.test_preparation_course,
        StudentPerfCol.writing_score,
        # StudentPerfCol.reading_score,
        StudentPerfCol.math_score
    ]].values
    all_classes = df[StudentPerfCol.gender]

    (train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs,
                                                                                all_classes,
                                                                                train_size=0.8,
                                                                                random_state=1)


"""
for i in [10,20,30,40,50,60,70,80,90,100,150,200]:
    prepare_inputs()
    results = nearest_neighbors(i)
    print(str( str(i) + "    "  + str(results[1])))
"""
all_results = []

prepare_inputs()
results = naive_bayes()
all_results.append(results)
print( results[0] + ""+ str(results[1]))


results = decision_tree()
all_results.append(results)
print( results[0] + ""+ str(results[1]))


results = nearest_neighbors(70)
all_results.append(results)
print( results[0] + ""+ str(results[1]))


results = neural_network()
all_results.append(results)
print( results[0] + ""+ str(results[1]))


for r in all_results:
    plot_confusion_matrix(r[2], test_inputs, test_classes)
    plt.show()


def print_diagram(data: [[]]):
    fig = plt.figure()
    d = [pair[1] for pair in data]
    legend = [pair[0] for pair in data]
    plt.bar(legend, d)
    plt.xlabel("Klasyfikatory")
    plt.ylabel("Procenty")
    plt.show()
