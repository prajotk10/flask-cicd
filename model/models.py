import numpy as np
import pandas as pd
# data = pd.read_csv(r'/home/ec2-user/MegaProject/Dataset/modified_dataset.csv')
# data.sample(5)
# data.shape
# (data['Disease'].unique())
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# X = data.drop(columns=['Disease'])
# y = data['Disease']
# y
# le =LabelEncoder()
# y = le.fit_transform(y)
# y
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)
# X_train.shape,X_test.shape,y_train.shape,y_test.shape
# from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
# from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
# svc = SVC(kernel='sigmoid',gamma=1.0)
# knc = KNeighborsClassifier()
# mnb = MultinomialNB()
# dtc = DecisionTreeClassifier(max_depth=5)
# lrc = LogisticRegression(solver='liblinear',penalty='l1')
# rfc = RandomForestClassifier(n_estimators=50,random_state=2)
# abc = AdaBoostClassifier(n_estimators=50,random_state=2)
# bc = BaggingClassifier(n_estimators=50,random_state=2)
# etc = ExtraTreesClassifier(n_estimators=50,random_state=2)
# gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
# xgb = XGBClassifier(n_estimators=50,random_state=2)
# gnb = GaussianNB()
# mnb = MultinomialNB()
# bnb = BernoulliNB()
# accuracy_bar = []
# precision_bar = []
# def train_classifier(clf,X_train,y_train,X_test,y_test):
#     clf.fit(X_train,y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test,y_pred)
#     precision = precision_score(y_test,y_pred,average='micro')
#     accuracy_bar.append(accuracy)
#     precision_bar.append(precision)
#     confusion = confusion_matrix(y_test,y_pred)
#     return accuracy,precision,confusion
#     train_classifier(svc,X_train,y_train,X_test,y_test)
#     train_classifier(mnb,X_train,y_train,X_test,y_test)
#     train_classifier(dtc,X_train,y_train,X_test,y_test)
#     train_classifier(lrc,X_train,y_train,X_test,y_test)
#     train_classifier(rfc,X_train,y_train,X_test,y_test)
#     train_classifier(abc,X_train,y_train,X_test,y_test)
#     train_classifier(bc,X_train,y_train,X_test,y_test)
#     train_classifier(etc,X_train,y_train,X_test,y_test)
#     train_classifier(gbdt,X_train,y_train,X_test,y_test)
#     train_classifier(xgb,X_train,y_train,X_test,y_test)
#     train_classifier(gnb,X_train,y_train,X_test,y_test)
#     train_classifier(mnb,X_train,y_train,X_test,y_test)
#     train_classifier(bnb,X_train,y_train,X_test,y_test)
# print(accuracy_bar)
# print(precision_bar)
# categories = ['SVC','MultinomialNB','DecisionTree','LogisticRegression','RandomForest','AdaBoostClassifier','BaggingClassifier','ExtraTreeClassfier','GradientBoosting','XGBClassifier','GaussianNB','MultinomialNB','BernoulliNB']
# import matplotlib.pyplot as plt
# plt.barh(categories,accuracy_bar, color='maroon')
# plt.ylabel('Categories')
# plt.xlabel('Accuracy')
# plt.title('Accuracy Score of models')
# plt.show()
# plt.barh(categories,precision_bar, color='maroon')
# plt.ylabel('Categories')
# plt.xlabel('Precision')
# plt.title('Precision Score of models')
# plt.show()
import pickle
# pickle.dump(mnb,open('MNB.pkl','wb'))
# pickle.dump(rfc,open('RFC.pkl','wb'))
# mnb = pickle.load(open('MNB.pkl','rb'))
rfc = pickle.load(open(r'./model/RFC.pkl','rb'))
#test 1
print("Predicted Label : ",rfc.predict(X_test.iloc[0].values.reshape(1,-1)))
print("Actual Label :",y_test[0])
#test 2
print("Predicted Label : ",rfc.predict(X_test.iloc[10].values.reshape(1,-1)))
print("Actual Label :",y_test[10])
category_mapping = dict(zip(le.classes_, range(len(le.classes_))))
category_mapping
disease_dict = {value: key for key, value in category_mapping.items()}
disease_dict
symptom_dict = {col: i + 1 for i, col in enumerate(X.columns)}
symptom_dict
treatment = pd.read_csv(r'./model/treatment.csv', encoding="cp1252")
treatment.sample()
symptom_dict = {'Blindness': 1,
 'Bronchitis': 2,
 'Canker': 3,
 'Caseation': 4,
 'Cephalitis': 5,
 'Coccidiosis': 6,
 'Distension': 7,
 'Dyspnea': 8,
 'Inflammation': 9,
 'Tremors': 10,
 'Asphyxiation': 11,
 'Cyanosis': 12,
 'Disheveled': 13,
 'Immobility': 14,
 'Malaise': 15,
 'Mycosis': 16,
 'Oviposition': 17,
 'Respiratory distress': 18,
 'Rhinorrhea': 19,
 'Arthralgia': 20,
 'Dehydration': 21,
 'Edema': 22,
 'Immobilization': 23,
 'Inactivity': 24,
 'Kyphosis': 25,
 'Neoplasia': 26,
 'Omphalitis': 27,
 'Plucking': 28,
 'Rhinorrhea.1': 29,
 'Ulceration': 30,
 'Epiphora': 31,
 'Listlessness': 32,
 'Opacification': 33,
 'Ovulation': 34,
 'Pasty': 35,
 'Splenomegaly': 36,
 'Suppuration': 37,
 'Toxin': 38,
 'Dampness': 39,
 'Diarrhea': 40,
 'Stop laying': 41,
 'Sudden death': 42,
 'Petechiae': 43}
disease_dict = {0: 'Avian Influenza',
 1: 'Botulism',
 2: 'Bumblefoot',
 3: 'Fowl Cholera',
 4: 'Fowl Pox',
 5: 'Infectious Bronchitis',
 6: 'Infectious Coryza',
 7: 'Marek’s Disease',
 8: 'Mushy Chick Disease',
 9: 'Newcastle Disease',
 10: 'Pullorum',
 11: 'Quail Disease',
 12: 'Thrush'}
reversed_conditions_synonyms = {
    "Can't see": "Blindness",
    "Chest infection": "Bronchitis",
    "Sore": "Canker",
    "Cheese-like tissue": "Caseation",
    "Brain swelling": "Cephalitis",
    "Parasite infection": "Coccidiosis",
    "Bloating": "Distension",
    "Hard to breathe": "Dyspnea",
    "Swelling": "Inflammation",
    "Shaking": "Tremors",
    "Can't breathe": "Asphyxiation",
    "Blue skin": "Cyanosis",
    "Messy": "Disheveled",
    "Can't move": "Immobility",
    "Feeling unwell": "Malaise",
    "Fungal infection": "Mycosis",
    "Laying eggs": "Oviposition",
    "Breathing trouble": "Respiratory distress",
    "Runny nose": "Rhinorrhea",
    "Joint pain": "Arthralgia",
    "Thirsty": "Dehydration",
    "Swelling": "Edema",
    "Can't move": "Immobilization",
    "Not moving": "Inactivity",
    "Hunchback": "Kyphosis",
    "Tumor": "Neoplasia",
    "Belly button infection": "Omphalitis",
    "Pulling out": "Plucking",
    "Runny nose": "Rhinorrhea",
    "Sore": "Ulceration",
    "Watery eyes": "Epiphora",
    "No energy": "Listlessness",
    "Cloudy": "Opacification",
    "Egg release": "Ovulation",
    "Doughy": "Pasty",
    "Big spleen": "Splenomegaly",
    "Pus": "Suppuration",
    "Poison": "Toxin",
    "Wetness": "Dampness",
    "Runny poop": "Diarrhea",
    "No eggs": "Stop laying",
    "Unexpected death": "Sudden death",
    "Red spots": "Petechiae"
}
def get_predicted_value(symptoms):
    input_vector = np.zeros(len(symptom_dict))
    for item in symptoms:
        if item in symptom_dict:
            index = symptom_dict[item]
            if index < len(input_vector):
                input_vector[index] = 1
    prediction = rfc.predict([input_vector])[0]
    if prediction in disease_dict:
        return disease_dict[prediction]
def get_selected_values(dictionary, keys):
    return [dictionary[key] for key in keys if key in dictionary]
    # test 
symptoms = input("Enter your Symptoms....................")
user_symptoms = [s.strip() for s in symptoms.split(',')]
user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
user_symptoms1 = get_selected_values(reversed_conditions_synonyms, user_symptoms)
# print(user_symptoms1)
predicted_disease = get_predicted_value(user_symptoms1)
print(predicted_disease)
def helper(dis):

    pre = treatment[treatment['Disease'] == dis][['Treatment 1', 'Treatment 2', 'Treatment 3']]
    pre = [col for col in pre.values]
    
    return pre
tret = helper(predicted_disease)
i = 1
print("=================Treatment==================")
for t in tret:
    print(i, ": ", t)
    i += 1