import pymysql
import pandas as pd 
import numpy as np # 고수학 연산을 위해 임포트
import argparse 
import json 
import os 
# 시각화를 위한 패키지 임포트
#import matplotlib.pyplot as plt
#import seaborn as sns
import joblib
# 전처리 라이브러리
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 학습을 위한 라이브러리 임포트
from sklearn.linear_model import LogisticRegression #Logistic(Regression)Classifier
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.svm import SVC #Support Vector Machine
from sklearn.naive_bayes import GaussianNB #Naive Bayesian
from sklearn.neighbors import KNeighborsClassifier #K Nearest Neighbor
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosing
from sklearn.neural_network import MLPClassifier #Neural Network

from sklearn.metrics import accuracy_score
from sklearn import model_selection
###############
# 전역변수 
MODEL_PATH = './model/'
SEX = ''
AGE = ''
RES = '' 
TOP_N = 5 
###############
def get_df_from_sql():
    conn = pymysql.connect(
    host='login-lecture.cxh9jd3ejp84.ap-northeast-2.rds.amazonaws.com',
    port=int(3306),
    user="admin",
    passwd="jkr123^^7",
    db="login_lecture",
    charset='utf8')
    cursor=conn.cursor()

    sql1 = 'select * from Member'
    cursor.execute(sql1)
    result1 = cursor.fetchall()

    sql2 = 'select * from Order_test'
    cursor.execute(sql2)
    result2 = cursor.fetchall()

    member = pd.DataFrame(result1)
    food = pd.DataFrame(result2)

    member.columns = ['Member_Key', 'Id', 'Pw', 'Name', 'Age', 'Sex', 'Residence', 'Adress']
    food.columns = ['OrderNo', 'Status', 'Store', 'Type_of_Food', 'Menu', 'Amount', 'Rating', 'Date', 'FormatedTime', 'Day', 'Member_Key']
    merge_inner = pd.merge(member, food)
    interested_df = merge_inner[['Member_Key', 'Age', 'Sex', 'Residence', 'Store', 'Type_of_Food', 'Menu', 'Rating', 'FormatedTime', 'Day']]
    dataset_NaN_deleted = interested_df.dropna()
    return dataset_NaN_deleted.copy() 

# json serialize error 방지를 위한 클래스 
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)
        
def train_preprocess(data): 
    # select column 
    X = data[['Rating', 'FormatedTime', 'Day']]
    Y = data[['Menu']]

    # categorical encoding x, y 
    for col in X.columns:
        xle = LabelEncoder()
        X[col] = xle.fit_transform(X[col])
        joblib.dump(xle, MODEL_PATH + f'{SEX}_{col}_x_encoder.pkl')

    yle = LabelEncoder()
    Y['Menu'] = yle.fit_transform(Y['Menu'])

    yle_name_mapping = dict(zip(yle.classes_, yle.transform(yle.classes_)))
    print('========================================')
    print('\n ===== y label encoder mapping: ===== \n', yle_name_mapping, '\n')
    # Y 라벨 맵 저장 
    with open(MODEL_PATH + f'{SEX}_label_map.json','w', encoding='utf-8') as f: 
        json.dump(yle_name_mapping, f, indent="\t", ensure_ascii=False, cls=NumpyArrayEncoder) #ensure_ascii=False, 

    # minmax scaler for x 
    mmscaler = MinMaxScaler()
    X = mmscaler.fit_transform(X)
    joblib.dump(mmscaler, MODEL_PATH + f'{SEX}_x_scaler.pkl')
    return X, Y 

def inference_preprocess(data):
    # select column
    X = data[['FormatedTime', 'Day']]
    # load encoder & fit 
    for col in X.columns:
        try: 
            xle = joblib.load(MODEL_PATH + f'{SEX}_{col}_x_encoder.pkl')
        except: 
            raise ValueError(MODEL_PATH + f'{SEX}_{col}_x_encoder.pkl 가 존재하지 않습니다. 학습부터 진행하세요.')
        X[col] = xle.fit_transform(X[col])
    # load scaler & fit 
    try:
        mmscaler = joblib.load(MODEL_PATH + f'{SEX}_x_scaler.pkl')
    except: 
        raise ValueError(f'{SEX}_x_scaler.pkl 가 존재하지 않습니다. 학습부터 진행하세요.')
    X = mmscaler.fit_transform(X)
    
    # inference 때는 rating 컬럼은 없을 것이므로, 최대 rating으로 (scaled 에선 1) 할당 (높은게 좋은 추천이므로)
    X = np.insert(X, 0, 1, axis=1) # 0번째 (Rating) 축에 1이라는 constant value 추가 

    return X 

def train_best_model(X, Y): 
    # 모델 
    models = [] 
    models.append(("LR", LogisticRegression()))
    models.append(("DT", DecisionTreeClassifier()))
    models.append(("SVM", SVC()))
    models.append(("NB", GaussianNB()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("RF", RandomForestClassifier()))
    models.append(("GB", GradientBoostingClassifier()))
    models.append(("ANN", MLPClassifier()))
    print('========================================')
    print('===== Models Comparison =====')
    # Train 
    fit_models = [] 
    for name, model in models: # no validation 
        print(f'===== Start training {name} =====')
        model.fit(X, Y.values.ravel()) # 학습 끝
        y_pred = model.predict(X)
        acc =  accuracy_score(Y, y_pred)
        print(f'>> Train acc.: ', acc)
        fit_models.append((name, model))
    # CV교차검증
    results = []
    names = [] 
    for name, model in fit_models:
        print(f"\n ===== Start CrossValidation for model << {name} >> =====")
        kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X, Y.values.ravel(), cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
    # # visualize CV results
    # fig = plt.figure()
    # fig.suptitle('Classifier Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.show()

    # Select best model & save as pkl file
    acc_list = np.mean(results, axis=1)
    best_acc_idx = np.argsort(acc_list)[-1]
    best_model_name = names[best_acc_idx] #[names[i] for i in top3_acc_idx]
    best_model = [m for (n, m) in models if n == best_model_name][0]
    # 모델 저장 
    joblib.dump(best_model, MODEL_PATH + f'{SEX}_{AGE}_{RES}_best_model.pkl')

def inference(X):
    try: 
        best_model = joblib.load(MODEL_PATH + f'{SEX}_{AGE}_{RES}_best_model.pkl')
    except: 
        raise ValueError(f'{SEX}_{AGE}_{RES}_best_model.pkl 가 존재하지 않습니다. 학습부터 진행하세요.')

    inference_result = None 
    try: 
        probs = best_model.predict_proba(X)
        inference_result = np.argsort(probs, axis=1)[:,  -TOP_N:]
    except: 
        raise ValueError('Failed to inference.')
    return inference_result

def make_data_per_sex(args, df):
    if args.sex == '남성':
        data = df[(df['Sex'] == '남성') & (df['Age'] == args.age) & (df['Residence'] == args.residence)]
    elif args.sex == '여성':
        data = df[(df['Sex'] == '여성') & (df['Age'] == args.age) & (df['Residence'] == args.residence)]
    else: 
        raise ValueError("--sex 인자를 잘못 입력했습니다. << 남성 >> 혹은 << 여성 >>으로 입력해주세요.")
    return data 

if __name__ == "__main__": 
    try: 
        os.makedirs(MODEL_PATH, exist_ok=True)
    except: 
        raise ValueError("Failed to create << model >> directory.")
    # argparse 
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--mode', required=True, help='train or inference')
    parser.add_argument('--sex', required=True, help='남성 or 여성')
    parser.add_argument('--residence', required=True, help='residence')
    parser.add_argument('--age', required=True, help='age')
    # 사용 예) python main.py --mode train --sex 남성 --residence 세종시 --age 20대 
    args = parser.parse_args()
    # 전역 변수화 
    SEX = args.sex 
    AGE = args.age
    RES = args.residence
    # 밑에꺼는 고정값일때 이렇게 쓰는거임 고정값으로 쓰려면 위에 주석처리 추가하고 --mode train 까지만 해야함
    #SEX = '남성'
    #AGE = '20대'
    #RES = '세종시'
    # 학습 or 추론 
    if args.mode == 'train': 
        # sql 연결 및 data 가져오기 
        df = get_df_from_sql()
        # 사용할 데이터 parsing 
        data = make_data_per_sex(args, df)
        # 전처리 후 학습 
        X, Y = train_preprocess(data)
        train_best_model(X, Y)

    elif args.mode == 'inference':
        data = pd.read_csv('inference_data.csv', encoding='cp949')
        X = inference_preprocess(data)
        inference_result = inference(X)
        print('추론 결과: ')
        print('==============')
        print('top 5  4  3  2  1')
        print(f'{inference_result}')
        print('==============')
        print(f'결과 라벨 맵핑은 model/{SEX}_label_map.json 파일을 참고')
        # label_map.json 매칭 

    else: 
        raise ValueError("--mode 인자를 잘못 입력했습니다. << train >> 혹은 << inference >>으로 입력해주세요.")