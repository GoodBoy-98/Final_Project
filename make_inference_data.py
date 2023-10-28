import pandas as pd 

# inference 대상 3명 분 만드는 예제 샘플 
data = {'FormatedTime':[18,18,18], 'Day':['수요일', '목요일', '일요일'], 'Sex':['남성', '남성', '남성'],'Age':['20대','20대','20대'], 'Residence':['세종시','세종시','세종시']}
df = pd.DataFrame.from_dict(data=data, orient='columns')
df.to_csv("inference_data.csv", index=False, encoding="cp949")