import streamlit as st
import pickle
import pandas as pd
import numpy as np
st.set_page_config(page_title="ASD DETECTION")
with open('dataframe1.pkl','rb') as file:
    df = pickle.load(file)

with open('module_1_pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

age_category=st.selectbox('SELECT AGE CATEGORY',['TODDLER(1-3 YEARS)','CHILDREN(4-11 YEARS)','TEENS(12-17 YEARS)','ADULTS(18 YEARS AND OLDER)'])
if age_category=='TODDLER(1-3 YEARS)':
    pass
if age_category=='CHILDREN(4-11 YEARS)':

    st.header('Complete the form by answering the following questions')
    #score
    #age
    age=st.selectbox('SELECT AGE(YEARS)',['4','5','6','7','8','9','10','11'])

    #gender
    gender_selectbox=st.selectbox('SELECT GENDER',['MALE','FEMALE'])
    gender='m' if gender_selectbox=='MALE' else 'f'

    #Jaundice
    jaundice_selectbox=st.selectbox('BORN WITH JAUNDICE',['YES','NO'])
    Jaundice='yes' if jaundice_selectbox=='YES' else 'no'

    #country
    country = st.selectbox('COUNTRY OF RESIDENCE', sorted(df['country'].unique().tolist()))


    #relation
    relation=st.selectbox('WHO IS COMPLETING THE TEST',sorted(df['relation'].unique().tolist()))

    #A1
    A1_selectbox = st.selectbox('Q1-SHE/HE OFTEN NOTICES SMALL SOUNDS WHEN OTHERS DO NOT',['YES','NO'])
    A1 = 1 if A1_selectbox == 'YES' else 0


    #A2
    A2_selectbox = st.selectbox('Q2-SHE/HE USUALLY CONCENTRATES MORE ON THE WHOLE PICTURE,RATHER THAN SMALL DETAILS',['YES','NO'])
    A2 = 1 if A2_selectbox == 'NO' else 0


    #A3
    A3_selectbox = st.selectbox('Q3-IN A SOCIAL GROUP, SHE/HE CAN EASILY KEEP TRACK OF SEVERAL DIFFERENT PEOPLE’S CONVERSATIONS',['YES','NO'])
    A3 = 1 if A3_selectbox == 'NO' else 0


    #A4
    A4_selectbox = st.selectbox('Q4-SHE/HE FINDS IT EASY TO GO BACK AND FORTH BETWEEN DIFFERENT ACTIVITIES',['YES','NO'])
    A4 = 1 if A4_selectbox == 'NO' else 0

    #A5
    A5_selectbox = st.selectbox('Q5-SHE/HE DOESN’T KNOW HOW TO KEEP A CONVERSATION GOING WITH HIS/HER PEERS ',['YES','NO'])
    A5 = 1 if A5_selectbox == 'YES' else 0

    #A6
    A6_selectbox = st.selectbox('Q6-SHE/HE IS GOOD AT SOCIAL CHIT-CHAT ',['YES','NO'])
    A6 = 1 if A6_selectbox == 'NO' else 0


    #A7
    A7_selectbox = st.selectbox('Q7-WHEN SHE/HE IS READ A STORY, SHE/HE FINDS IT DIFFICULT TO WORK OUT THE CHARACTER’S INTENTIONS OR FEELINGS ',['YES','NO'])
    A7 = 1 if A7_selectbox == 'YES' else 0


    #A8
    A8_selectbox = st.selectbox('Q8-WHEN SHE/HE WAS IN PRESCHOOL, SHE/HE USED TO ENJOY PLAYING GAMES INVOLVING PRETENDING WITH OTHER CHILDREN ',['YES','NO'])
    A8 = 1 if A8_selectbox == 'NO' else 0


    #A9
    A9_selectbox = st.selectbox('Q9-SHE/HE FINDS IT EASY TO WORK OUT WHAT SOMEONE IS THINKING OR FEELING JUST BY LOOKING AT THEIR FACE ',['YES','NO'])
    A9 = 1 if A9_selectbox == 'NO' else 0


    #A10
    A10_selectbox = st.selectbox('Q10-SHE/HE FINDS IT HARD TO MAKE NEW FRIENDS',['YES','NO'])
    A10 = 1 if A10_selectbox == 'YES' else 0


    #score
    score=A1+A2+A3+A4+A5+A6+A7+A8+A9+A10


    #PREDICTION
    if st.button('Predict ASD'):
        # form a dataframe
        data=[[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,age,gender,Jaundice,country,score,relation]]
        columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','age','gender','Jaundice','country','score','relation']
        # Convert to DataFrame
        one_df = pd.DataFrame(data, columns=columns)
        # predict
        prediction=pipeline.predict(one_df)[0]
        if prediction=='YES':
            st.text("The model predicts that the child has Autism Spectrum Disorder (ASD).")
        else:
            st.text("The model predicts that the child does not have Autism Spectrum Disorder (ASD).")

if age_category=='ADULTS(18 YEARS AND OLDER)':
    with open('dataframe1_adult.pkl','rb') as file:
        df1 = pickle.load(file)

    with open('module_1_pipeline_adult.pkl','rb') as file:
        pipeline2 = pickle.load(file)
        # 'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','result'
    st.header('Complete the form by answering the following questions')
    #score
    #age
    import streamlit as st

    # Define the input widget
    age = st.number_input(label="ENTER THE AGE", min_value=18, max_value=70, step=1)

    # Validate the input
    if age % 1 != 0:
        st.error("Please enter a valid age")
    elif age > 70 or age <18 :
        st.error("Age must be between 18 and 100")
    else:
        st.success(" Age entered: {}".format(age))

    #gender
    gender_selectbox=st.selectbox('SELECT GENDER',['MALE','FEMALE'])
    gender='m' if gender_selectbox=='MALE' else 'f'

    #Jaundice
    jaundice_selectbox=st.selectbox('BORN WITH JAUNDICE',['YES','NO'])
    jaundice='yes' if jaundice_selectbox=='YES' else 'no'

    #country
    country_of_res = st.selectbox('COUNTRY OF RESIDENCE', sorted(df1['country_of_res'].unique().tolist()))


    #relation
    relation=st.selectbox('WHO IS COMPLETING THE TEST',sorted(df1['relation'].unique().tolist()))

    #A1
    A1_selectbox1 = st.selectbox('Q1-I often notice small sounds when others do not',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A1_Score = 1 if A1_selectbox1 == 'YES' else 0


    #A2
    A2_selectbox1 = st.selectbox('Q2-SHE/HE USUALLY CONCENTRATES MORE ON THE WHOLE PICTURE,RATHER THAN SMALL DETAILS',['YES','NO'])
    A2_Score = 1 if A2_selectbox1 == 'YES' else 0


    #A3
    A3_selectbox1 = st.selectbox('Q3-IN A SOCIAL GROUP, SHE/HE CAN EASILY KEEP TRACK OF SEVERAL DIFFERENT PEOPLE’S CONVERSATIONS',['YES','NO'])
    A3_Score = 1 if A3_selectbox1 == 'YES' else 0


    #A4
    A4_selectbox1 = st.selectbox('Q4-SHE/HE FINDS IT EASY TO GO BACK AND FORTH BETWEEN DIFFERENT ACTIVITIES',['YES','NO'])
    A4_Score = 1 if A4_selectbox1 == 'YES' else 0

    #A5
    A5_selectbox1 = st.selectbox('Q5-SHE/HE DOESN’T KNOW HOW TO KEEP A CONVERSATION GOING WITH HIS/HER PEERS ',['YES','NO'])
    A5_Score = 1 if A5_selectbox1 == 'YES' else 0

    #A6
    A6_selectbox1 = st.selectbox('Q6-SHE/HE IS GOOD AT SOCIAL CHIT-CHAT ',['YES','NO'])
    A6_Score = 1 if A6_selectbox1 == 'YES' else 0


    #A7
    A7_selectbox1 = st.selectbox('Q7-WHEN SHE/HE IS READ A STORY, SHE/HE FINDS IT DIFFICULT TO WORK OUT THE CHARACTER’S INTENTIONS OR FEELINGS ',['YES','NO'])
    A7_Score = 1 if A7_selectbox1 == 'YES' else 0


    #A8
    A8_selectbox1 = st.selectbox('Q8-WHEN SHE/HE WAS IN PRESCHOOL, SHE/HE USED TO ENJOY PLAYING GAMES INVOLVING PRETENDING WITH OTHER CHILDREN ',['YES','NO'])
    A8_Score = 1 if A8_selectbox1 == 'YES' else 0


    #A9
    A9_selectbox1 = st.selectbox('Q9-SHE/HE FINDS IT EASY TO WORK OUT WHAT SOMEONE IS THINKING OR FEELING JUST BY LOOKING AT THEIR FACE ',['YES','NO'])
    A9_Score = 1 if A9_selectbox1 == 'YES' else 0


    #A10
    A10_selectbox1 = st.selectbox('Q10-SHE/HE FINDS IT HARD TO MAKE NEW FRIENDS',['YES','NO'])
    A10_Score = 1 if A10_selectbox1 == 'YES' else 0


    #score
    result=A1_Score+A2_Score+A3_Score+A4_Score+A5_Score+A6_Score+A7_Score+A8_Score+A9_Score+A10_Score


    #PREDICTION
    if st.button('Predict ASD'):
        # form a dataframe
        data=[[A1_Score,A2_Score,A3_Score,A4_Score,A5_Score,A6_Score,A7_Score,A8_Score,A9_Score,A10_Score,age,gender,jaundice,country_of_res,result,relation]]
        columns=['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','age','gender','jaundice','country_of_res','result','relation']
        # Convert to DataFrame
        one_df = pd.DataFrame(data, columns=columns)
        # predict
        prediction=pipeline2.predict(one_df)[0]
        if prediction=='YES':
            st.text("The model predicts that the child has Autism Spectrum Disorder (ASD).....")
        else:
            st.text("The model predicts that the child does not have Autism Spectrum Disorder (ASD)......")
