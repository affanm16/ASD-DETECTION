import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="ASD DETECTION")
st.header('DETECTION OF ASD USING MACHINE LEARNING', divider='rainbow')

import streamlit as st

with open('dataframe1.pkl','rb') as file:
    df = pickle.load(file)

with open('module_1_pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

age_category=st.selectbox('SELECT AGE CATEGORY',['TODDLER(1-3 YEARS)','CHILDREN(4-11 YEARS)','TEENS(12-17 YEARS)','ADULTS(18 YEARS AND OLDER)'])
if age_category=='CHILDREN(4-11 YEARS)':

    st.subheader('To proceed,complete the :green[Assessment Form] and provide relevant inputs', divider='violet')
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
    st.subheader(':green[AQ-10 (Child Version)]', divider=True)
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
    # Define custom CSS styles
    st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: #0099ff;  # Blue background
                color: #ffffff;  # White text color
            }
            div.stButton > button:hover {
                background-color: #BCF5E5;  # Green background on hover
                color: #E9F7F7;  # Red text color on hover
            }
        </style>
    """, unsafe_allow_html=True)

    if st.button('PREDICT'):
        # form a dataframe
        data=[[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,age,gender,Jaundice,country,score,relation]]
        columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','age','gender','Jaundice','country','score','relation']
        # Convert to DataFrame
        one_df = pd.DataFrame(data, columns=columns)
        # predict
        prediction=pipeline.predict(one_df)[0]
        if prediction=='YES':
            st.text("The model predicts that the child has Autism Spectrum Disorder (ASD).")
            st.info('The assessment results are indicative, not diagnostic, and should prompt individuals to seek comprehensive evaluation by qualified healthcare professionals for an accurate diagnosis of ASD.', icon="ℹ️")
        else:
            st.text("The model predicts that the child does not have Autism Spectrum Disorder (ASD).")
if age_category=='ADULTS(18 YEARS AND OLDER)':

    with open('dataframe1_adult.pkl','rb') as file:
        df1 = pickle.load(file)

    with open('module_1_pipeline_adult.pkl','rb') as file:
        pipeline2 = pickle.load(file)
        # 'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','result'
    st.subheader('To proceed,complete the :green[Assessment Form] and provide relevant inputs', divider='blue')
    import streamlit as st

    # Define the input widget
    age = st.number_input(label="ENTER THE AGE", min_value=18, max_value=70, step=1)

    # Validate the input
    if age % 1 != 0:
        st.error("Please enter a valid age")
    elif age > 70 or age <18 :
        st.error("Age must be between 18 and 70")
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
    st.subheader(':green[Autism Spectrum Quotient (AQ)-10]', divider=True)
    #A1
    A1_selectbox1 = st.selectbox('Q1-I often notice small sounds when others do not',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A1_Score = 1 if A1_selectbox1 == 'Definitely Agree' or 'Slightly Agree' else 0


    #A2
    A2_selectbox1 = st.selectbox('Q2-I usually concentrate more on the whole picture, rather than the small details ',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A2_Score = 1 if A2_selectbox1 == 'Slightly Disagree' or 'Definitely Disagree' else 0


    #A3
    A3_selectbox1 = st.selectbox('Q3-I find it easy to do more than one thing at once',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A3_Score = 1 if A3_selectbox1 == 'Slightly Disagree' or 'Definitely Disagree' else 0


    #A4
    A4_selectbox1 = st.selectbox('Q4-If there is an interruption, I can switch back to what I was doing very quickly',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A4_Score = 1 if A4_selectbox1 == 'Slightly Disagree' or 'Definitely Disagree' else 0

    #A5
    A5_selectbox1 = st.selectbox('Q5-I find it easy to ‘read between the lines’ when someone is talking to me ',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A5_Score = 1 if A5_selectbox1 == 'Slightly Disagree' or 'Definitely Disagree' else 0

    #A6
    A6_selectbox1 = st.selectbox('Q6-I know how to tell if someone listening to me is getting bored',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A6_Score = 1 if A6_selectbox1 == 'Slightly Disagree' or 'Definitely Disagree' else 0

    #A7
    A7_selectbox1 = st.selectbox('Q7-When I’m reading a story I find it difficult to work out the characters’ intentions ',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A7_Score = 1 if A7_selectbox1 == 'Definitely Agree' or 'Slightly Agree' else 0

    #A8
    A8_selectbox1 = st.selectbox('Q8-I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc) ',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A8_Score = 1 if A8_selectbox1 == 'Definitely Agree' or 'Slightly Agree' else 0

    #A9
    A9_selectbox1 = st.selectbox('Q9I find it easy to work out what someone is thinking or feeling just by looking at their face ',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A9_Score = 1 if A9_selectbox1 == 'Slightly Disagree' or 'Definitely Disagree' else 0

    #A10
    A10_selectbox1 = st.selectbox('Q10-I find it difficult to work out people’s intentions',['Definitely Agree','Slightly Agree', 'Slightly Disagree', 'Definitely Disagree' ])
    A10_Score = 1 if A10_selectbox1 == 'Definitely Agree' or 'Slightly Agree' else 0

    #score
    result=A1_Score+A2_Score+A3_Score+A4_Score+A5_Score+A6_Score+A7_Score+A8_Score+A9_Score+A10_Score
    st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: #0099ff;  # Blue background
                color: #ffffff;  # White text color
            }
            div.stButton > button:hover {
                background-color: #BCF5E5;  # Green background on hover
                color: #E9F7F7;  # Red text color on hover
            }
        </style>
    """, unsafe_allow_html=True)

    #PREDICTION
    if st.button('PREDICT'):
        # form a dataframe
        data=[[A1_Score,A2_Score,A3_Score,A4_Score,A5_Score,A6_Score,A7_Score,A8_Score,A9_Score,A10_Score,age,gender,jaundice,country_of_res,result,relation]]
        columns=['A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score','A7_Score','A8_Score','A9_Score','A10_Score','age','gender','jaundice','country_of_res','result','relation']
        # Convert to DataFrame
        one_df = pd.DataFrame(data, columns=columns)
        # predict
        prediction=pipeline2.predict(one_df)[0]
        if prediction=='YES':
            st.text("The model predicts that the child has Autism Spectrum Disorder (ASD).....")
            st.info('The assessment results are indicative, not diagnostic, and should prompt individuals to seek comprehensive evaluation by qualified healthcare professionals for an accurate diagnosis of ASD.', icon="ℹ️")
        else:
            st.text("The model predicts that the child does not have Autism Spectrum Disorder (ASD)......")
if age_category=='TEENS(12-17 YEARS)':

    with open('dataframe1_adolescents.pkl','rb') as file:
        df4 = pickle.load(file)

    with open('module_1_pipeline_adolescent.pkl','rb') as file:
        pipeline4 = pickle.load(file)
    
    st.subheader('To proceed,complete the :green[Assessment Form] and provide relevant inputs', divider='violet')
    #score
    #age
    import streamlit as st
    # Define the input widget
    age = st.number_input(label="ENTER THE AGE", min_value=12, max_value=17, step=1)

    # Validate the input
    if age % 1 != 0:
        st.error("Please enter a valid age")
    elif age > 17 or age <12 :
        st.error("Age must be between 12 and 17")
    else:
        st.success(" Age entered: {}".format(age))

    #gender
    gender_selectbox=st.selectbox('SELECT GENDER',['MALE','FEMALE'])
    Sex='0' if gender_selectbox=='MALE' else '1'

    #Jaundice
    jaundice_selectbox=st.selectbox('BORN WITH JAUNDICE',['YES','NO'])
    born_with_jaundice='1' if jaundice_selectbox=='YES' else '0'
    #FAMILY MEM WITH PDD
    family_mem_with_pdd=st.selectbox('FAMILY MEMBER WITH PDD',['YES','NO'])
    family_member_with_PDD='1' if family_mem_with_pdd=='YES' else 0
    
    ethinicity_dict={
    0: 'White-European',
    1: 'Black',
    2: 'White-European',
    3: 'Hispanic',
    4: 'Asian',
    5: 'Black',
    6: 'Hispanic',
    7: 'Latino',
    8: 'Middle Eastern',
    9: 'Others',
    10: 'South Asian',
    11: 'White-European',
    12:'other'
    }
    ethinicity_input=st.selectbox('ETHINICITY',sorted(list(ethinicity_dict.values())))
    label_ethnicity = None
    for key, value in ethinicity_dict.items():
        if value == ethinicity_input:
            label_ethnicity = key
            break
    
    # Store user input in the DataFrame
    dict_relation={
        
    0: 'Health care professional',
    1: 'Others',
    2: 'Parent',
    3: 'Relative',
    4: 'Self'
    }
    relation_input=st.selectbox('WHO IS COMPLETING THE TEST',sorted(list(dict_relation.values())))
    label_whos_completing_test = None
    for key, value in dict_relation.items():
        if value == relation_input:
            label_whos_completing_test = key
            break
    label_country=10
    #used app before
    used_before_selectbox=st.selectbox('USED APP BEFORE?',['YES','NO'])
    used_screening_app_before=1 if used_before_selectbox=='YES' else 0
    st.subheader(':green[AQ-10 (Adolescent Version]',divider=True)
    #A1
    A1_selectbox1 = st.selectbox('Q1-She/he notices patterns in things all the time ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q1_Score = 1 if A1_selectbox1 == 'Definitely Agree' or 'Slightly Agree' else 0


    #A2
    A2_selectbox1 = st.selectbox('Q2-S/he usually concentrates more on the whole picture, rather than the small details',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q2_Score = 1 if A2_selectbox1 == 'Slightly Disagree' or 'Definitely Disagree' else 0


    #A3
    A3_selectbox1 = st.selectbox('Q3-In a social group, s/he can easily keep track of several different people’s conversations  ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q3_Score = 1 if A3_selectbox1 =='Slightly Disagree' or 'Definitely Disagree' else 0



    #A4
    A4_selectbox1 = st.selectbox('Q4-If there is an interruption, s/he can switch back to what s/he was doing very quickly ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q4_Score = 1 if A4_selectbox1 =='Slightly Disagree' or 'Definitely Disagree' else 0

    #A5
    A5_selectbox1 = st.selectbox('Q5-S/he frequently finds that s/he doesn’t know how to keep a conversation going ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q5_Score= 1 if A5_selectbox1 == 'Definitely Agree' or 'Slightly Agree' else 0
    #A6
    A6_selectbox1 = st.selectbox('Q6-S/he is good at social chit-chat ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q6_Score = 1 if A6_selectbox1 =='Slightly Disagree' or 'Definitely Disagree' else 0

    #A7
    A7_selectbox1 = st.selectbox('Q7-When s/he was younger, s/he used to enjoy playing games involving pretending with other children ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q7_Score = 1 if A7_selectbox1 =='Slightly Disagree' or 'Definitely Disagree' else 0

    #A8
    A8_selectbox1 = st.selectbox('Q8-S/he finds it difficult to imagine what it would be like to be someone else ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q8_Score = 1 if A8_selectbox1 =='Definitely Agree' or 'Slightly Agree' else 0

    #A9
    A9_selectbox1 = st.selectbox('Q9-S/he finds social situations easy',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q9_Score = 1 if A9_selectbox1 =='Slightly Disagree' or 'Definitely Disagree' else 0

    #A10
    A10_selectbox1 = st.selectbox('Q10-S/he finds it hard to make new friends ',['Definitely Agree' ,'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree'])
    Q10_Score = 1 if A10_selectbox1 =='Definitely Agree' or 'Slightly Agree' else 0
    st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: #0099ff;  # Blue background
                color: #ffffff;  # White text color
            }
            div.stButton > button:hover {
                background-color: #BCF5E5;  # Green background on hover
                color: #E9F7F7;  # Red text color on hover
            }
        </style>
    """, unsafe_allow_html=True)

    #PREDICTION
    if st.button('Predict'):
        # age	gender	born_with_jaundice	family_member_with_PDD	label_whos_completing_test	label_ethnicity	label_country	used_screening_app_before	Q1_Score	Q2_Score	Q3_Score	Q4_Score	Q5_Score	Q6_Score	Q7_Score	Q8_Score	Q9_Score	Q10_Score	
        # form a dataframe
        data=[[age,Sex,born_with_jaundice,family_member_with_PDD,label_whos_completing_test,label_ethnicity,label_country,used_screening_app_before,Q1_Score,Q2_Score,Q3_Score,Q4_Score,Q5_Score,Q6_Score,Q7_Score,Q8_Score,Q9_Score,Q10_Score]]
        columns=['age','gender','born_with_jaundice','family_member_with_PDD','label_whos_completing_test','label_ethnicity','label_country','used_screening_app_before','Q1_Score','Q2_Score','Q3_Score','Q4_Score','Q5_Score','Q6_Score','Q7_Score','Q8_Score','Q9_Score','Q10_Score']
        # Convert to DataFrame
        one_df = pd.DataFrame(data, columns=columns)
        # predict
        prediction=pipeline4.predict(one_df)[0]
        if prediction==0:
            st.text("The model predicts that the child does not have Autism Spectrum Disorder (ASD)!!!")
        else:
            st.text("The model predicts that the child has Autism Spectrum Disorder (ASD)!!!")
            st.info('The assessment results are indicative, not diagnostic, and should prompt individuals to seek comprehensive evaluation by qualified healthcare professionals for an accurate diagnosis of ASD.', icon="ℹ️")

if age_category=='TODDLER(1-3 YEARS)':
    with open('dataframe1_toddler.pkl','rb') as file:
        df_toddler = pickle.load(file)

    with open('module_1_pipeline_toddler.pkl','rb') as file:
        pipeline5 = pickle.load(file)

    st.subheader('To proceed,complete the:green[Assessment Form] and provide relevant inputs', divider='blue')
    #score
    #age
    Age_Mons=st.selectbox('SELECT AGE(YEARS)',['1','2','3'])

    #gender
    gender_selectbox=st.selectbox('SELECT GENDER',['MALE','FEMALE'])
    Sex='m' if gender_selectbox=='MALE' else 'f'

    #Jaundice
    jaundice_selectbox=st.selectbox('BORN WITH JAUNDICE',['YES','NO'])
    Jaundice='yes' if jaundice_selectbox=='YES' else 'no'

    #country
    Ethnicity='middle eastern'


    #relation
    who_completed_the_test=st.selectbox('WHO IS COMPLETING THE TEST',sorted(df_toddler['Who completed the test'].unique().tolist()))

    #family_asd
    Family_mem_with_ASD='no'
    st.subheader(':green[Q-CHAT-10- Quantitative Checklist for Autism in Toddlers]',divider=True)
    #A1
    A1_selectbox = st.selectbox('Q1-Does your child look at you when you call his/her name?',['Always','Usually','Sometimes','Rarely','Never'] )
    A1 = 1 if A1_selectbox == 'Sometimes'or 'Rarely' or 'Never' else 0


    #A2
    A2_selectbox = st.selectbox('Q2-How easy is it for you to get eye contact with your child?',['Very easy','Quite easy','Quite difficult','Very difficult','Impossible'])
    A2 = 1 if A2_selectbox =='Quite difficult' or 'Very difficult'or 'Impossible'  else 0


    #A3
    A3_selectbox = st.selectbox('Q3-Does your child point to indicate that s/he wants something? (e.g. a toy that is ]out of reach) ',['Many times a day',' A few times a day', 'A few times a week', 'Less than once a week','Never' ])
    A3 = 1 if A3_selectbox =='A few times a week' or 'Less than once a week' or 'Never' else 0


    #A4
    A4_selectbox = st.selectbox('Q4-Does your child point to share interest with you? (e.g. pointing at an interesting sight)  ',['Many times a day',' A few times a day', 'A few times a week', 'Less than once a week','Never' ])
    A4 = 1 if A4_selectbox == 'A few times a week' or 'Less than once a week' or 'Never' else 0

    #A5
    A5_selectbox = st.selectbox('Q5-Does your child pretend? (e.g. care for dolls, talk on a toy phone)',['Many times a day',' A few times a day', 'A few times a week', 'Less than once a week','Never' ])
    A5 = 1 if A5_selectbox =='A few times a week' or 'Less than once a week' or 'Never' else 0

    #A6
    A6_selectbox = st.selectbox('Q6-Does your child follow where you’re looking?  ',['Many times a day',' A few times a day', 'A few times a week', 'Less than once a week','Never' ])
    A6 = 1 if A6_selectbox =='A few times a week' or 'Less than once a week' or 'Never' else 0


    #A7
    A7_selectbox = st.selectbox('Q7-If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?  (e.g. stroking hair, hugging them)  ',['Always', 'Usually', 'Sometimes' , 'Rarely', 'Never']) 
    A7 = 1 if A7_selectbox == 'Sometimes' or 'Rarely' or 'Never'else 0


    #A8
    A8_selectbox = st.selectbox('Q8-Would you describe your child’s first words as:',['Very typical','Quite typical','Slightly unusual','Very unusual', 'My child doesn’t speak' ])
    A8 = 1 if A8_selectbox =='Slightly unusual'or'Very unusual' or'My child doesn’t speak' else 0


    #A9
    A9_selectbox = st.selectbox('Q9-Does your child use simple gestures?  (e.g. wave goodbye)',['Many times a day',' A few times a day', 'A few times a week', 'Less than once a week','Never' ])
    A9 = 1 if A9_selectbox == 'A few times a week' or 'Less than once a week' or 'Never' else 0


    #A10
    A10_selectbox = st.selectbox('Q10-Does your child stare at nothing with no apparent purpose?',['Many times a day',' A few times a day', 'A few times a week', 'Less than once a week','Never' ])
    A10 = 1 if A10_selectbox == 'Many times a day'or' A few times a day'or 'A few times a week' else 0


    #score
    Qchat_10_score=A1+A2+A3+A4+A5+A6+A7+A8+A9+A10
    st.markdown("""
        <style>
            div.stButton > button:first-child {
                background-color: #0099ff;  # Blue background
                color: #ffffff;  # White text color
            }
            div.stButton > button:hover {
                background-color: #BCF5E5;  # Green background on hover
                color: #E9F7F7;  # Red text color on hover
            }
        </style>
    """, unsafe_allow_html=True)

    #PREDICTION
    if st.button('PREDICT'):
        # form a dataframe
        data=[[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,Age_Mons,Qchat_10_score,Sex,Ethnicity,Jaundice,Family_mem_with_ASD,who_completed_the_test]]
        columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','Age_Mons','Qchat-10-Score','Sex','Ethnicity','Jaundice','Family_mem_with_ASD','Who completed the test']
        # Convert to DataFrame
        one_df = pd.DataFrame(data, columns=columns)
        # predict
        prediction=pipeline5.predict(one_df)[0]
        if Qchat_10_score>3:
            st.text("The model predicts that the child has Autism Spectrum Disorder (ASD).")
            st.info('The assessment results are indicative, not diagnostic, and should prompt individuals to seek comprehensive evaluation by qualified healthcare professionals for an accurate diagnosis of ASD.', icon="ℹ️")
        else:
            st.text("The model predicts that the child does not have Autism Spectrum Disorder (ASD).")