# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:48:59 2020

@author: Preeti
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from plotly.subplots import make_subplots



def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Confusion matrix")
        plot_confusion_matrix(model,xtest,ytest)
        st.pyplot()
    if "ROC" in metrics_list:
        st.subheader("ROC")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plot_roc_curve(model,xtest,ytest)
        st.pyplot(width=5)
    if "Precision Recall Curve" in metrics_list:
        st.subheader("Precision Recall Curve")
        plot_precision_recall_curve(model,xtest,ytest)
        st.pyplot()


df = pd.read_csv("dataset/finalll.csv")

courses = pd.read_csv("dataset/courses.csv")
assessments = pd.read_csv("dataset/assessments.csv")
studentAssessment = pd.read_csv("dataset/studentAssessment.csv")
studentInfo = pd.read_csv("dataset/studentInfo.csv")
studentRegistration = pd.read_csv("dataset/studentRegistration.csv")
vle = pd.read_csv("dataset/vle.csv")


class_name = ["final_result"]

st.title('Learning Analytics ')
st.write('\n')  
st.write('\n')

slct = st.sidebar.selectbox(" ",("Home","Show Dataset","Data Visualization","Classification Model","Model Comparison"))

if slct == "Home":
    st.header("About the Project")
    st.write("Don't you wish to get better grades in university? Every student has a strategy towards studying for their modules. Some methods include cramming or spacing out the revision of topics covered. This project aims towards studying the patterns of top students in a university and reconfirm existing tips that can help improve learning and score better grades…….")
    
    st.header("What is the OULA dataset about ?")
    st.write("Open University Learning Analytics dataset contains information about 22 courses, 32,593 students, their assessment results, and logs of their interactions with the VLE represented by daily summaries of student clicks (10,655,280 entries). The dataset is divided into 7 csv files. ")
        
    st.header("Why do we choose this dataset ?")
    st.write("We always thought about how we could improve the way we study and get better grades in university. There are many books and online courses out there that share different methods of studying and how to better retain information. Techniques such as spaced repetition and active recall leads to better long-term learning while cramming, although highly effective for tests/exams, results in faster forgetting. ")
    st.write("Our goal is to find out whether consistency in work will result in better grades. What are some of the habits that successful students have that allow them to get the grades they want? Are they just plain smarter? What strategy do they use when it comes to studying for an important exam like finals? Do they do anything different from average students? ")
    st.write("The Open University currently collects similar data on an on-going basis as input to algorithms they developed to identify students at risk for failing a course. Identification of at-risk students then triggers automated intervention measures to encourage behavior that would create success. For example, the algorithm might identify a student with low grades on intermediate assessments (quizzes). That student may be sent an automated email reminder about available tutoring options. The goal of the data collection effort is to maximize student success, which has numerous benefits for the University. ")
    st.write("This subset of anonymized data was made available to the public for educational purposes on Machine Learning approaches. ")
    
    st.header("Purpose of Our Project")
    st.write("For the purpose of this project, the data will be used to determine if socio-economic and/or behavior-based data can be used to predict a student's performance in a course. Performance is determined by the final result of the student’s effort and is characterized by completing the course with a passing score, either with or without Distinction. ")
    
    st.header("Specific Questions of Interest")
    st.write("● Can we predict a student's final status in a course based on socio-economic factors and/or patterns of interaction with the VLE?")
    st.write("● Desired Targets:\n 1. Prediction of student pass/ no pass the course after course completion (goal: 90% accuracy)\n 2. Prediction of student pass/ no pass the course after 30 days of commencement (goal: 75% accuracy) ")

    

if slct == "Show Dataset":
    st.header("Student Data")
    st.write(df)
    
    st.write('\n\n')
   
if slct == "Data Visualization":
    
    print("")
    #courses------------------> courses
    st.header("Courses")
    st.write("")
    st.write("")
    st.subheader("Each course has its code_module code, which is a unique identifier and a code_presentation presentation. The combination of the two makes a single entry.This table also contains information on the duration of each of the presentations.")
    st.write(courses)
    st.write("")
    st.write("")
    courses['presentation_year'] = courses['code_presentation'].str.slice(stop=-1)
    courses['presentation_month'] = courses['code_presentation'].str.slice(start=-1)
    st.subheader("Each course may have 2 presentations in a year i.e. B and J. B presentations are usually different from J presentations. It is good to analyze the types separately.")
    st.write(courses)
    st.write("")
    st.write("")
    st.subheader("We have two groupings of days. This can be related to the year or the period in which each presentation takes place.")
    fig = px.histogram(courses, x="module_presentation_length")
    fig.update_yaxes(title='No.of modules')
    st.plotly_chart(fig)

    st.write("")
    st.write("")
    st.subheader("As we can see, the start month shows a clear separation between presentations that last up to ~ 245 days and those that last from ~ 260 days.")
    fig = px.scatter(courses, x="module_presentation_length", y="presentation_month",color="presentation_month")
    st.plotly_chart(fig)
    
    
    #assessments------------------------>assessment
    st.header("Assessment")
    st.subheader("The combination of code_module and code_presentation determines a single course and must also exist in the courses file.There should not be two or more unique id_assessments.")
    st.subheader("Only 3 types of assessment: TMA (assessed by monitor), CMA (assessed by computer) and Exam (final exam).")
    st.subheader("The evaluation date is counted in days from the beginning of the course.")
    
    st.write("")
    st.write(assessments.head())
    st.write("")
    st.write("")
    
    st.subheader("Sum of weights is 100% for all types of assessment except tests.")
    st.subheader("All courses are 100% by weight, except GGG courses, which may be a policy of the course itself.")
    st.write("")
    st.write("")
    st.write(assessments.loc[assessments['assessment_type'].isin(['TMA', 'CMA'])].groupby(['code_module', 'code_presentation']).agg({'weight':'sum'}))
    st.write("")
    st.write("")
    
    st.subheader("How many final exams in each course ?")
    st.subheader("CCC-type courses have two final exams, which does not seem to be usual.")
    st.write("")
    st.write(assessments.loc[assessments['assessment_type'] == 'Exam'].groupby(['code_module', 'code_presentation']).agg({'assessment_type':'count'}))
    st.write("")
    st.write("")
    st.subheader("How is the distribution of the number of extra activities (CMA and TMA) per offer?")
    st.write(assessments.groupby(['code_module','code_presentation', 'assessment_type']).agg({'assessment_type':'count'}))
    st.subheader("We can see that in the BBB and DDD courses there were changes in the number of activities applied in different course presentations.")
    st.write("")
    st.write("")

    #st.header("")
    #st.write("")
    #st.write("")
    #st.subheader("")
    #st.write("")
    #st.write("")
    #st.plotly_chart()
    
    
    
    bbb_index = list(assessments.loc[assessments['code_module'] == 'BBB'].loc[assessments['code_presentation'] == '2014J'].index)
    ddd_index = list(assessments.loc[assessments['code_module'] == 'DDD'].loc[assessments['code_presentation'] == '2013B'].index)
    assessments_ = assessments.drop(bbb_index+ddd_index)
    
    st.write("")
    st.write("")
    st.header("Distribution of the number of extra activities (CMA and TMA) per offer")
    st.write("")
    st.write("")
    extra_counts = assessments.loc[assessments['assessment_type'] != 'Exam'].groupby(['code_module', 'code_presentation']).agg({'id_assessment':'count'})
    fig = plt.hist(extra_counts['id_assessment'])    
    fig = px.histogram(extra_counts, x='id_assessment') 
    fig.update_xaxes(title='Number of activities')
    fig.update_yaxes(title= 'Number of courses')
    st.plotly_chart(fig)
    
    #st.write(extra_counts.id_assessment.describe())
    st.subheader("We see that few courses have more than 10 extra activities, and on average courses have 8.27 activities.")
    
    st.write("")
    _ = assessments.loc[assessments['assessment_type'] != 'Exam']
    fig = px.histogram(_ , x='weight')
    fig.update_xaxes(title='Activity weight')
    fig.update_yaxes(title= 'Number of activities')
    st.plotly_chart(fig)
    st.subheader("We see that there are a large number of activities with a weight of 0, corresponding to the activities of the GGG course, which gives weight 0 to everything except the final test. In addition, we have a large number of activities with a weight of 20% to 35%, and no activity with a weight between 36 percent and 100%. Which indicates that many courses have a large number of extra activities, agreeing with the average number of activities that we saw earlier.")

    st.write("")
    st.write("")
    st.header("Distribution of weights when we group by type of activity")
    st.write("")
    st.write("")
    _ = assessments.loc[assessments['assessment_type'] != 'Exam'].loc[assessments['assessment_type'] == 'CMA']
    fig = px.histogram(_ , x='weight')
    fig.update_xaxes(title='Activity weight(CMA)')
    fig.update_yaxes(title= 'Number of activities')
    st.plotly_chart(fig)  
    st.write("")
    st.write("")
    _ = assessments.loc[assessments['assessment_type'] != 'Exam'].loc[assessments['assessment_type'] == 'TMA']
    fig = px.histogram(_ , x='weight')
    fig.update_xaxes(title='Activity weight(TMA)')
    fig.update_yaxes(title= 'Number of activities')
    st.plotly_chart(fig)     
    st.subheader("We see that activities corrected by monitors (TMA) carry more weight. Thus, it is possible to interpret CMAs as minor activities, as practical work, and TMAs as course projects, for example")

    st.write("")
    st.write("")
    st.header("Chances are of having an activity on each day of any course")
    st.write("")
    st.write("")
    _ = assessments.dropna().loc[assessments['assessment_type'] != 'Exam']
    fig = px.histogram(_ , x='date')
    
    fig.update_xaxes(title='Activity probability')
    fig.update_yaxes(title= 'Course Day')
    st.plotly_chart(fig)
    st.subheader("We see that there are peaks along the course in which activities are most likely. These peaks can coincide with the end of modules within a course, for example. The fastest activity of the courses expires on the 12th, which may correspond to an activity of introduction to the content. In addition, a hiatus draws attention around the day 160 ~ 170 when there are no activities. This can be explained as a period that students have to prepare for the final activities of the course. This graph could be improved by normalizing the day of each activity using the duration of each offer. Thus, we would no longer have absolute days and would have a percentage of course completion. This could even show characteristics that are independent of the month of offer")

    _ = assessments.set_index (['code_module', 'code_presentation']).merge (courses.set_index(['code_module', 'code_presentation']), left_index = True, right_index = True)
    _ = _.reset_index ()
    _ = _.loc [_ ['assessment_type'] != 'Exam']
    _ ['relative_date'] = _ ['date'] / _['module_presentation_length']
    fig = px.histogram(_ , x='relative_date')
    fig.update_xaxes(title='Course progress')
    fig.update_yaxes(title= 'Activity probability')
    st.plotly_chart(fig)    
    st.subheader("In this new version, which takes into account the relative progress of the course, we see that less activities are due by 100% of the course, leaving this date exclusively for tests. In addition, we see more pronounced peaks.")

#############-------------------->vle vle
    st.write("")
    st.write("")
    st.header("VLE")
    st.write("")
    st.write("")
    st.subheader("The csv file contains information about the available materials in the VLE. Typically these are html pages, pdf files, etc. Students have access to these materials online and their interactions with the materials are recorded. The vle.")
    st.write("")
    st.write("")
    st.write(vle.head())
        
    st.write("")
    st.write("")
    st.subheader("Can a material be associated with more than one course")
    st.write("")
    st.write("")
    st.write(vle.groupby(['id_site', 'code_module']).agg({'code_module':'count'}).code_module.value_counts())
    st.write(np.unique(vle.id_site.value_counts().values))
    st.subheader("We see that there is only a value of 1 in the count of distinct modules associated with material ids. Thus, a material is unique. ")
    
    st.write("")
    st.write("")
    st.header("The most common types of materials")
    st.write("")
    st.write("")
    _ = vle.groupby('activity_type').agg({'activity_type': 'count'}).to_dict()
    _ = _['activity_type']
    types = _.keys()
    counts = _.values()
    fig = px.bar(types,counts)

    fig.update_xaxes (title='Count')
    fig.update_yaxes (title='Material type')
    st.plotly_chart(fig)
    st.write(len (vle))
    st.write("")
    st.subheader("We see that two of the most used resources seem to be generic names: oucontent and resource. We interpret oucontent as external content and resource as an attached file. In addition to these two, we have subpages as one of the most referenced items, which We believe are pages for other content within the same course, and URLs as links to external sites.")
    st.write("")
    st.write("")
    
    
    ###########student Info
    st.write("")
    st.write("")
    st.header("StudentInfo")
    st.write("")
    st.subheader("This file contains demographic information about the students together with their results. ")
    st.write("")
    st.write("")
    st.write(studentInfo.head())
    st.write("")
    st.subheader("This table contains student data for the various course offerings.")
    st.write("")
    st.subheader("A single entry is defined by a triple (``ìd_student,code_module,code_presentation```)")
    st.subheader("The students appear to be all from England and neighboring countries, the regions all appear to be in the UK.That is why we have the IMD measure, which is a \"poverty index\" used in these countries.The graduation levels also seem to be the ones used there, We interpret it as follows:*Post Graduate Qualification: Post Graduate *HE Qualification: Graduation / Bachelor *A Level or Equivalent: High School or Equivalent Lower Than A Level: Incomplete High School *No Formal quals: No training.   We have information about a student's previous attempts in the same module and the number of credits completed. We don't know how many credits each module is worth, but that same table can show that information. There are students with some type of disability")
    
    st.write("")
    st.write("")
    st.subheader("Number of students in the data")
    st.write(len(np.unique(studentInfo['id_student'])))
    
    st.write("")
    st.header("Average number of modules taken by students")
    st.write("")
    st.write("")
    st.write(studentInfo.groupby('id_student').agg({'code_module':'nunique'}).describe())
    st.write("")
    st.subheader("The average number of different modules is 1,086, which indicates that the vast majority of students take only one module. However, the maximum number of modules is 3, but few students reach that number. When we look at the number of entries in the table and the number of students, we have about 10,000 more entries than the number of students. Since few students take more than one module, We attribute that number of entries to more than multiple attempts. We will investigate this later.")

    st.write("")
    st.write("")
    st.subheader("Gender Distribution")
    #st.bar_chart(studentInfo['gender'].value_counts())
    fig = px.bar(studentInfo['gender'].value_counts())
    fig.update_yaxes(title='No. of students')
    fig.update_xaxes(title='Gender')
    st.plotly_chart(fig)
    st.write("")
    st.write("")
    fig = px.pie(studentInfo,values=studentInfo['id_student'],names = studentInfo['gender'],title='Gender Distribution')
    st.plotly_chart(fig)
    st.subheader("We have more men than women among students, but the proportion is not so distant: 55% men and 45% women.")

    st.write("")
    st.write("")
    st.header("Where do the students from the courses come from?")
    st.write("")
    st.write("")
    fig = px.pie(studentInfo,values=studentInfo['id_student'],names =studentInfo['region'] ,title='Regionwise Distribution')
    st.plotly_chart(fig)
    
    st.write("")
    st.write("")
    
    st.header("Mobility of students throughout their training")
    st.write("")
    st.write("")
    _ = studentInfo.groupby(['id_student']).agg({'region':'count'})
    st.write(_['region'].value_counts(normalize=True))
    st.subheader("Most of the students remained in the same region throughout all modules. 12 percent of students lived in more than 1 region.")
    

    st.write("")
    st.write("")
    st.header("Proportion of students with some type of disability")
    st.write("")
    st.write("")
    st.write(studentInfo['disability'].value_counts(normalize=True))
    st.subheader("9.7 percent of students have some type of disability.")
    st.subheader("Disability")
    st.bar_chart(studentInfo['disability'].value_counts())
    
    
    fig = px.pie(studentInfo,values=studentInfo['id_student'],names = studentInfo['disability'],title='Disability Distribution')
    st.plotly_chart(fig)
    dftemp = studentInfo[studentInfo['disability']=='Y']
    fig = px.pie(dftemp,values=dftemp['id_student'],names =dftemp['region'] ,title='Regionwise Disability Distribution')
    st.plotly_chart(fig)
    
    
    
    
    
    
    #-------------------->studentREg
    st.write("")
    st.write("")
    st.header("Student Registration")
    st.write("")
    st.subheader("This file contains information about the time when the student registered for the module presentation. For students who unregistered the date of unregistration is also recorded.")
    st.write("")
    st.write(studentRegistration.head())
    
    st.write("")
    st.write("")
    st.header("When do students register the most?")
    st.write("")
    st.write("")
    fig = px.histogram(studentRegistration ['date_registration'])
    fig.update_xaxes(title='Course days')
    fig.update_yaxes(title='Probability of registration')
    st.plotly_chart(fig)
    st.write(studentRegistration['date_registration'].describe())
    st.subheader("Usually, students register before the course starts. The registration peak starts 50 days before the start, but some students even register later. The last time someone registered was on 167.")
    
    st.write("")
    st.write("")
    st.header("When do more students drop out of courses?")
    st.write("")
    st.write("")
    fig = px.histogram(studentRegistration ['date_unregistration'].dropna())
    fig.update_xaxes(title='Course days')
    fig.update_yaxes(title='Probability of registration')
    st.write(studentRegistration['date_unregistration'].describe())
    st.subheader("About 1/4 of the students drop out of the course before it even starts, which is interesting, because then 1/4 of the student records in courses are cases in which one person did not even attend a class. Peak withdrawals occur in the first 50 days.")
    
    st.write("")
    st.write("")
    st.header("Do students who pass the course register earlier?")
    st.write("")
    st.write("")
    # Join studentInfo with studentRegistration to bring approval data for registration dates
    studentRegistrationInfo = studentInfo.set_index (['code_module', 'code_presentation', 'id_student']). merge (studentRegistration.set_index (['code_module', 'code_presentation', 'id_student']), left_index = True, right_index = True)
    # Return with index columns for normal columns
    studentRegistrationInfo = studentRegistrationInfo.reset_index ()
    
    fig = px.histogram(studentRegistrationInfo.loc [studentRegistrationInfo ['final_result'] == 'Pass'] ['date_registration'])
    fig2 = px.histogram(studentRegistration ['date_registration'])
    
    fig.update_xaxes (title='Course days')
    fig.update_yaxes (title='Probability of registration')
    fig2.update_xaxes (title='Course days')
    fig2.update_yaxes (title='Probability of registration')
    st.subheader('Registration date of approved students vs. all students')
    st.subheader("Approved")
    st.plotly_chart(fig)
    st.subheader("All")
    st.plotly_chart(fig2)

    st.write("")
    st.write("")
    st.write(studentRegistrationInfo.loc [studentRegistrationInfo ['final_result'] == 'Pass'] ['date_registration']. describe ())
    st.write("")
    st.write(studentRegistrationInfo)
    st.subheader("Although different, the distribution of registration dates for approved students does not differ much from the distribution of all students.")

    st.write("")
    st.write("")
    st.header("Do students who pass the course register earlier?")
    st.write("")
    st.write("")
    _ = studentRegistrationInfo.loc[studentRegistrationInfo['final_result'] == 'Fail'].groupby('date_registration').agg({'final_result': 'count'})
    fig1 = px.scatter(_.index, _.final_result)
    
    _ = studentRegistrationInfo.loc[studentRegistrationInfo['final_result'] == 'Withdrawn'].loc[studentRegistrationInfo['date_unregistration'] > 0].groupby('date_registration').agg({'final_result': 'count'})
    fig2 = px.scatter(_.index, _.final_result)
    fig1.update_yaxes (title='Number of students')
    fig1.update_xaxes (title='Course days')
    fig2.update_yaxes (title='Number of students')
    fig2.update_xaxes (title='Course days')
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, horizontal_spacing=0.3, vertical_spacing=0.08,
                            subplot_titles=('Fail', 'Withdrawn'))
        
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=2, col=1)
    
    fig.update_layout(height=800)
    st.plotly_chart(fig)
    
    st.subheader("We chose to study the number of students who dropped out or failed modules vs. the day they registered. We expected to see a larger number of students with positive days, indicating that they registered after classes started. However, the abnormality We noticed was the peak of students who registered around 150 days earlier and dropped out of the course.")

    
    #------------------->studentAssessment
    st.write("")
    st.write("")
    st.header("Student Assessment")
    st.write("")
    st.subheader("This file contains the results of students’ assessments. If the student does not submit the assessment, no result is recorded. The final exam submissions is missing, if the result of the assessments is not stored in the system. ")
    st.write("")
    st.subheader("This is the table that relates students to assessments.")
    st.write("")
    st.write(studentAssessment.head())
    st.subheader("We have 173 lines with invalid values. We will investigate what those values ​​are.")
    
    st.write("")
    st.write(studentAssessment.loc[set(studentAssessment.index)-set(studentAssessment.dropna().index)])
    st.subheader("Invalid values ​​are in the score column. Therefore, the note of these activities can possibly be ignored.")
    st.write("")
    st.write("")
    
    st.write("")
    st.write("")
    st.header("Average grade of activities")
    st.write("")
    st.write("")
    fig = px.histogram(studentAssessment ['score'])
    fig.update_yaxes (title='Number of activities')
    fig.update_xaxes (title='Score')
    st.plotly_chart (fig)
    st.subheader("We have a lot of 100%, but this is not very informative of the weight that these 100 percent carry. We will incorporate weight data from the activities to get a better idea of ​​what the grades are like.")

    # Join studentAssessment with assessments to bring information about weight and type of activities.
    studentAssessmentAssessment = assessments.set_index (['id_assessment']). merge (studentAssessment.set_index (['id_assessment']), left_index = True, right_index = True)
    # Return with index columns for normal columns
    studentAssessmentAssessment = studentAssessmentAssessment.reset_index ()
    
    # Multiplied weight by score to find out how many "points" each student made on average per activity
    studentAssessmentAssessment ['weighted_score'] = (studentAssessmentAssessment ['weight'] / 100) * (studentAssessmentAssessment ['score'] / 100)
    studentAssessmentAssessment ['weighted_score'] *= 100
    
    fig = px.histogram(studentAssessmentAssessment['weighted_score'])
    fig.update_yaxes (title='Number of activities')
    fig.update_xaxes (title='Absolute Score')
    st.plotly_chart (fig)
    st.subheader("We see that when taking weight into account, the distribution changes a lot. This is because of the many activities that have a weight of 0. In addition, we see that activities that are not tests count towards scores below 40.")
    
    st.write("")
    st.write("")
    st.header("Number of submissions per activity")
    st.write("")
    st.write("")
    fig = px.histogram(studentAssessment.groupby (['id_assessment']). agg ({'id_assessment': 'count'}) ['id_assessment'])
    fig.update_yaxes (title='Number of occurrences')
    fig.update_xaxes (title='Number of submissions')
    st.plotly_chart (fig)
    st.write(studentAssessment.groupby (['id_assessment']). agg ({'id_assessment': 'count'}) ['id_assessment']. describe ())
    st.subheader("On average, activities receive 925 submissions.")
    
    
    st.write("")
    st.write("")
    st.header("Result Analysis")
    # Join studentAssessment with assessments to bring information about weight and type of activities.
    studentAssessmentInfo = studentInfo.set_index (['id_student']). merge (studentAssessment.set_index (['id_student']), left_index = True, right_index = True)
    # Return with index columns for normal columns
    studentAssessmentInfo = studentAssessmentInfo.reset_index ()
    fig1 = px.histogram(studentAssessmentInfo.loc [studentAssessmentInfo ['final_result'] == 'Pass'] ['score'])
    fig2 = px.histogram(studentAssessmentInfo.loc [studentAssessmentInfo ['final_result'] == 'Fail'] ['score'])
    fig3 = px.histogram(studentAssessmentInfo.loc [studentAssessmentInfo ['final_result'] == 'Withdrawn'] ['score'])
    fig4 = px.histogram(studentAssessmentInfo.loc [studentAssessmentInfo ['final_result'] == 'Distinction'] ['score'])
    
    fig1.update_yaxes(title='Score probability')
    fig1.update_xaxes(title='Note')
    fig2.update_yaxes(title='Score probability')
    fig2.update_xaxes(title='Note')
    fig3.update_yaxes(title='Score probability')
    fig3.update_xaxes(title='Note')
    fig4.update_yaxes(title='Score probability')
    fig4.update_xaxes(title='Note')
    
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, horizontal_spacing=0.3, vertical_spacing=0.08,
                            subplot_titles=('Pass', 'Fail', 'Withdrawn', 'Distinction'))
        
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.add_trace(fig2['data'][0], row=1, col=2)
    fig.add_trace(fig3['data'][0], row=2, col=1)
    fig.add_trace(fig4['data'][0], row=2, col=2)
    
    fig.update_layout(height=1200)
    st.plotly_chart(fig)
    st.subheader("We see some interesting behaviors in the distribution of students' grades. Firstly, students who have passed the distinction have better grades than the rowers. Then, the passers have a higher proportion of grades greater than 70 in relation to those who fail, and less grades less than 60.When analyzing failing and dropping students, we noticed that dropouts have a higher proportion of better grades than failing ones. This may indicate one of the reasons why a student gives up on a subject, which is poor performance in some key activity.")

    st.subheader("Final Result in given Data")
    #st.bar_chart(studentInfo['final_result'].value_counts())
    fig = px.bar(studentInfo['final_result'].value_counts())
    fig.update_yaxes(title='No. of students')
    fig.update_xaxes(title='Result')
    st.plotly_chart(fig)
    
    st.subheader("Highest Education")
    fig = px.bar(studentInfo['highest_education'].value_counts())
    fig.update_yaxes(title='No. of students')
    fig.update_xaxes(title='Highest Education')
    st.plotly_chart(fig)
    
    st.subheader("Studied Credits")
    fig = px.bar(studentInfo['studied_credits'].value_counts())
    fig.update_yaxes(title='No. of students')
    fig.update_xaxes(title='studied_credits')
    st.plotly_chart(fig)
    




    
if slct == "Classification Model":

    clfr = st.sidebar.selectbox("Classifier",("KNN","Decision Tree","Random Forest","Logistic Regression"))

    if clfr == "Logistic Regression":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        iterations = st.sidebar.slider("Iterations", min_value=5, max_value=75, value=50, step=5, format=None, key='iterations')
        C = st.sidebar.number_input("Regularization Factor",min_value=0.01,max_value=1.0,step=0.01,key='C')
        solver = st.sidebar.radio("Solver",("newton-cg", "lbfgs", "liblinear", "sag", "saga"),key='solver')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = LogisticRegression(max_iter=iterations,solver=solver,C=C)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            #accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            #st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)
    
    if clfr == "Decision Tree":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes",50,200,step=1,key='max_leaf_nodes')
        criterion = st.sidebar.radio("Criterion",("gini", "entropy"),key='criterion')
        max_features = st.sidebar.radio("Features",("auto", "sqrt", "log2"),key='max_features')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes,criterion=criterion,max_features=max_features)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            #accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            #st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)
            
    if clfr == "Random Forest":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        n_estimators = st.sidebar.slider("No.of trees in the forest", min_value=10, max_value=1000, value=100, step=10, format=None, key='n_estimators')
        criterion = st.sidebar.radio("Criterion",("gini", "entropy"),key='criterion')
        max_leaf_nodes = st.sidebar.number_input("Max Leaf Nodes",50,200,step=1,key='max_leaf_nodes')
        random_state= st.sidebar.slider("Random State", min_value=0, max_value=42, value=0, step=1, format=None, key='random_state')
        max_features = st.sidebar.radio("Features",("auto", "sqrt", "log2"),key='max_features')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = DecisionTreeClassifier()
            model = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes,criterion=criterion,max_features=max_features,random_state=random_state)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            #accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            #st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)

    if clfr == "KNN":
        
        x =df.drop(columns=['final_result'])
        y=df['final_result']
        xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
        #parameters
        st.sidebar.subheader("Parameters: ")
        n_neighbors = st.sidebar.slider("No. of neighbors", min_value=3, max_value=20, value=5, step=1, format=None, key='n_neighbors')
        algorithm = st.sidebar.radio("Algorithm",("auto", "ball_tree", "kd_tree", "brute"),key='algorithm')
        leaf_size = st.sidebar.slider("Leaf size", min_value=10, max_value=50, value=30, step=1, format=None, key='leaf_size')
        p = st.sidebar.radio("Power parameter",(1,2),key='p')
        #metrics
        metrics = st.sidebar.multiselect("Select metrics",("Confusion Matrix","ROC","Precision Recall Curve"))
        #classify
        if st.sidebar.button("Classify"):
            model = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=algorithm,leaf_size=leaf_size,p=p)
            model.fit(xtrain,ytrain)
            ypred = model.predict(xtest)
            st.write("Accuracy: ",model.score(xtest,ytest))
            st.write("")
            #accuracy = cross_val_score(model,x, y, scoring='accuracy', cv = 10)
            #st.write("Cross validated Accuracy : " , accuracy.mean())
            st.write("")
            st.write("Model Precision: ", precision_score(ytest,ypred,labels=class_name))
            plot_metrics(metrics)
if slct == "Model Comparison":
    st.subheader("Accuracy vs ML models")
    p = ["KNN","LR","DT","RF"]
    q = [76,86,87,91]

    fig = go.Figure(data=go.Scatter(x=p, y=q))
    fig.update_yaxes (title='Accuracy')
    fig.update_xaxes (title='ML models')

    st.plotly_chart (fig)
    st.write("")
    st.write("")

    st.subheader("AUC vs ML models")    
    p = ["KNN","LR","DT","RF"]
    q = [74,90,90,95]

    fig = go.Figure(data=go.Scatter(x=p, y=q))
    fig.update_yaxes (title='AUC')
    fig.update_xaxes (title='ML models')

    st.plotly_chart (fig)

    st.subheader("AP vs ML models")    
    p = ["KNN","LR","DT","RF"]
    q = [85,95,94,98]

    fig = go.Figure(data=go.Scatter(x=p, y=q))
    fig.update_yaxes (title='AP')
    fig.update_xaxes (title='ML models')

    st.plotly_chart (fig)
