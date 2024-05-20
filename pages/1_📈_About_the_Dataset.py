import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from scipy.stats import mannwhitneyu

# Streamlit page configuration
st.set_page_config(
    page_title='About the Dataset',
    page_icon=':chart_with_upwards_trend:',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title(':chart_with_upwards_trend: The Dataset')

# load the dataset
data = pd.read_csv('data/pima_indian_diabetes.csv')
data['Outcome'].replace({0:'No Diabetes', 1:'Diabetes'}, inplace=True)

# handle the inaccurate data --- physiological data impossible with value of '0'
data['Glucose'].replace({0: np.nan}, inplace=True)
data['BloodPressure'].replace({0: np.nan}, inplace=True)
data['SkinThickness'].replace({0: np.nan}, inplace=True)
data['Insulin'].replace({0: np.nan}, inplace=True)
data['BMI'].replace({0: np.nan}, inplace=True)

pregnancies = {'Pregnancies': 'Number of Pregnancies'}
glucose = {'Glucose': 'Blood Glucose Concentration'}
blood_pressure = {'BloodPressure': 'Blood Pressure'}
skin_thickness = {'SkinThickness': 'Skin Thickness'}
insulin = {'Insulin': 'Insulin Level'}
bmi = {'BMI': 'Body Mass Index'}
dpf = {'DiabetesPedigreeFunction': 'Diabetes Pedigree Function'}
age = {'Age': 'Patient Age'}

option = {'Pregnancies': pregnancies,
          'Glucose': glucose,
          'Blood Pressure': blood_pressure,
          'Skin Thickness': skin_thickness,
          'Insulin Level': insulin,
          'BMI': bmi,
          'Diabetes Pedigree Function': dpf,
          'Patient Age': age}

def plot_histogram(var):
    for k,v in var.items():
        fig = px.histogram(data, x=k, height=600, nbins=50, title=f'Distribution of {v}')
        fig.update_layout(template='plotly_dark', xaxis=dict(showticklabels=True))
        fig.update_layout(xaxis_title=v, yaxis_title='Frequency', margin=dict(l=50, r=50, t=50, b=50))
        fig.update_traces(hovertemplate=f'{v}: %{{x}}<br>Frequency: %{{y}}')
        container2.plotly_chart(fig, theme='streamlit', use_container_width=True)

def plot_scatter(x_var,y_var):
    custom_colors = ['#C0369D', '#FCA636']
    x = option[x_var]
    y = option[y_var]
    x_var = list(x.keys())[0]
    y_var = list(y.keys())[0]
    scatterplot = px.scatter(data, x=x_var, y=y_var, color='Outcome', height=600, color_discrete_sequence=custom_colors, title=f'Correlation between {x_var} and {y_var}')
    scatterplot.update_layout(template='plotly_dark')
    container4.plotly_chart(scatterplot, theme='streamlit', use_container_width=True)

def compare_groups(var):
    colors = ['#C0369D','#FCA636']
    # visualization
    for k,v in var.items():
        sns.set_style("darkgrid")
        figure, axes = plt.subplots(1,2, figsize=(15,5), gridspec_kw={'width_ratios':[2,1]})
        figure.patch.set_facecolor('black')
        sns.histplot(data=data, x=data[k], hue=data['Outcome'], kde=True, ax=axes[0], palette=colors)
        axes[0].set_xlabel('', color='white')
        axes[0].set_ylabel('Frequency', color='white')
        sns.boxplot(data=data, x=data['Outcome'], y=data[k], ax=axes[1], palette=colors)
        axes[1].set_ylabel(' ')
        axes[1].set_xticks([0,1])
        axes[1].set_xlabel(' ')
        #axes[1].grid(alpha=0.4)
        for tick_label in axes[0].get_xticklabels():
            tick_label.set_color('white')
        for tick_label in axes[0].get_yticklabels():
            tick_label.set_color('white')
        for tick_label in axes[1].get_xticklabels():
            tick_label.set_color('white')
        for tick_label in axes[1].get_yticklabels():
            tick_label.set_color('white')
        figure.suptitle(f'Distribution of {v} between Patients with and without Diabetes', fontsize=15, color='white')
        plt.tight_layout(pad=1)
        container5.pyplot(figure)
        container5.markdown(F'#### Summary Statistic of Variable {k} between Groups')
        group_data = data.groupby('Outcome')
        container5.table(group_data[k].describe())


container = st.container(border=True)
container.markdown('### :one: Pima Indians Diabetes Dataset')
container.markdown('#### Dataset Information')
container.write('''The original Pima Indians diabetes dataset from UCI machine learning repository is a binary classification dataset. 
                Several constraints were placed on the selection of instances from a larger database. 
                In particular, all patients here are females at least 21 years old of Pima Indian heritage. 
                The dataset is utilized as it is from the UCI repository.''')
container.markdown('#### Variable Descirption')
container.write('1. Pregancies: Number of times pregnant.')
container.write('2. Glucose: Plasma glucose conecentration at 2 hours in an oral glucose tolerance test (GTIT).')
container.write('3. BloodPressure: Diastolic Blood Pressure (mm/Hg).')
container.write('4. SkinThickness: Tricep skin fold thickness (mm).')
container.write('5. Insulin: 2 hour serum insulin (µh/mL).')
container.write('6. BMI: Body mass index (weight in kg/Height in m).')
container.write('7. DiabetesPedigreeFunction: Diabetes pedigree function.')
container.write('8. Age: Age in years.')
container.write('9. Outcome: Binary value indicating non-diabetic or diabetic.')

container.markdown('#### ')
container.dataframe(data, use_container_width=True)
container.markdown('#### Acknowledgement')
container.markdown('''Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). 
                   [Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/) 
                   <i>In Proceedings of the Symposium on Computer Applications and Medical Care </i> (pp. 261--265). IEEE Computer Society Press.''', unsafe_allow_html=True)
container.download_button(
    label='Download Dataset',
    data=data.to_csv().encode('utf-8'),
    file_name=f'Pima_Indian_Diabetes_Dataset.csv',
    mime='text/csv'
)

container1 = st.container(border=True)
container1.markdown('### :two: Proportion of Patients with Diabetes and without Diabetes')
custom_colors = ['#C0369D', '#FCA636']
outcome = data['Outcome'].value_counts()
pie = px.pie(outcome, values=outcome.values, names=['Without diabetes', 'Diabetes'], color_discrete_sequence=custom_colors, height=600)
pie.update_layout(template='plotly_dark')
container1.plotly_chart(pie, theme='streamlit', use_container_width=True)

container2 = st.container(border=True)
container2.markdown('### :three: Distribution of Variables in the Dataset')
container2.markdown('### ')
user_option = container2.selectbox('Select a variable', option.keys())
plot_histogram(option[user_option])

container3 = st.container(border=True)
container3.markdown('### :four: Correlation between Variables')
container3.markdown('### ')
correlation = data[list(data.columns[:-1])].corr()
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')  # Set the figure background color
ax.set_facecolor('black')         # Set the axes background color
sns.set_style('dark')
heatmap = sns.heatmap(correlation, annot=True, annot_kws={'fontsize':5}, cbar=False, cmap='plasma', xticklabels=option.keys(), yticklabels=option.keys(), ax=ax)
heatmap.tick_params(axis='x', labelsize=5)  
heatmap.tick_params(axis='y', labelsize=5)
# Set the tick labels color to white
for tick_label in ax.get_xticklabels():
    tick_label.set_color('white')
for tick_label in ax.get_yticklabels():
    tick_label.set_color('white')
plt.show()
container3.pyplot(plt)

container4 = st.container(border=True)
container4.markdown('### :five: Correlation between Predictor Variables')
container4.markdown('### ')
col1, col2 = container4.columns(2)
x_var = col1.selectbox('Select variable x', option.keys())
y_var = col2.selectbox('Select variable y', option.keys())
plot_scatter(x_var,y_var)

container5 = st.container(border=True)
container5.markdown('### :six: Comparison of Variables Between Patients with and without Diabetes')
container5.markdown('### ')
user_option_compare = container5.selectbox('Select a variable', key='compare', options=option.keys())
compare_groups(option[user_option_compare])

container6 = st.container(border=True)
container6.markdown('### :seven: Statistical Comparison of Variables between Diabetic and Non-diabetic Groups')
container6.markdown('### ')
summary_stat = pd.read_csv('data/summary_stat.csv')
container6.dataframe(summary_stat, use_container_width=True)






