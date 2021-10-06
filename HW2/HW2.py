import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def my_merge(df1, df2):

    return pd.merge(
        left=df1,
        right=df2,
        how="outer",    # choose the all row that two df have
        on="name",      # combine two df through 'name' col
        indicator=False  # do not show where the row come from
    )


hw2_data = pd.read_csv("HW2_data.csv", sep=",",
                       names=['name', 'typename', 'data'])

sex_data = hw2_data[(hw2_data.typename == 'Sex')]
age_data = hw2_data[(hw2_data.typename == 'Age')]
hr_data = hw2_data[(hw2_data.typename == 'HR')]
height_data = hw2_data[(hw2_data.typename == 'Height')]
weight_data = hw2_data[(hw2_data.typename == 'Weight')]
bp_data = hw2_data[(hw2_data.typename == 'BP')]

result = my_merge(sex_data, age_data)
result = my_merge(result, hr_data)
result = my_merge(result, height_data)
result = my_merge(result, weight_data)
result = my_merge(result, bp_data)
result = result.drop(['typename_x', 'typename_y'], axis=1)

colname = ['Name', 'Sex', 'Age', 'HR', 'Height', 'Weight', 'BP']
result = result.set_axis(colname, axis=1, inplace=False)

for i in colname[2:]:
    result[i] = pd.to_numeric(result[i], errors='ignore')

for i in colname[2:]:
    result.loc['50', i] = result[i].mean(skipna=True)
    result[i] = result[i].fillna(result[i].mean(skipna=True))
result.loc['50', 'Name'] = 'Mean'

pd.set_option("display.precision", 2)
print(result)
print("Above result is for Q1 and Q2")
print("-----------------------------------------------")
for i in colname[2:]:
    print("max %s is : %d, below is the list of people."
          % (i, result[i].max()))
    for j in list(result.loc[result[i] == result[i].max(), 'Name']):
        print(j)
print("Above result is for Q3")
print("-----------------------------------------------")

m_plot = result.loc[result['Sex'] == 'M'].plot.scatter(x='Height',
                                                       y='Weight',
                                                       c='DarkBlue')
f_plot = result.loc[result['Sex'] == 'F'].plot.scatter(x='Height',
                                                       y='Weight',
                                                       c='Red',
                                                       ax=m_plot,
                                                       title="Height and Weight of 2 Sex for Q4")
plt.show()
# ------------------------------------------------------
age_statstic = [0, 0, 0, 0, 0, 0, 0, 0]
for i in result["Age"][0:50]:
    if (1 <= int(i) < 11):
        age_statstic[0] += 1
    elif (11 <= int(i) < 21):
        age_statstic[1] += 1
    elif (21 <= int(i) < 31):
        age_statstic[2] += 1
    elif (31 <= int(i) < 41):
        age_statstic[3] += 1
    elif (41 <= int(i) < 51):
        age_statstic[4] += 1
    elif (51 <= int(i) < 61):
        age_statstic[5] += 1
    elif (61 <= int(i) < 71):
        age_statstic[6] += 1
    else:
        age_statstic[7] += 1

age_df = pd.DataFrame({'data': age_statstic},
                      index=['1~10', '11~20', '21~30', '31~40', '41~50', '51~60', '61~70', '71~80'])
age_barplot = age_df.plot.bar(
    y='data', rot=0, legend=False, title="Age for Q5")
plt.show()
# ------------------------------------------------------
age_pieplot = age_df.plot.pie(y='data', figsize=(5, 5),
                                autopct='%1.1f%%', legend=False, title="Age for Q5")
plt.show()
# ------------------------------------------------------
sex_df = pd.DataFrame({'data': [result.loc[result["Sex"] == 'M'].count()["Name"], result.loc[result["Sex"] == 'F'].count()["Name"]]},
                      index=['M', 'F'])
sex_pieplot = sex_df.plot.pie(y='data', figsize=(5, 5),
                                autopct='%1.1f%%', legend=False, title="Sex for Q6")
plt.show()
