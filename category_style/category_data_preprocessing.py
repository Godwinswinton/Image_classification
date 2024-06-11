#Import pandas for handeling csv files
import pandas as pd

#read the csv file to perform pre-pre=ocessing
df = pd.read_csv("category_style/data/list_attr_celeba.csv")

#Select few features where to apply style condition
df_1 = df[['Young', 'Eyeglasses', 'No_Beard', 'Attractive', 'Male']]

#apply style category for based on the selected features
def categorize_appearance(row):
    if row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Casual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Formal Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == 1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Formal Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == 1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == 1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Intellectual Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == 1 and row['Male'] == -1:
        return 'Formal Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == 1:
        return 'Formal Appearance'
    elif row['Young'] == -1 and row['Eyeglasses'] == -1 and row['No_Beard'] == -1 and row['Attractive'] == -1 and row['Male'] == -1:
        return 'Formal Appearance'

# Apply function to create 'style' column
df_1['style'] = df_1.apply(categorize_appearance, axis=1)

#Perform one-hot encoding for the style feature
style_dummies = pd.get_dummies(df_1['style'])
style_dummies = style_dummies.applymap(lambda x: 1 if x else 0)

#save the csv file
style_dummies.to_csv('attribute.csv', index=False)