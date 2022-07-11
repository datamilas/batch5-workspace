



df <- read.csv(file = 'RandomForestClassifier_age_df.csv')#,row.names="age")

df
pairwise.prop.test(x = df[,c("Yes")], df[,c("Yes")]+df[,c("No")])
#df[,c("medical_specialty")]
