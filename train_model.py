import pandas as pd

# Télécharger automatiquement le dataset Kaggle
url = "https://github.com/sharmaroshan/Students-Performance-Analytics/raw/master/StudentsPerformance.csv"
df = pd.read_csv(url)

# Transformer en vos 4 variables
df['score_moyen'] = (df['reading score'] + df['writing score']) / 2
df['Revision'] = (df['test preparation course'] == 'completed').astype(int)
df['Confiance'] = (df['score_moyen'] >= 60).astype(int)
df['Malade'] = (df['lunch'] == 'free/reduced').astype(int)
df['Succes'] = (df['score_moyen'] >= 50).astype(int)

# Garder seulement vos 4 variables
df_final = df[['Revision', 'Confiance', 'Malade', 'Succes']]

# Sauvegarder
df_final.to_csv('ma_base_donnees.csv', index=False)

print(f"✅ {len(df_final)} étudiants prêts !")
print(df_final.head(10))