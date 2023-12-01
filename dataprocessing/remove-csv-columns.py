import pandas as pd 

f=pd.read_csv("Web_Scrapped_websites.csv")
keep_col = ['Website','Trustworthiness']
new_f = f[keep_col]
new_f.to_csv("newFile.csv", index=False)